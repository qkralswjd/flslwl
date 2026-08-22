"""1단계 자동 레벨링 메인 상태머신.

────────────────────────────────────────────────────────────
전체 흐름
────────────────────────────────────────────────────────────

    [IDLE]
       │ start() 호출
       ▼
    [TELEPORTING]  ← 텔레포트 키 입력 → 목적지(허수아비) OCR → 클릭
       │ 텔레포트 성공
       ▼
    [MOVING_TO_DUMMY]  ← 허수아비 좌표로 Pico 클릭 이동
       │ 이동 완료 (타임아웃)
       ▼
    [ATTACKING_DUMMY]  ← 허수아비 반복 클릭 공격
       │ OCR 레벨 >= 5
       ▼
    [PATROLLING]  ← 웨이포인트 순환 이동
       │ 웨이포인트 도착
       ▼
    [HUNTING]  ← 기존 tracker가 적 탐지 + NearestNeighborTracker 공격
       │ 적 없음 → PATROLLING
       │ 아데나 감지 → LOOTING
       ▼
    [LOOTING]  ← 아데나 클릭
       │ 완료 → HUNTING or PATROLLING
       ▼
    [DONE]  ← 레벨 >= 15

어느 상태에서든:
    레벨 >= 15 → DONE
    stop() 호출 → IDLE

────────────────────────────────────────────────────────────
"""

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from automation.level_reader    import LevelReader
from automation.loot_detector   import LootDetector
from automation.teleport_handler import TeleportHandler
from automation.waypoint_mover  import WaypointMover

logger = logging.getLogger("hunting_sm")


class HuntingState(Enum):
    IDLE            = auto()   # 시작 전
    TELEPORTING     = auto()   # 텔레포트 실행 중
    MOVING_TO_DUMMY = auto()   # 허수아비 위치로 이동 중
    ATTACKING_DUMMY = auto()   # 허수아비 공격 중 (→5레벨)
    PATROLLING      = auto()   # 웨이포인트 순환 이동 중
    HUNTING         = auto()   # 몬스터 탐지 + 공격 중
    LOOTING         = auto()   # 아데나 줍기 중
    DONE            = auto()   # 완료 (15레벨)


class HuntingStateMachine:
    """1단계 자동 레벨링 전체 흐름 관리."""

    def __init__(self, config: dict, pico_worker, frame_grabber: Callable[[], np.ndarray]):
        """
        Args:
            config: config_automation.json 내용
            pico_worker: PicoSerialWorker 인스턴스
            frame_grabber: capturer.grab() 반환 함수
        """
        self.cfg           = config
        self.pico          = pico_worker
        self.grab          = frame_grabber

        self.state         = HuntingState.IDLE
        self._entered_at   = time.time()

        # ── 레벨 리더 ──────────────────────────────────────────────
        lvl_cfg = config.get("level_ocr", {})
        self.level_reader = LevelReader(
            region          = lvl_cfg.get("region", {"x":0,"y":0,"width":100,"height":30}),
            read_interval_s = lvl_cfg.get("read_interval_s", 2.0),
        )
        self.target_level_1 = lvl_cfg.get("target_level_1", 5)   # 허수아비 → 사냥터
        self.target_level_2 = lvl_cfg.get("target_level_2", 15)  # 종료

        # ── 텔레포트 핸들러 ────────────────────────────────────────
        tp_cfg = config.get("teleport", {})
        cap_cfg = config.get("capture_offset", {"x": 0, "y": 0})
        cap_offset = (cap_cfg.get("x", 0), cap_cfg.get("y", 0))

        self.teleporter = TeleportHandler(
            key                 = tp_cfg.get("key", "F1"),
            destination_region  = tp_cfg.get("destination_region",
                                             {"x":400,"y":200,"width":300,"height":400}),
            destination_text    = tp_cfg.get("destination_text", "허수아비"),
            capture_offset      = cap_offset,
            wait_after_key_ms   = tp_cfg.get("wait_after_key_ms", 800),
            wait_after_click_ms = tp_cfg.get("wait_after_click_ms", 2000),
        )

        # ── 허수아비 공격 설정 ─────────────────────────────────────
        dummy_cfg = config.get("dummy", {})
        self.dummy_coord       = (dummy_cfg.get("attack_coord", {}).get("x", 960),
                                  dummy_cfg.get("attack_coord", {}).get("y", 540))
        self.dummy_attack_interval = dummy_cfg.get("attack_interval_ms", 500) / 1000.0
        self._dummy_move_done  = False
        self._last_dummy_atk   = 0.0
        self._dummy_move_timeout = dummy_cfg.get("move_timeout_ms", 3000) / 1000.0
        self._dummy_move_start = 0.0

        # ── 웨이포인트 무버 ────────────────────────────────────────
        wp_cfg = config.get("waypoints", [])
        self.mover = WaypointMover(
            waypoints      = wp_cfg if wp_cfg else [{"x":960,"y":540,"label":"기본","wait_ms":1000}],
            capture_offset = cap_offset,
            move_timeout_ms = config.get("move_timeout_ms", 4000),
            loop           = True,
        )

        # ── 아데나 탐지기 ─────────────────────────────────────────
        loot_cfg = config.get("loot", {})
        roi_cfg  = config.get("roi_offset", {"x":0,"y":0})
        self.loot_detector = LootDetector(
            scan_region    = loot_cfg.get("scan_region",
                                          {"x":0,"y":0,"width":1440,"height":780}),
            loot_keywords  = loot_cfg.get("keywords", ["아데나","Adena"]),
            scan_interval_s = loot_cfg.get("scan_interval_s", 0.5),
            roi_offset     = (roi_cfg.get("x",0), roi_cfg.get("y",0)),
            capture_offset = cap_offset,
        )

        # ── 루팅 상태 ─────────────────────────────────────────────
        self._loot_targets: list = []
        self._loot_idx            = 0
        self._loot_click_interval = loot_cfg.get("click_interval_ms", 400) / 1000.0
        self._last_loot_click     = 0.0
        self._loot_timeout_ms     = loot_cfg.get("timeout_ms", 3000)
        self._loot_start_t        = 0.0

        # ── 사냥 상태 ─────────────────────────────────────────────
        hunt_cfg = config.get("hunt", {})
        self._hunt_idle_timeout_ms = hunt_cfg.get("idle_timeout_ms", 3000)
        self._last_enemy_seen_t    = time.time()

        # ── 통계 ──────────────────────────────────────────────────
        self.kills    = 0
        self.start_time = time.time()

    # ── 공개 API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """자동 레벨링 시작."""
        logger.info("=" * 60)
        logger.info("[HuntingSM] ▶ 1단계 자동 레벨링 시작")
        logger.info(f"  목표: {self.target_level_1}레벨(허수아비) → {self.target_level_2}레벨(완료)")
        logger.info("=" * 60)
        self._enter(HuntingState.TELEPORTING)

    def stop(self) -> None:
        """중지 및 IDLE 복귀."""
        logger.info("[HuntingSM] ■ 중지")
        self._enter(HuntingState.IDLE)

    @property
    def state_name(self) -> str:
        return self.state.name

    @property
    def elapsed_min(self) -> float:
        return (time.time() - self.start_time) / 60.0

    def update(self, frame: np.ndarray, enemies: list) -> None:
        """매 detection tick마다 호출.

        Args:
            frame:   현재 캡처 프레임
            enemies: NearestNeighborTracker.update() 반환 Enemy 리스트
        """
        if self.state == HuntingState.IDLE:
            return

        # ── 레벨 체크 (어느 상태에서든) ───────────────────────────
        level = self.level_reader.read(frame)
        if level is not None and level >= self.target_level_2:
            logger.info(f"[HuntingSM] 🎉 목표 레벨({self.target_level_2}) 달성! 자동 레벨링 완료")
            self._enter(HuntingState.DONE)
            return

        # ── 상태별 처리 ───────────────────────────────────────────
        if self.state == HuntingState.TELEPORTING:
            self._update_teleporting()

        elif self.state == HuntingState.MOVING_TO_DUMMY:
            self._update_moving_to_dummy()

        elif self.state == HuntingState.ATTACKING_DUMMY:
            self._update_attacking_dummy(frame, level)

        elif self.state == HuntingState.PATROLLING:
            self._update_patrolling(frame, enemies)

        elif self.state == HuntingState.HUNTING:
            self._update_hunting(frame, enemies)

        elif self.state == HuntingState.LOOTING:
            self._update_looting(frame)

        elif self.state == HuntingState.DONE:
            pass

    # ── 상태별 처리 ───────────────────────────────────────────────────

    def _update_teleporting(self) -> None:
        """텔레포트 실행 (블로킹 1회)."""
        success = self.teleporter.execute(self.grab, self.pico)
        if success:
            logger.info("[HuntingSM] 텔레포트 성공 → 허수아비로 이동")
            self._enter(HuntingState.MOVING_TO_DUMMY)
        else:
            logger.warning("[HuntingSM] 텔레포트 실패 — 재시도")
            time.sleep(1.0)
            # 상태 유지 → 다음 tick에 재시도

    def _update_moving_to_dummy(self) -> None:
        """허수아비 위치로 이동."""
        now = time.time()
        if not self._dummy_move_done:
            # 처음 진입 시 클릭
            ax = self.dummy_coord[0]
            ay = self.dummy_coord[1]
            logger.info(f"[HuntingSM] 허수아비 위치로 이동: ({ax},{ay})")
            self.pico.click(ax, ay)
            self._dummy_move_start = now
            self._dummy_move_done  = True
            return

        # 이동 타임아웃 후 도착으로 간주
        if now - self._dummy_move_start >= self._dummy_move_timeout:
            logger.info("[HuntingSM] 허수아비 도착 → 공격 시작")
            self._dummy_move_done = False
            self._enter(HuntingState.ATTACKING_DUMMY)

    def _update_attacking_dummy(self, frame: np.ndarray, level: Optional[int]) -> None:
        """허수아비를 반복 클릭 공격. 5레벨 되면 사냥터로."""
        now = time.time()

        # 레벨 달성 체크
        if level is not None and level >= self.target_level_1:
            logger.info(f"[HuntingSM] ⭐ {self.target_level_1}레벨 달성! 사냥터로 이동")
            self.mover.start()
            self._enter(HuntingState.PATROLLING)
            return

        # 주기적 공격 클릭
        if now - self._last_dummy_atk >= self.dummy_attack_interval:
            ax, ay = self.dummy_coord
            self.pico.click(ax, ay)
            self._last_dummy_atk = now
            elapsed = int(now - self._entered_at)
            logger.debug(
                f"[HuntingSM] 허수아비 공격 중... "
                f"레벨={level or '?'} 경과={elapsed}s"
            )

    def _update_patrolling(self, frame: np.ndarray, enemies: list) -> None:
        """웨이포인트 순환. 적이 보이면 HUNTING으로."""
        # 적 발견 시 즉시 사냥 전환
        if enemies:
            logger.info(f"[HuntingSM] 적 {len(enemies)}명 발견 → HUNTING")
            self._last_enemy_seen_t = time.time()
            self._enter(HuntingState.HUNTING)
            return

        # 웨이포인트 이동
        status = self.mover.tick(self.pico)
        if status == "ARRIVED":
            label = self.mover.current_label
            logger.info(f"[HuntingSM] 웨이포인트 '{label}' 도착")

    def _update_hunting(self, frame: np.ndarray, enemies: list) -> None:
        """몬스터 공격 + 아데나 탐지. 적 없으면 PATROLLING."""
        now = time.time()

        # 적 있으면 타임스탬프 갱신 (NearestNeighborTracker가 공격 처리)
        if enemies:
            self._last_enemy_seen_t = now
        else:
            # 적 없이 일정 시간 경과 → 순찰로 복귀
            idle_ms = (now - self._last_enemy_seen_t) * 1000.0
            if idle_ms >= self._hunt_idle_timeout_ms:
                logger.info("[HuntingSM] 적 없음 → PATROLLING 복귀")
                self._enter(HuntingState.PATROLLING)
                return

        # 아데나 탐지
        loot = self.loot_detector.find(frame)
        if loot:
            logger.info(f"[HuntingSM] 아데나 {len(loot)}개 발견 → LOOTING")
            self._loot_targets = list(loot)
            self._loot_idx     = 0
            self._loot_start_t = now
            self._enter(HuntingState.LOOTING)

    def _update_looting(self, frame: np.ndarray) -> None:
        """아데나를 하나씩 클릭."""
        now = time.time()

        # 타임아웃
        elapsed_ms = (now - self._loot_start_t) * 1000.0
        if elapsed_ms >= self._loot_timeout_ms:
            logger.info("[HuntingSM] 루팅 타임아웃 → HUNTING 복귀")
            self._enter(HuntingState.HUNTING)
            return

        # 모두 클릭 완료
        if self._loot_idx >= len(self._loot_targets):
            logger.info("[HuntingSM] 루팅 완료 → HUNTING 복귀")
            self.kills += 1
            self._enter(HuntingState.HUNTING)
            return

        # 클릭 간격 체크
        if now - self._last_loot_click < self._loot_click_interval:
            return

        target = self._loot_targets[self._loot_idx]
        sx, sy, text, conf = target
        logger.info(f"[HuntingSM] 아데나 클릭: ({sx},{sy}) '{text}'")
        self.pico.click(sx, sy)
        self._last_loot_click = now
        self._loot_idx += 1

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _enter(self, new_state: HuntingState) -> None:
        prev = self.state
        self.state       = new_state
        self._entered_at = time.time()
        logger.info(f"[HuntingSM] {prev.name} → {new_state.name}")

    def get_status(self) -> dict:
        """UI 표시용 현재 상태 딕셔너리."""
        level = self.level_reader.get_cached()
        return {
            "state":        self.state.name,
            "level":        level or "?",
            "kills":        self.kills,
            "elapsed_min":  round(self.elapsed_min, 1),
            "waypoint":     self.mover.current_label
                            if self.state in (HuntingState.PATROLLING, HuntingState.HUNTING)
                            else "-",
        }
