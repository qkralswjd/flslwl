"""1단계 자동 레벨링 메인 상태머신 (요정 캐릭터 전용).

────────────────────────────────────────────────────────────
전체 흐름 (요정 기준)
────────────────────────────────────────────────────────────

    [IDLE]
       │ start() 호출
       ▼
    [USE_SCROLL_DUMMY]
       │ F6(말하는 두루마리) 키 입력
       │ 목적지 창 OCR → "허수아비 수련장" 클릭
       │ 텔레포트 완료 대기
       ▼
    [MOVE_TO_DUMMY]
       │ 허수아비 좌표로 Pico 클릭 이동
       │ move_timeout 후 도착 간주
       ▼
    [ATTACKING_DUMMY]   ← 허수아비 반복 클릭 공격
       │   HP < 50% → F5 물약 자동 사용 (쿨타임 3초)
       │   OCR 레벨 >= target_level_dummy(5) → USE_SPEED_POTION
       ▼
    [USE_SPEED_POTION]
       │ F9(속도향상물약) 키 입력 후 잠깐 대기
       ▼
    [MOVE_TO_HUNT_ZONE]
       │ hunt_waypoints 순환 이동 시작
       │ 이동 중에도 HP 체크 → F5 물약
       ▼
    [HUNTING_10]
       │ 기존 tracker가 적 탐지 + NearestNeighborTracker 공격
       │ HP < 50% → F5 물약 자동 사용
       │ 아데나 감지 → LOOTING → 복귀
       │ OCR 레벨 >= target_level_hunt(10) → DONE_PHASE1
       ▼
    [DONE_PHASE1]  ← 1단계 완료

어느 상태에서든:
    stop() 호출 → IDLE
    HP < 50% → F5 물약 (쿨타임 체크)

────────────────────────────────────────────────────────────
"""

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from automation.hp_reader      import HpReader
from automation.level_reader   import LevelReader
from automation.loot_detector  import LootDetector
from automation.teleport_handler import TeleportHandler
from automation.waypoint_mover import WaypointMover

logger = logging.getLogger("hunting_sm")


class HuntingState(Enum):
    IDLE               = auto()   # 시작 전
    USE_SCROLL_DUMMY   = auto()   # F6 말하는 두루마리 → 허수아비 수련장
    MOVE_TO_DUMMY      = auto()   # 허수아비 위치로 이동 중
    ATTACKING_DUMMY    = auto()   # 허수아비 공격 중 (→ Lv.target_level_dummy)
    USE_SPEED_POTION   = auto()   # F9 속도향상물약 사용
    MOVE_TO_HUNT_ZONE  = auto()   # 사냥터로 이동 중
    HUNTING_10         = auto()   # 사냥터에서 사냥 (→ Lv.target_level_hunt)
    LOOTING            = auto()   # 아데나 줍기 중
    DONE_PHASE1        = auto()   # 1단계 완료


class HuntingStateMachine:
    """요정 1단계 자동 레벨링 전체 흐름 관리.

    config_automation.json 구조:
        keys.potion        : 물약 키 (기본 "F5")
        keys.scroll        : 말하는 두루마리 키 (기본 "F6")
        keys.speed_potion  : 속도향상물약 키 (기본 "F9")
        keys.potion_cooldown_ms : 물약 쿨타임 ms (기본 3000)
        hp_bar.region      : HP 바 영역 {"x","y","width","height"}
        hp_bar.threshold_pct: HP 경고 기준 % (기본 50.0)
        hp_bar.read_interval_s: HP 읽기 간격 (기본 0.5)
        level_ocr.region   : 레벨 OCR 영역
        level_ocr.target_level_dummy : 허수아비 종료 레벨 (기본 5)
        level_ocr.target_level_hunt  : 1단계 완료 레벨 (기본 10)
        scroll_dummy.destination_region : 두루마리 목적지 창 영역
        scroll_dummy.destination_text   : 목적지 텍스트 (기본 "허수아비")
        scroll_dummy.wait_after_key_ms  : 키 후 대기 ms
        scroll_dummy.wait_after_click_ms: 클릭 후 대기 ms
        dummy.attack_coord  : 허수아비 클릭 좌표 {"x","y"}
        dummy.attack_interval_ms : 공격 간격 ms (기본 500)
        dummy.move_timeout_ms    : 이동 타임아웃 ms (기본 3000)
        hunt_waypoints.points    : 사냥터 웨이포인트 리스트
        hunt_waypoints.move_timeout_ms : 웨이포인트 이동 타임아웃
        loot.*               : 아데나 줍기 설정
    """

    def __init__(
        self,
        config: dict,
        pico_worker,
        frame_grabber: Callable[[], np.ndarray],
    ):
        """
        Args:
            config       : config_automation.json 내용
            pico_worker  : PicoSerialWorker 인스턴스
            frame_grabber: capturer.grab() 등 프레임 반환 함수
        """
        self.cfg  = config
        self.pico = pico_worker
        self.grab = frame_grabber

        self.state       = HuntingState.IDLE
        self._entered_at = time.time()

        # ── 캡처 오프셋 ────────────────────────────────────────────────
        cap_cfg    = config.get("capture_offset", {"x": 0, "y": 0})
        roi_cfg    = config.get("roi_offset",     {"x": 0, "y": 0})
        self._cap_offset = (cap_cfg.get("x", 0), cap_cfg.get("y", 0))
        self._roi_offset = (roi_cfg.get("x", 0), roi_cfg.get("y", 0))

        # ── 키 설정 ────────────────────────────────────────────────────
        keys_cfg                = config.get("keys", {})
        self.key_potion         = keys_cfg.get("potion",        "F5")
        self.key_scroll         = keys_cfg.get("scroll",        "F6")
        self.key_speed_potion   = keys_cfg.get("speed_potion",  "F9")
        self._potion_cooldown   = keys_cfg.get("potion_cooldown_ms", 3000) / 1000.0
        self._last_potion_t     = 0.0   # 마지막 물약 사용 시각

        # ── HP 리더 ────────────────────────────────────────────────────
        hp_cfg = config.get("hp_bar", {})
        self.hp_reader = HpReader(
            region          = hp_cfg.get("region", {"x":0,"y":0,"width":200,"height":10}),
            threshold_pct   = hp_cfg.get("threshold_pct", 50.0),
            read_interval_s = hp_cfg.get("read_interval_s", 0.5),
        )

        # ── 레벨 리더 ──────────────────────────────────────────────────
        lvl_cfg = config.get("level_ocr", {})
        self.level_reader = LevelReader(
            region          = lvl_cfg.get("region", {"x":0,"y":0,"width":80,"height":25}),
            read_interval_s = lvl_cfg.get("read_interval_s", 2.0),
        )
        self.target_level_dummy = lvl_cfg.get("target_level_dummy", 5)
        self.target_level_hunt  = lvl_cfg.get("target_level_hunt",  10)

        # ── 말하는 두루마리 텔레포트 핸들러 ───────────────────────────
        sd_cfg = config.get("scroll_dummy", {})
        self.scroll_teleporter = TeleportHandler(
            key                 = self.key_scroll,
            destination_region  = sd_cfg.get("destination_region",
                                             {"x":400,"y":150,"width":300,"height":400}),
            destination_text    = sd_cfg.get("destination_text", "허수아비"),
            capture_offset      = self._cap_offset,
            wait_after_key_ms   = sd_cfg.get("wait_after_key_ms",   800),
            wait_after_click_ms = sd_cfg.get("wait_after_click_ms", 3000),
        )

        # ── 허수아비 드래그 공격 설정 ──────────────────────────────────
        dummy_cfg = config.get("dummy", {})
        drag_from_cfg            = dummy_cfg.get("drag_from", {"x": 960, "y": 600})
        drag_to_cfg              = dummy_cfg.get("drag_to",   {"x": 960, "y": 400})
        self.dummy_drag_from     = (drag_from_cfg.get("x", 960), drag_from_cfg.get("y", 600))
        self.dummy_drag_to       = (drag_to_cfg.get("x",   960), drag_to_cfg.get("y",   400))
        self.dummy_drag_steps    = dummy_cfg.get("drag_steps", 8)
        self.dummy_atk_interval  = dummy_cfg.get("attack_interval_ms", 500) / 1000.0
        self.dummy_move_timeout  = dummy_cfg.get("move_timeout_ms", 3000) / 1000.0
        self._dummy_move_done    = False
        self._dummy_move_start   = 0.0
        self._last_dummy_atk     = 0.0

        # ── 사냥터 웨이포인트 무버 ─────────────────────────────────────
        hwp_cfg   = config.get("hunt_waypoints", {})
        wp_points = hwp_cfg.get("points", [{"x":960,"y":400,"label":"사냥터-A","wait_ms":2000}])
        wp_timeout = hwp_cfg.get("move_timeout_ms", 5000)
        self.hunt_mover = WaypointMover(
            waypoints       = wp_points,
            capture_offset  = self._cap_offset,
            move_timeout_ms = wp_timeout,
            loop            = True,
        )

        # ── 아데나 탐지기 ─────────────────────────────────────────────
        loot_cfg = config.get("loot", {})
        self.loot_detector = LootDetector(
            scan_region     = loot_cfg.get("scan_region",
                                           {"x":0,"y":0,"width":1440,"height":780}),
            loot_keywords   = loot_cfg.get("keywords", ["아데나","Adena"]),
            scan_interval_s = loot_cfg.get("scan_interval_s", 0.5),
            roi_offset      = self._roi_offset,
            capture_offset  = self._cap_offset,
        )

        # ── 루팅 상태 ─────────────────────────────────────────────────
        self._loot_targets: list  = []
        self._loot_idx            = 0
        self._loot_click_iv       = loot_cfg.get("click_interval_ms", 400) / 1000.0
        self._last_loot_click     = 0.0
        self._loot_timeout        = loot_cfg.get("timeout_ms", 3000) / 1000.0
        self._loot_start_t        = 0.0
        self._loot_return_state   = HuntingState.HUNTING_10  # 루팅 후 복귀 상태

        # ── 사냥 중 적 비활성 타임아웃 ────────────────────────────────
        self._hunt_idle_timeout   = 3.0   # 적 없을 때 이 시간 초과 → 웨이포인트 이동
        self._last_enemy_seen_t   = time.time()

        # ── 속도 물약 대기 ─────────────────────────────────────────────
        self._speed_potion_wait   = 1.0   # F9 후 대기 시간 (초)

        # ── 통계 ──────────────────────────────────────────────────────
        self.kills       = 0
        self.potions_used = 0
        self.start_time  = time.time()

    # ── 공개 API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """자동 레벨링 시작."""
        logger.info("=" * 60)
        logger.info("[HuntingSM] ▶ 요정 1단계 자동 레벨링 시작")
        logger.info(f"  허수아비 목표: Lv.{self.target_level_dummy}  "
                    f"사냥터 목표: Lv.{self.target_level_hunt}")
        logger.info(f"  물약 키: {self.key_potion}  "
                    f"두루마리 키: {self.key_scroll}  "
                    f"속도물약 키: {self.key_speed_potion}")
        logger.info("=" * 60)
        self._enter(HuntingState.USE_SCROLL_DUMMY)

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
        """매 detection tick마다 호출합니다.

        Args:
            frame:   현재 캡처 프레임 (BGR ndarray)
            enemies: NearestNeighborTracker.update() 반환 Enemy 리스트
        """
        if self.state == HuntingState.IDLE:
            return
        if self.state == HuntingState.DONE_PHASE1:
            return

        # ── HP 체크 (IDLE/DONE 제외 모든 상태) ────────────────────────
        self._check_hp_and_use_potion(frame)

        # ── 상태별 처리 ───────────────────────────────────────────────
        if self.state == HuntingState.USE_SCROLL_DUMMY:
            self._update_use_scroll_dummy()

        elif self.state == HuntingState.MOVE_TO_DUMMY:
            self._update_move_to_dummy()

        elif self.state == HuntingState.ATTACKING_DUMMY:
            self._update_attacking_dummy(frame)

        elif self.state == HuntingState.USE_SPEED_POTION:
            self._update_use_speed_potion()

        elif self.state == HuntingState.MOVE_TO_HUNT_ZONE:
            self._update_move_to_hunt_zone(frame, enemies)

        elif self.state == HuntingState.HUNTING_10:
            self._update_hunting_10(frame, enemies)

        elif self.state == HuntingState.LOOTING:
            self._update_looting(frame)

    # ── 상태별 처리 메서드 ────────────────────────────────────────────────

    def _update_use_scroll_dummy(self) -> None:
        """F6 말하는 두루마리 → 목적지 창 → '허수아비 수련장' 클릭."""
        logger.info(
            f"[HuntingSM] {self.key_scroll} 말하는 두루마리 사용 "
            f"→ 허수아비 수련장으로 텔레포트"
        )
        success = self.scroll_teleporter.execute(self.grab, self.pico)
        if success:
            logger.info("[HuntingSM] ✅ 텔레포트 성공 → 허수아비 이동 시작")
            self._dummy_move_done = False
            self._enter(HuntingState.MOVE_TO_DUMMY)
        else:
            logger.warning("[HuntingSM] ⚠ 텔레포트 실패 — 2초 후 재시도")
            time.sleep(2.0)
            # 상태 유지 → 다음 tick에 재시도

    def _update_move_to_dummy(self) -> None:
        """허수아비 방향으로 드래그 준비 (이동 없이 바로 공격 시작)."""
        now = time.time()

        if not self._dummy_move_done:
            fx, fy = self.dummy_drag_from
            tx, ty = self.dummy_drag_to
            logger.info(f"[HuntingSM] 허수아비 도착 → 드래그 공격 시작: ({fx},{fy})→({tx},{ty})")
            self._dummy_move_start = now
            self._dummy_move_done  = True

        # 짧은 대기 후 바로 공격 상태로 전환
        if now - self._dummy_move_start >= 0.5:
            self._dummy_move_done = False
            self._enter(HuntingState.ATTACKING_DUMMY)

    def _update_attacking_dummy(self, frame: np.ndarray) -> None:
        """허수아비 방향으로 드래그 반복 공격. 목표 레벨 달성 시 속도 물약 사용."""
        now = time.time()

        # 레벨 읽기
        level = self.level_reader.read(frame)

        # 목표 레벨 달성 → 속도 물약으로
        if level is not None and level >= self.target_level_dummy:
            logger.info(
                f"[HuntingSM] ⭐ Lv.{level} 달성! "
                f"(목표 Lv.{self.target_level_dummy}) "
                f"→ 속도향상물약 사용"
            )
            self._enter(HuntingState.USE_SPEED_POTION)
            return

        # 주기적 드래그 공격 (마우스 누른 채 드래그 → 놓기)
        if now - self._last_dummy_atk >= self.dummy_atk_interval:
            fx, fy = self.dummy_drag_from
            tx, ty = self.dummy_drag_to
            self.pico.drag(fx, fy, tx, ty, self.dummy_drag_steps)
            self._last_dummy_atk = now
            elapsed = int(now - self._entered_at)
            logger.debug(
                f"[HuntingSM] 허수아비 드래그 공격: ({fx},{fy})→({tx},{ty}) "
                f"레벨={level or '?'} 경과={elapsed}s"
            )

    def _update_use_speed_potion(self) -> None:
        """F9 속도향상물약 사용 후 대기."""
        logger.info(f"[HuntingSM] {self.key_speed_potion} 속도향상물약 사용")
        self.pico.key_tap_name(self.key_speed_potion, hold_ms=80)
        time.sleep(self._speed_potion_wait)

        logger.info("[HuntingSM] 속도향상물약 완료 → 사냥터로 이동 시작")
        self.hunt_mover.start()
        self._enter(HuntingState.MOVE_TO_HUNT_ZONE)

    def _update_move_to_hunt_zone(
        self, frame: np.ndarray, enemies: list
    ) -> None:
        """hunt_waypoints 순환 이동. 적 발견 시 즉시 HUNTING_10으로."""
        # 적 발견 시 즉시 사냥 전환
        if enemies:
            logger.info(f"[HuntingSM] 이동 중 적 {len(enemies)}명 발견 → HUNTING_10")
            self._last_enemy_seen_t = time.time()
            self._enter(HuntingState.HUNTING_10)
            return

        # 웨이포인트 이동 tick
        status = self.hunt_mover.tick(self.pico)
        if status == "ARRIVED":
            label = self.hunt_mover.current_label
            logger.info(f"[HuntingSM] 웨이포인트 '{label}' 도착 — 대기 중")

    def _update_hunting_10(
        self, frame: np.ndarray, enemies: list
    ) -> None:
        """사냥터 사냥. 아데나 탐지, Lv.target_level_hunt 달성 시 DONE_PHASE1."""
        now = time.time()

        # 레벨 체크 → 10레벨 달성 = 1단계 완료
        level = self.level_reader.read(frame)
        if level is not None and level >= self.target_level_hunt:
            logger.info(
                f"[HuntingSM] 🎉 Lv.{level} 달성! "
                f"(목표 Lv.{self.target_level_hunt}) "
                f"→ 1단계 완료"
            )
            self._enter(HuntingState.DONE_PHASE1)
            return

        # 적 있으면 타임스탬프 갱신
        if enemies:
            self._last_enemy_seen_t = now
        else:
            # 적 없이 idle_timeout 초과 → 웨이포인트 이동
            idle_s = now - self._last_enemy_seen_t
            if idle_s >= self._hunt_idle_timeout:
                logger.info(
                    f"[HuntingSM] 적 없음 {idle_s:.0f}s "
                    f"→ 웨이포인트 이동 재개"
                )
                self._enter(HuntingState.MOVE_TO_HUNT_ZONE)
                return

        # 아데나 탐지 → LOOTING
        loot = self.loot_detector.find(frame)
        if loot:
            logger.info(f"[HuntingSM] 아데나 {len(loot)}개 발견 → LOOTING")
            self._loot_targets       = list(loot)
            self._loot_idx           = 0
            self._loot_start_t       = now
            self._loot_return_state  = HuntingState.HUNTING_10
            self._enter(HuntingState.LOOTING)

    def _update_looting(self, frame: np.ndarray) -> None:
        """아데나를 하나씩 클릭."""
        now = time.time()

        # 타임아웃 체크
        if now - self._loot_start_t >= self._loot_timeout:
            logger.info("[HuntingSM] 루팅 타임아웃 → 사냥 복귀")
            self._enter(self._loot_return_state)
            return

        # 모두 클릭 완료
        if self._loot_idx >= len(self._loot_targets):
            logger.info("[HuntingSM] 루팅 완료 → 사냥 복귀")
            self.kills += 1
            self._enter(self._loot_return_state)
            return

        # 클릭 간격 체크
        if now - self._last_loot_click < self._loot_click_iv:
            return

        sx, sy, text, conf = self._loot_targets[self._loot_idx]
        logger.info(f"[HuntingSM] 아데나 클릭: ({sx},{sy}) '{text}'")
        self.pico.click(sx, sy)
        self._last_loot_click = now
        self._loot_idx += 1

    # ── 공통 HP 체크 ──────────────────────────────────────────────────────

    def _check_hp_and_use_potion(self, frame: np.ndarray) -> None:
        """HP가 기준 이하면 F5 물약 사용 (쿨타임 체크)."""
        hp_pct = self.hp_reader.read(frame)
        now    = time.time()

        if (hp_pct < self.hp_reader.threshold_pct
                and now - self._last_potion_t >= self._potion_cooldown):
            logger.info(
                f"[HuntingSM] 💊 HP {hp_pct:.0f}% < "
                f"{self.hp_reader.threshold_pct:.0f}% "
                f"→ {self.key_potion} 물약 사용"
            )
            self.pico.key_tap_name(self.key_potion, hold_ms=80)
            self._last_potion_t  = now
            self.potions_used   += 1

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _enter(self, new_state: HuntingState) -> None:
        """상태 전환."""
        prev = self.state
        self.state       = new_state
        self._entered_at = time.time()
        logger.info(f"[HuntingSM] {prev.name} → {new_state.name}")

    # ── 상태 조회 API ─────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """UI 표시용 현재 상태 딕셔너리."""
        level  = self.level_reader.get_cached()
        hp_pct = self.hp_reader.get_cached()

        waypoint = "-"
        if self.state in (HuntingState.MOVE_TO_HUNT_ZONE,
                          HuntingState.HUNTING_10):
            waypoint = self.hunt_mover.current_label

        return {
            "state":        self.state.name,
            "level":        level if level is not None else "?",
            "hp_pct":       round(hp_pct, 1),
            "kills":        self.kills,
            "potions":      self.potions_used,
            "elapsed_min":  round(self.elapsed_min, 1),
            "waypoint":     waypoint,
        }
