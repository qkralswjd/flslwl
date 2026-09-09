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
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from automation.hp_reader      import HpReader
from automation.level_reader   import LevelReader
from automation.loot_detector  import LootDetector
from automation.teleport_handler import TeleportHandler
from automation.waypoint_mover import WaypointMover

if TYPE_CHECKING:
    from tracking.tracker import NearestNeighborTracker

logger = logging.getLogger("hunting_sm")

# 텔레포트 최대 재시도 횟수: 이 횟수 초과 시 IDLE로 전환해 안전 정지
MAX_TELEPORT_RETRY = 5


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
        tracker: Optional["NearestNeighborTracker"] = None,
    ):
        """
        Args:
            config       : config_automation.json 내용
            pico_worker  : PicoSerialWorker 인스턴스
            frame_grabber: capturer.grab() 등 프레임 반환 함수
            tracker      : NearestNeighborTracker (field 모드 시 주입).
                           주입하면 HUNTING_10에서 SequentialTargetSM이 공격을 담당하고,
                           tracker._sm.state == IDLE일 때만 다음 WP로 이동합니다.
                           None이면 기존 직접 click+drag 방식 유지.
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
        # attack_duration_s: 0이면 레벨 달성까지 무한, 양수면 해당 초 후 다음 단계
        self.dummy_attack_duration = dummy_cfg.get("attack_duration_s", 0.0)
        self._dummy_move_done    = False
        self._dummy_move_start   = 0.0
        self._last_dummy_atk     = 0.0
        self._dummy_drag_done    = False  # 드래그 1회만 실행

        # ── 사냥터 웨이포인트 무버 (hunt_waypoints → HUNTING_10 진입용, loop=False) ──
        hwp_cfg   = config.get("hunt_waypoints", {})
        wp_points = hwp_cfg.get("points", [{"x":960,"y":400,"label":"사냥터-A","wait_ms":2000}])
        wp_timeout = hwp_cfg.get("move_timeout_ms", 5000)
        self.hunt_mover = WaypointMover(
            waypoints       = wp_points,
            capture_offset  = self._cap_offset,
            move_timeout_ms = wp_timeout,
            loop            = False,
        )

        # ── 사냥터 순찰 무버 (patrol_waypoints → HUNTING_10 내부 루프, loop=True) ──
        pwp_cfg     = config.get("patrol_waypoints", {})
        pwp_points  = pwp_cfg.get("points", wp_points)   # 없으면 hunt_waypoints 재사용
        pwp_timeout = pwp_cfg.get("move_timeout_ms", 8000)
        self.patrol_mover = WaypointMover(
            waypoints       = pwp_points,
            capture_offset  = self._cap_offset,
            move_timeout_ms = pwp_timeout,
            loop            = True,   # 사냥터 내 무한 순찰
        )
        self._patrol_started = False  # HUNTING_10 진입 시 최초 1회 start()

        # ── 순찰 전투 서브플로우 (patrol_combat) ──────────────────────
        #   _pc_state: "PATROL" | "SCAN" | "KILL_WAIT" | "LOOT_SCAN"
        #     PATROL    : 순찰 이동 중 (patrol_mover.tick 호출)
        #     SCAN      : 웨이포인트 도착 — 적/아데나 탐지
        #     KILL_WAIT : 공격 후 몬스터 사망 대기
        #     LOOT_SCAN : 사망 후 아데나 OCR 스캔
        pc_cfg = config.get("patrol_combat", {})
        self._pc_kill_wait      = pc_cfg.get("kill_wait_ms",        2500) / 1000.0
        self._pc_atk_interval   = pc_cfg.get("attack_interval_ms",   800) / 1000.0
        self._pc_max_attacks    = pc_cfg.get("max_attacks_per_wp",      3)

        self._pc_state          = "PATROL"   # 현재 서브상태
        self._pc_kill_start_t   = 0.0        # 공격 완료 시각
        self._pc_last_atk_t     = 0.0        # 마지막 공격 시각
        self._pc_attack_count   = 0          # 현재 WP에서 공격 횟수
        self._pc_scan_start_t   = 0.0        # SCAN 진입 시각 (3초 idle 타임아웃용)

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
        # config_automation.json patrol_waypoints.idle_timeout_s 로 설정 가능
        self._hunt_idle_timeout   = pwp_cfg.get("idle_timeout_s", 3.0)
        self._last_enemy_seen_t   = time.time()

        # ── 속도 물약 대기 ─────────────────────────────────────────────
        self._speed_potion_wait   = 1.0   # F9 후 대기 시간 (초)
        self._speed_potion_sent_at = 0.0  # F9 키 입력 시각 (비블로킹 대기용)

        # ── 텔레포트 재시도 제어 (비블로킹) ──────────────────────────────
        self._teleport_retry_at   = 0.0   # 이 시각 전에는 재시도 안 함
        self._teleport_fail_count = 0     # 연속 실패 횟수 (MAX_TELEPORT_RETRY 도달 시 IDLE)

        # ── 통계 ──────────────────────────────────────────────────────
        self.kills       = 0
        self.potions_used = 0
        self.start_time  = time.time()

        # ── field 모드 Tracker 연동 ────────────────────────────────────
        # tracker 주입 시: HUNTING_10에서 SequentialTargetSM이 공격 담당
        #   - PATROL: tracker._sm.state == IDLE일 때만 patrol_mover.tick()
        #   - SCAN  : tracker._sm.set_active(True) → SM이 공격 처리
        #   - KILL_WAIT 대신 tracker._sm.state == IDLE 복귀 감지
        # tracker=None 이면 기존 직접 click+drag 방식 유지
        self._tracker = tracker
        if tracker is not None:
            logger.info(f"[HuntingSM] field 모드 — tracker 주입 완료 ({type(tracker).__name__})")
        else:
            logger.info("[HuntingSM] tracker=None — 기존 직접 click+drag 모드")

    # ── 공개 API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """자동 레벨링 시작 (IDLE → USE_SCROLL_DUMMY → ... → DONE_PHASE1).

        텔레포트(F6)부터 시작해 전체 흐름을 실행합니다.
        허수아비 좌표만 공격하려면 start_at_dummy()를 사용하세요.
        """
        logger.info("=" * 60)
        logger.info("[HuntingSM] ▶ 요정 1단계 자동 레벨링 시작 (전체 흐름)")
        logger.info(f"  허수아비 목표: Lv.{self.target_level_dummy}  "
                    f"사냥터 목표: Lv.{self.target_level_hunt}")
        logger.info(f"  물약 키: {self.key_potion}  "
                    f"두루마리 키: {self.key_scroll}  "
                    f"속도물약 키: {self.key_speed_potion}")
        logger.info("=" * 60)
        self._enter(HuntingState.USE_SCROLL_DUMMY)

    def start_at_dummy(self) -> None:
        """허수아비 공격부터 시작 (텔레포트 생략).

        이미 허수아비 수련장에 있을 때 사용합니다 (1~5레벨 단계).
        drag_from/drag_to 좌표가 설정돼 있어야 합니다.
        Lv.target_level_dummy(기본 5) 달성 시 USE_SPEED_POTION → 사냥터로 전환합니다.
        """
        logger.info("=" * 60)
        logger.info("[HuntingSM] ▶ 허수아비 공격 시작 (1~5레벨 단계)")
        logger.info(f"  드래그: ({self.dummy_drag_from[0]},{self.dummy_drag_from[1]}) → "
                    f"({self.dummy_drag_to[0]},{self.dummy_drag_to[1]})")
        logger.info(f"  목표 레벨: Lv.{self.target_level_dummy}  "
                    f"공격 간격: {self.dummy_atk_interval*1000:.0f}ms")
        logger.info(f"  물약 키: {self.key_potion}  HP 임계: {self.hp_reader.threshold_pct:.0f}%")
        logger.info("=" * 60)
        self._dummy_move_done = False
        self._enter(HuntingState.ATTACKING_DUMMY)

    def start_at_hunt_zone(self) -> None:
        """사냥터 사냥부터 시작 (5~10레벨 단계).

        이미 사냥터에 있을 때 사용합니다.
        """
        logger.info("=" * 60)
        logger.info("[HuntingSM] ▶ 사냥터 사냥 시작 (5~10레벨 단계)")
        logger.info(f"  목표 레벨: Lv.{self.target_level_hunt}")
        logger.info("=" * 60)
        self.hunt_mover.start()
        self._enter(HuntingState.MOVE_TO_HUNT_ZONE)

    def start_at_hunting_10(self) -> None:
        """hunt_waypoints 이동 없이 바로 HUNTING_10(순찰+전투) 시작.

        이미 사냥터에 캐릭터가 있을 때 사용합니다.
        """
        logger.info("=" * 60)
        logger.info("[HuntingSM] ▶ 순찰 사냥 즉시 시작 (HUNTING_10 직진입)")
        logger.info(f"  목표 레벨: Lv.{self.target_level_hunt}")
        logger.info("=" * 60)
        self._enter(HuntingState.HUNTING_10)

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
        """F6 말하는 두루마리 → 목적지 창 → '허수아비 수련장' 클릭.

        TeleportHandler.tick()을 매 update() 호출마다 한 번씩 실행.
        tick()이 None을 반환하면 진행 중 -> 즉시 return (비블로킹).
        tick()이 False를 반환하면 2초 후 재시도 (retry_at 설정 후 reset).
        tick()이 True를 반환하면 MOVE_TO_DUMMY로 전환.
        """
        now = time.monotonic()

        # 실패 후 재시도 대기 중이면 즉시 return (tick 차단 없음)
        if now < self._teleport_retry_at:
            return

        result = self.scroll_teleporter.tick(self.grab, self.pico)

        if result is True:
            logger.info("[HuntingSM] 텔레포트 성공 -> 허수아비 이동 시작")
            self._teleport_fail_count = 0   # 성공 시 카운터 초기화
            self._dummy_move_done = False
            self._enter(HuntingState.MOVE_TO_DUMMY)   # _enter에서 retry_at 리셋

        elif result is False:
            self._teleport_fail_count += 1
            self.scroll_teleporter.reset()   # tick 단계 초기화

            if self._teleport_fail_count >= MAX_TELEPORT_RETRY:
                # 최대 재시도 횟수 초과 -> 안전 상태(IDLE)로 전환
                logger.error(
                    f"[RECOVERY] TELEPORT_FAIL | "
                    f"attempt={self._teleport_fail_count}/{MAX_TELEPORT_RETRY} | "
                    f"next=IDLE -- 텔레포트 포기, 자동사냥 정지"
                )
                self._enter(HuntingState.IDLE)   # _enter에서 fail_count/retry_at 리셋
            else:
                # 아직 재시도 가능 -> 2초 후 재시도 (비블로킹)
                self._teleport_retry_at = now + 2.0
                logger.warning(
                    f"[HuntingSM] 텔레포트 실패 "
                    f"({self._teleport_fail_count}/{MAX_TELEPORT_RETRY}) "
                    f"-- 2초 후 재시도 (비블로킹)"
                )
        # result is None: 진행 중, 다음 tick까지 대기

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
        """허수아비 방향으로 드래그 반복 공격. 목표 레벨 달성 시 속도 물약 사용.

        1~5레벨 구간 핵심 루프:
          - drag_from → drag_to 방향으로 공격 드래그
          - attack_interval_ms마다 반복
          - HP < threshold_pct → F5 물약 (공통 HP 체크에서 처리)
          - OCR Lv >= target_level_dummy(5) → USE_SPEED_POTION 전환
        """
        now = time.time()

        # 레벨 읽기 (2초 캐시 적용됨 — OCR 부하 최소화)
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

        # 시간 기반 종료: attack_duration_s > 0 이면 해당 시간 후 다음 단계
        if self.dummy_attack_duration > 0:
            elapsed = now - self._entered_at
            if elapsed >= self.dummy_attack_duration:
                logger.info(
                    f"[HuntingSM] ⏱ 허수아비 {self.dummy_attack_duration:.0f}초 공격 완료 "
                    f"→ 속도향상물약 사용"
                )
                self._enter(HuntingState.USE_SPEED_POTION)
                return

        # 드래그 공격 1회만 실행 (게임이 자동 반복하므로 1번이면 충분)
        if not self._dummy_drag_done:
            fx, fy = self.dummy_drag_from
            tx, ty = self.dummy_drag_to
            self.pico.drag(fx, fy, tx, ty, self.dummy_drag_steps)
            self._dummy_drag_done = True
            elapsed = int(now - self._entered_at)
            lv_str = str(level) if level is not None else "?"
            logger.info(
                f"[HuntingSM][DUMMY] 드래그 공격 1회 실행 ({fx},{fy})→({tx},{ty}) "
                f"Lv={lv_str}/{self.target_level_dummy} 경과={elapsed}s"
            )

    def _update_use_speed_potion(self) -> None:
        """F9 속도향상물약 사용 후 비블로킹 대기.

        최초 진입 시 키 입력 + _speed_potion_sent_at 기록.
        대기 완료 전까지는 즉시 return (tick 차단 없음).
        """
        now = time.monotonic()

        # 최초 진입: 키 입력 + 타임스탬프 기록
        if self._speed_potion_sent_at == 0.0:
            logger.info(f"[HuntingSM] {self.key_speed_potion} 속도향상물약 사용")
            self.pico.key_tap_name(self.key_speed_potion, hold_ms=80)
            self._speed_potion_sent_at = now
            return  # 이번 tick은 여기서 종료

        # 대기 중
        if now - self._speed_potion_sent_at < self._speed_potion_wait:
            return  # 아직 대기 중, tick 차단 없음

        # 대기 완료 -> 사냥터 이동
        logger.info("[HuntingSM] 속도향상물약 완료 -> 사냥터로 이동 시작")
        self._speed_potion_sent_at = 0.0   # 리셋 (재진입 대비)
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
        elif status == "DONE":
            logger.info("[HuntingSM] 사냥터 도착 완료 → HUNTING_10 전환")
            self._enter(HuntingState.HUNTING_10)

    def _update_hunting_10(
        self, frame: np.ndarray, enemies: list
    ) -> None:
        """사냥터 사냥 — PATROL_COMBAT 서브플로우.

        각 웨이포인트 도착 시 그 자리에서 전투를 완료한 뒤 다음으로 이동합니다.

        서브상태 전이:
            PATROL    → patrol_mover.tick() → ARRIVED 시 SCAN 전환
            SCAN      → enemies 있으면 공격 후 KILL_WAIT
                        enemies 없고 아데나 있으면 LOOTING (복귀→SCAN)
                        enemies 없고 아데나 없으면 PATROL (다음 WP)
            KILL_WAIT → kill_wait_ms 경과 후 LOOT_SCAN
            LOOT_SCAN → 아데나 있으면 LOOTING (복귀→SCAN)
                        아데나 없으면 PATROL (다음 WP)
        """
        now = time.time()

        # ── 최초 진입 시 순찰 시작 ─────────────────────────────────────
        if not self._patrol_started:
            self.patrol_mover.start()
            self._patrol_started = True
            self._pc_state       = "PATROL"
            logger.info("[HuntingSM] 순찰 시작 (patrol_waypoints loop)")

        # ── 레벨 체크 → 목표 레벨 달성 = 1단계 완료 ──────────────────
        level = self.level_reader.read(frame)
        if level is not None and level >= self.target_level_hunt:
            logger.info(
                f"[HuntingSM] Lv.{level} 달성! "
                f"(목표 Lv.{self.target_level_hunt}) → 1단계 완료"
            )
            self._enter(HuntingState.DONE_PHASE1)
            return

        # ═══════════════════════════════════════════════════════════════
        # PATROL_COMBAT 서브플로우
        # ═══════════════════════════════════════════════════════════════

        # ── [PATROL] 이동 중 ─────────────────────────────────────────
        if self._pc_state == "PATROL":
            # field 모드(tracker 있음): SequentialTargetSM이 IDLE일 때만 이동
            # → 전투 중(LOCKING~COOLDOWN)에는 patrol_mover를 전진시키지 않음
            if self._tracker is not None:
                from tracking.tracker import TargetState
                if self._tracker._sm.state != TargetState.IDLE:
                    # 전투 중 — patrol_mover 이동 클릭 차단, 적 처리 대기
                    return

                # ── 이동 중에도 적 발견 시 즉시 SCAN 전환 ─────────────
                # main.py가 SceneMotionFilter를 바이패스하므로
                # 이동 중 탐지된 enemies도 유효한 실제 적임
                if enemies and self.patrol_mover._state == "MOVING":
                    label = self.patrol_mover.current_label
                    logger.info(
                        f"[HuntingSM][field] 이동 중 적 {len(enemies)}명 발견 "
                        f"('{label}') → 즉시 SCAN 전환"
                    )
                    # patrol_mover는 MOVING 상태 유지 (다음 PATROL 진입 시 계속)
                    self._pc_state        = "SCAN"
                    self._pc_attack_count = 0
                    return

            status = self.patrol_mover.tick(self.pico)
            if status == "ARRIVED":
                label = self.patrol_mover.current_label
                logger.info(f"[HuntingSM] 순찰 '{label}' 도착 → 전투 스캔 (3초 대기)")
                self._pc_state        = "SCAN"
                self._pc_attack_count = 0
                self._pc_scan_start_t = now   # SCAN 진입 시각 기록
                # SCAN 진입 시 loot 캐시 무효화 → 이전 캐시 오탐 방지
                self.loot_detector.invalidate()
            return

        # ── [SCAN] 도착 지점 탐지 ────────────────────────────────────
        if self._pc_state == "SCAN":
            # ── field 모드: tracker._sm에 공격 위임 ──────────────────
            if self._tracker is None:
                logger.warning(
                    "[HuntingSM][SCAN] ⚠ tracker=None → 기존모드(max_attacks) 경로 탑승. "
                    "field 모드로 실행했다면 'python main.py field' 로 재실행하세요."
                )
            if self._tracker is not None:
                from tracking.tracker import TargetState
                sm = self._tracker._sm

                # SM이 이미 전투 중(비IDLE) → 완료 대기
                if sm.state != TargetState.IDLE:
                    return

                # SM이 IDLE = 전투 완료 또는 적 없음
                if enemies:
                    # 적 있음 → SM 활성화해서 자동 공격 시작
                    # (이동 중 발견한 적도 포함 — MOG2 리셋 후 탐지된 실제 적)
                    sm.set_active(True)
                    self._pc_attack_count += 1
                    self._pc_scan_start_t = now  # 적 발견 시 타임아웃 리셋
                    logger.info(
                        f"[HuntingSM][field] 적 {len(enemies)}명 감지 "
                        f"→ SequentialTargetSM 공격 위임 "
                        f"[{self._pc_attack_count}/{self._pc_max_attacks}]"
                    )
                    self._pc_kill_start_t = now
                    self._pc_state = "KILL_WAIT"
                    return

                # ── 3초 idle 타임아웃: 적 없으면 다음 WP로 이동 ──────
                # 이게 핵심: 적이 없을 때 설정해둔 동선대로 계속 이동
                idle_elapsed = now - self._pc_scan_start_t
                if idle_elapsed >= self._hunt_idle_timeout:
                    # 아데나 마지막 체크 후 다음 WP
                    loot = self.loot_detector.find(frame)
                    if loot:
                        logger.info(f"[HuntingSM][field] 아데나 {len(loot)}개 발견 → LOOTING")
                        sm.set_active(False)
                        self._loot_targets      = list(loot)
                        self._loot_idx          = 0
                        self._loot_start_t      = now
                        self._loot_return_state = HuntingState.HUNTING_10
                        self._enter(HuntingState.LOOTING)
                        return
                    logger.info(
                        f"[HuntingSM][field] {idle_elapsed:.1f}초 동안 적 없음 "
                        f"→ 다음 WP 이동"
                    )
                    sm.set_active(False)
                    self._pc_state        = "PATROL"
                    self._pc_attack_count = 0
                    return

                # 타임아웃 전: 계속 탐지 대기 (적 올 때까지)
                return

            # ── 기존 모드(tracker=None): 직접 click+drag ─────────────
            # 최대 공격 횟수 초과 → 다음 WP로 진행
            if self._pc_attack_count >= self._pc_max_attacks:
                logger.info(
                    f"[HuntingSM] 공격 {self._pc_attack_count}회 완료 "
                    f"(최대 {self._pc_max_attacks}) → 다음 WP"
                )
                self._pc_state        = "PATROL"
                self._pc_attack_count = 0
                return

            # 공격 쿨타임 체크
            if now - self._pc_last_atk_t < self._pc_atk_interval:
                return

            if enemies:
                # 가장 가까운 적 선택
                target = min(enemies, key=lambda e: (
                    (e.center_x - 960) ** 2 + (e.center_y - 540) ** 2
                ))
                tx = target.center_x + self._roi_offset[0]
                ty = target.center_y + self._roi_offset[1]

                self._pc_attack_count += 1
                logger.info(
                    f"[HuntingSM] 몬스터 ({tx},{ty}) 공격 "
                    f"[{self._pc_attack_count}/{self._pc_max_attacks}]"
                )

                # 1. 몬스터 위치 클릭 (이동+선택)
                self.pico.click(tx, ty)

                # 2. 드래그 공격
                fx, fy   = self.dummy_drag_from
                ttx, tty = self.dummy_drag_to
                self.pico.drag(fx, fy, ttx, tty, self.dummy_drag_steps)

                self._pc_last_atk_t   = now
                self._pc_kill_start_t = now
                self._pc_state        = "KILL_WAIT"
                return

            # 적 없음 → 아데나 즉시 체크
            loot = self.loot_detector.find(frame)
            if loot:
                logger.info(f"[HuntingSM] 아데나 {len(loot)}개 발견 → LOOTING")
                self._loot_targets      = list(loot)
                self._loot_idx          = 0
                self._loot_start_t      = now
                self._loot_return_state = HuntingState.HUNTING_10
                self._enter(HuntingState.LOOTING)
                return

            # 적도 아데나도 없음 → 다음 WP
            logger.info("[HuntingSM] 탐지 없음 → 다음 WP")
            self._pc_state = "PATROL"
            return

        # ── [KILL_WAIT] 몬스터 사망 대기 ─────────────────────────────
        if self._pc_state == "KILL_WAIT":
            # ── field 모드: tracker._sm이 IDLE로 돌아오면 전투 완료 ──
            if self._tracker is not None:
                from tracking.tracker import TargetState
                sm = self._tracker._sm

                # SM이 IDLE = 적 처리 완료 (또는 타임아웃)
                if sm.state == TargetState.IDLE:
                    logger.info("[HuntingSM][field] SequentialTargetSM IDLE → 아데나 스캔")
                    self._pc_state = "LOOT_SCAN"
                # 아직 전투 중이면 대기 (time-based 백업 타임아웃도 유지)
                elapsed = now - self._pc_kill_start_t
                if elapsed >= self._pc_kill_wait * 3:   # 기존 대기의 3배를 최대 한도
                    logger.warning(
                        f"[HuntingSM][field] KILL_WAIT 최대 대기 초과 "
                        f"({self._pc_kill_wait * 3:.1f}s) → 강제 LOOT_SCAN"
                    )
                    sm.set_active(False)
                    self._pc_state = "LOOT_SCAN"
                return

            # 기존 모드: 시간 기반 대기
            elapsed = now - self._pc_kill_start_t
            if elapsed >= self._pc_kill_wait:
                logger.info(
                    f"[HuntingSM] 사망 대기 완료 ({self._pc_kill_wait:.1f}s) "
                    f"→ 아데나 스캔"
                )
                self._pc_state = "LOOT_SCAN"
            return

        # ── [LOOT_SCAN] 사망 후 아데나 OCR ──────────────────────────
        if self._pc_state == "LOOT_SCAN":
            # field 모드: 아데나 스캔 전 SM 비활성화 (루팅 중 공격 차단)
            if self._tracker is not None:
                self._tracker._sm.set_active(False)

            # LOOT_SCAN 진입 직후 캐시 무효화 → 이전 캐시 오탐 방지
            # (최초 1회만 무효화: _last_scan_time이 0이면 이미 무효화된 상태)
            if self.loot_detector._last_scan_time != 0.0:
                self.loot_detector.invalidate()
                return  # 다음 tick에서 즉시 재스캔

            loot = self.loot_detector.find(frame)
            if loot:
                logger.info(f"[HuntingSM] 아데나 {len(loot)}개 발견 → LOOTING")
                self._loot_targets      = list(loot)
                self._loot_idx          = 0
                self._loot_start_t      = now
                self._loot_return_state = HuntingState.HUNTING_10
                self._enter(HuntingState.LOOTING)
            else:
                logger.info("[HuntingSM] 아데나 없음 → SCAN 재시도 (추가 적 탐지)")
                self._pc_state = "SCAN"
            return

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
        logger.info(f"[HuntingSM] {prev.name} -> {new_state.name}")

        # 상태 전환 시 비블로킹 타이머 리셋 (재진입 오작동 방지)
        if new_state != HuntingState.USE_SPEED_POTION:
            self._speed_potion_sent_at = 0.0
        # USE_SCROLL_DUMMY 진입: TeleportHandler tick 단계 초기화
        if new_state == HuntingState.USE_SCROLL_DUMMY:
            self.scroll_teleporter.reset()
        self._teleport_retry_at = 0.0
        # IDLE 전환(stop/복구): fail_count 리셋 (다음 start() 때 깨끗하게 시작)
        if new_state == HuntingState.IDLE:
            self._teleport_fail_count = 0
        # HUNTING_10 재진입 시 순찰 재시작
        # (서브상태 _pc_state는 처음에만 PATROL로 리셋 — LOOTING 복귀 시는 SCAN 유지)
        if new_state == HuntingState.HUNTING_10:
            self._patrol_started = False
            # LOOTING에서 복귀하는 경우 SCAN 유지 (추가 아데나 확인)
            if prev == HuntingState.LOOTING:
                self._pc_state = "SCAN"
            else:
                self._pc_state        = "PATROL"
                self._pc_attack_count = 0
        # HUNTING_10 이탈 시 field 모드 tracker SM 비활성화
        if prev == HuntingState.HUNTING_10 and getattr(self, "_tracker", None) is not None:
            self._tracker._sm.set_active(False)

    # ── 상태 조회 API ─────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """UI 표시용 현재 상태 딕셔너리."""
        level  = self.level_reader.get_cached()
        hp_pct = self.hp_reader.get_cached()

        waypoint = "-"
        if self.state == HuntingState.MOVE_TO_HUNT_ZONE:
            waypoint = self.hunt_mover.current_label
        elif self.state == HuntingState.HUNTING_10:
            waypoint = self.patrol_mover.current_label

        # SCAN 상태의 idle 경과시간 (필드모드 HUD용)
        _scan_idle = 0.0
        if self._pc_state == "SCAN" and self._pc_scan_start_t > 0:
            _scan_idle = time.time() - self._pc_scan_start_t

        return {
            "state":             self.state.name,
            "level":             level if level is not None else "?",
            "hp_pct":            round(hp_pct, 1),
            "kills":             self.kills,
            "potions":           self.potions_used,
            "elapsed_min":       round(self.elapsed_min, 1),
            "waypoint":          waypoint,
            "scan_idle_elapsed": round(_scan_idle, 1),
        }
