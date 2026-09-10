"""
modes/leveling.py — 요정 1단계 자동 레벨링 모드.

전체 흐름
─────────
    IDLE
      │ start() 호출
      ▼
    USE_SCROLL_DUMMY   ← F6 두루마리 → 허수아비 수련장 텔레포트
      ▼
    MOVE_TO_DUMMY      ← 허수아비 좌표로 이동 (타임아웃)
      ▼
    ATTACKING_DUMMY    ← 허수아비 반복 드래그 공격 → target_level_dummy 도달 시
      ▼
    USE_SPEED_POTION   ← F9 속도향상물약 사용
      ▼
    MOVE_TO_HUNT_ZONE  ← hunt_waypoints 순차 이동
      ▼
    HUNTING            ← 사냥터 순찰(patrol_waypoints) + 적 탐지/공격/루팅
      ▼
    LOOTING            ← 아데나 줍기 (재스캔 최대 3회, 8초 타임아웃)
      ▼                   (완료 → HUNTING 복귀)
    DONE               ← target_level_hunt 도달 → 종료

어느 상태에서든:
    stop() → IDLE
    HP < threshold → F5 물약 (쿨타임 체크)
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from core.state import BaseFSM

log = logging.getLogger(__name__)

# ─────────────────────────── 상태 열거 ──────────────────────────

class LevelingState(Enum):
    IDLE             = auto()
    USE_SCROLL_DUMMY = auto()
    MOVE_TO_DUMMY    = auto()
    ATTACKING_DUMMY  = auto()
    USE_SPEED_POTION = auto()
    MOVE_TO_HUNT_ZONE = auto()
    HUNTING          = auto()
    LOOTING          = auto()
    DONE             = auto()


# ─────────────────────────── 모드 클래스 ────────────────────────

class LevelingMode(BaseFSM):
    """
    요정 1단계 자동 레벨링 전체 흐름을 관리하는 FSM.

    Parameters
    ----------
    settings : Settings
        통합 설정 객체.
    pico : PicoWorker | NullPicoWorker
        키/클릭 입력 디바이스.
    frame_grabber : Callable[[], np.ndarray]
        BGR 프레임 반환 함수 (ScreenCapturer.grab 등).
    tracker : NearestNeighborTracker | None
        적 추적기. 주입 시 HUNTING 에서 SM 이 공격을 담당.
    hp_reader : HpReader | None
        HP 리더. None 이면 HP 체크 생략.
    level_reader : LevelReader | None
        레벨 리더. None 이면 레벨 체크 생략 (타임아웃 기반 전환).
    loot_detector : LootDetector | None
        루팅 탐지기. None 이면 루팅 단계 생략.
    """

    def __init__(
        self,
        settings,
        pico,
        frame_grabber: Callable[[], np.ndarray],
        tracker=None,
        hp_reader=None,
        level_reader=None,
        loot_detector=None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.pico = pico
        self.grab = frame_grabber
        self.tracker = tracker
        self.hp_reader = hp_reader
        self.level_reader = level_reader
        self.loot_detector = loot_detector

        # ── 키 설정 ───────────────────────────────────────────────
        keys = settings.keys
        self._key_potion       = getattr(keys, "potion",       "F5")
        self._key_scroll       = getattr(keys, "scroll",       "F6")
        self._key_speed_potion = getattr(keys, "speed_potion", "F9")
        self._potion_cd        = getattr(keys, "potion_cooldown_ms", 3000) / 1000.0
        self._last_potion_t    = 0.0

        # ── 허수아비 공격 설정 ────────────────────────────────────
        dummy = settings.dummy
        dfrom = getattr(dummy, "drag_from", {"x": 960, "y": 600})
        dto   = getattr(dummy, "drag_to",   {"x": 960, "y": 400})
        self._dummy_drag_from    = _pt(dfrom)
        self._dummy_drag_to      = _pt(dto)
        self._dummy_drag_steps   = getattr(dummy, "drag_steps", 8)
        self._dummy_atk_interval = getattr(dummy, "attack_interval_ms", 500) / 1000.0
        self._dummy_move_timeout = getattr(dummy, "move_timeout_ms", 3000) / 1000.0
        self._dummy_attack_dur   = getattr(dummy, "attack_duration_s", 0.0)
        self._last_dummy_atk     = 0.0

        # ── 레벨 목표 ─────────────────────────────────────────────
        lo = settings.level_ocr
        self._target_dummy = getattr(lo, "target_level_dummy", 5)
        self._target_hunt  = getattr(lo, "target_level_hunt",  10)

        # ── 텔레포트 핸들러 ───────────────────────────────────────
        self._teleport_handler = _build_teleport_handler(settings)
        self._teleport_retry   = 0

        # ── 웨이포인트 무버 ───────────────────────────────────────
        self._hunt_mover   = _build_waypoint_mover(settings, "hunt_waypoints",   loop=False)
        self._patrol_mover = _build_waypoint_mover(settings, "patrol_waypoints", loop=True)
        self._patrol_started = False

        # ── 루팅 상태 ─────────────────────────────────────────────
        loot_cfg = settings.loot
        self._loot_timeout   = getattr(loot_cfg, "timeout_ms",   8000) / 1000.0
        self._loot_rescan    = getattr(loot_cfg, "rescan_max",   3)
        self._loot_rescan_cnt = 0

        # ── 속도물약 대기 ─────────────────────────────────────────
        self._speed_potion_wait = getattr(settings.keys, "speed_potion_wait_ms", 500) / 1000.0

        self._running = False

    # ────────────────────────────── 공개 API ────────────────────────

    def start(self) -> None:
        """레벨링을 시작한다. IDLE → USE_SCROLL_DUMMY."""
        if self.is_state(LevelingState.IDLE):
            self._running = True
            self.transition(LevelingState.USE_SCROLL_DUMMY)

    def stop(self) -> None:
        """즉시 중단하고 IDLE 로 돌아간다."""
        self._running = False
        if self.tracker:
            self.tracker.set_active(False)
        self.transition(LevelingState.IDLE, force=True)

    @property
    def is_done(self) -> bool:
        return self.is_state(LevelingState.DONE)

    # ────────────────────────────── BaseFSM ──────────────────────────

    def _initial_state(self):
        return LevelingState.IDLE

    def tick(self, dt: float) -> None:
        if not self._running and not self.is_state(LevelingState.IDLE):
            return

        s = self.state
        if s == LevelingState.IDLE:
            return
        elif s == LevelingState.USE_SCROLL_DUMMY:
            self._tick_use_scroll()
        elif s == LevelingState.MOVE_TO_DUMMY:
            self._tick_move_to_dummy(dt)
        elif s == LevelingState.ATTACKING_DUMMY:
            self._tick_attacking_dummy()
        elif s == LevelingState.USE_SPEED_POTION:
            self._tick_speed_potion()
        elif s == LevelingState.MOVE_TO_HUNT_ZONE:
            self._tick_move_to_hunt()
        elif s == LevelingState.HUNTING:
            self._tick_hunting()
        elif s == LevelingState.LOOTING:
            self._tick_looting()
        elif s == LevelingState.DONE:
            pass

        # 공통: HP 체크
        self._check_hp()

    # ────────────────────────────── on_enter 훅 ─────────────────────

    def on_enter_USE_SCROLL_DUMMY(self):
        log.info("[Leveling] USE_SCROLL_DUMMY — 두루마리 텔레포트 시작")
        self._teleport_retry = 0
        if self._teleport_handler:
            self._teleport_handler.reset()

    def on_enter_MOVE_TO_DUMMY(self):
        log.info("[Leveling] MOVE_TO_DUMMY — 허수아비로 이동")
        self._dummy_move_start = time.monotonic()

    def on_enter_ATTACKING_DUMMY(self):
        log.info("[Leveling] ATTACKING_DUMMY — 허수아비 공격 시작")
        self._last_dummy_atk = 0.0

    def on_enter_USE_SPEED_POTION(self):
        log.info("[Leveling] USE_SPEED_POTION — 속도향상물약 사용")
        self.pico.key_tap_name(self._key_speed_potion, hold_ms=80)

    def on_enter_MOVE_TO_HUNT_ZONE(self):
        log.info("[Leveling] MOVE_TO_HUNT_ZONE — 사냥터 이동 시작")
        self._hunt_mover.start()

    def on_enter_HUNTING(self):
        log.info("[Leveling] HUNTING — 사냥 시작")
        if not self._patrol_started:
            self._patrol_mover.start()
            self._patrol_started = True
        if self.tracker:
            self.tracker.set_active(True)

    def on_enter_LOOTING(self):
        log.info("[Leveling] LOOTING — 루팅 시작")
        self._loot_rescan_cnt = 0
        if self.tracker:
            self.tracker.set_active(False)
        if self.loot_detector:
            self.loot_detector.invalidate()

    def on_exit_LOOTING(self):
        if self.loot_detector:
            self.loot_detector.invalidate()
        if self.tracker:
            self.tracker.set_active(True)

    def on_enter_DONE(self):
        log.info("[Leveling] DONE — 1단계 레벨링 완료!")
        self._running = False
        if self.tracker:
            self.tracker.set_active(False)

    # ────────────────────────────── 상태별 tick ──────────────────────

    def _tick_use_scroll(self):
        """텔레포트 핸들러 tick; 완료 또는 핸들러 없으면 MOVE_TO_DUMMY."""
        if self._teleport_handler is None:
            self.transition(LevelingState.MOVE_TO_DUMMY)
            return
        frame = self.grab()
        result = self._teleport_handler.tick(frame, self.pico)
        if result is True:
            log.info("[Leveling] 텔레포트 완료")
            self.transition(LevelingState.MOVE_TO_DUMMY)
        elif result is False:
            self._teleport_retry += 1
            log.warning("[Leveling] 텔레포트 실패 (%d)", self._teleport_retry)
            if self._teleport_retry >= 5:
                log.error("[Leveling] 텔레포트 최대 재시도 초과 → IDLE")
                self.stop()
            else:
                self._teleport_handler.reset()

    def _tick_move_to_dummy(self, dt: float):
        """허수아비 위치로 타임아웃 이동."""
        elapsed = time.monotonic() - self._dummy_move_start
        if elapsed >= self._dummy_move_timeout:
            log.info("[Leveling] 허수아비 이동 완료 (타임아웃)")
            self.transition(LevelingState.ATTACKING_DUMMY)

    def _tick_attacking_dummy(self):
        """허수아비 드래그 반복 공격."""
        now = time.monotonic()
        if now - self._last_dummy_atk >= self._dummy_atk_interval:
            fx, fy = self._dummy_drag_from
            tx, ty = self._dummy_drag_to
            self.pico.drag(fx, fy, tx, ty, steps=self._dummy_drag_steps)
            self._last_dummy_atk = now
            log.debug("[Leveling] 허수아비 드래그 공격")

        # 레벨 달성 체크
        if self.level_reader:
            frame = self.grab()
            lv = self.level_reader.read(frame)
            if lv is not None and lv >= self._target_dummy:
                log.info("[Leveling] 허수아비 목표 Lv.%d 달성", lv)
                self.transition(LevelingState.USE_SPEED_POTION)
                return

        # attack_duration_s 타임아웃 (레벨 리더 없을 때)
        if self._dummy_attack_dur > 0 and self.state_elapsed_s >= self._dummy_attack_dur:
            log.info("[Leveling] 허수아비 공격 타임아웃 — 다음 단계")
            self.transition(LevelingState.USE_SPEED_POTION)

    def _tick_speed_potion(self):
        """속도물약 대기 후 MOVE_TO_HUNT_ZONE."""
        if self.state_elapsed_s >= self._speed_potion_wait:
            self.transition(LevelingState.MOVE_TO_HUNT_ZONE)

    def _tick_move_to_hunt(self):
        """hunt_waypoints 순차 이동."""
        status = self._hunt_mover.tick(self.pico)
        if status in ("DONE", "ARRIVED"):
            log.info("[Leveling] 사냥터 도착")
            self.transition(LevelingState.HUNTING)

    def _tick_hunting(self):
        """사냥터 순찰 + 추적기 공격 + 루팅 전환."""
        frame = self.grab()

        # 레벨 완료 체크
        if self.level_reader:
            lv = self.level_reader.read(frame)
            if lv is not None and lv >= self._target_hunt:
                log.info("[Leveling] 목표 Lv.%d 달성 → DONE", lv)
                self.transition(LevelingState.DONE)
                return

        # 루팅 전환 체크
        if self.loot_detector:
            items = self.loot_detector.find(frame)
            if items:
                log.info("[Leveling] 아데나 발견 → LOOTING")
                self.transition(LevelingState.LOOTING)
                return

        # 순찰 이동
        self._patrol_mover.tick(self.pico)

    def _tick_looting(self):
        """아데나 루팅 — 재스캔 3회, 8초 타임아웃."""
        if self.state_elapsed_s >= self._loot_timeout:
            log.info("[Leveling] 루팅 타임아웃 → HUNTING 복귀")
            self.transition(LevelingState.HUNTING)
            return

        frame = self.grab()
        if not self.loot_detector:
            self.transition(LevelingState.HUNTING)
            return

        item = self.loot_detector.find_nearest(frame, ref_x=0, ref_y=0)
        if item:
            self.pico.click(item.cx, item.cy, pulse_ms=50)
            log.debug("[Leveling] 루팅 클릭 (%d,%d)", item.cx, item.cy)
            self.loot_detector.invalidate()
        else:
            self._loot_rescan_cnt += 1
            log.debug("[Leveling] 루팅 재스캔 %d/%d", self._loot_rescan_cnt, self._loot_rescan)
            if self._loot_rescan_cnt >= self._loot_rescan:
                log.info("[Leveling] 루팅 완료 → HUNTING 복귀")
                self.transition(LevelingState.HUNTING)

    # ────────────────────────────── 공통 헬퍼 ───────────────────────

    def _check_hp(self):
        """HP 낮으면 물약 사용 (쿨타임 체크)."""
        if self.hp_reader is None:
            return
        now = time.monotonic()
        if now - self._last_potion_t < self._potion_cd:
            return
        frame = self.grab()
        self.hp_reader.read(frame)
        if self.hp_reader.is_low():
            self.pico.key_tap_name(self._key_potion, hold_ms=50)
            self._last_potion_t = now
            log.info("[Leveling] HP 낮음 → %s 물약 사용", self._key_potion)


# ─────────────────────────── 팩토리 헬퍼 ────────────────────────

def _pt(d) -> tuple:
    """dict {'x':…,'y':…} 또는 (x,y) 를 (int,int) 로 변환."""
    if isinstance(d, dict):
        return int(d.get("x", 0)), int(d.get("y", 0))
    return int(d[0]), int(d[1])


def _build_teleport_handler(settings):
    """settings.teleport 로부터 TeleportHandler 를 생성. 설정 없으면 None."""
    try:
        from automation.teleport_handler import TeleportHandler  # 기존 flslwl 호환
    except ImportError:
        pass
    # bot2 내에는 TeleportHandler 가 없으므로 None 반환 (UI 에서 주입 가능)
    return None


def _build_waypoint_mover(settings, attr: str, loop: bool):
    """settings 에서 waypoint 목록을 읽어 _SimpleMover 를 생성."""
    wp_settings = getattr(settings, attr, None)
    if wp_settings is None:
        points = []
        timeout = 5000
    else:
        points  = getattr(wp_settings, "points",  [])
        timeout = getattr(wp_settings, "move_timeout_ms", 5000)
    return _SimpleMover(points=points, move_timeout_ms=timeout, loop=loop)


# ─────────────────────── 내장 간이 웨이포인트 무버 ──────────────

class _SimpleMover:
    """
    WaypointMover 의 경량 내장 구현.
    settings 에서 읽은 waypoints 리스트를 순차적으로 클릭 이동한다.
    """

    def __init__(self, points: list, move_timeout_ms: int, loop: bool) -> None:
        self.points = points
        self.timeout_s = move_timeout_ms / 1000.0
        self.loop = loop
        self._idx = 0
        self._start_t = 0.0
        self._active = False

    def start(self) -> None:
        self._idx = 0
        self._start_t = time.monotonic()
        self._active = True

    def reset(self) -> None:
        self._active = False
        self._idx = 0

    def tick(self, pico) -> str:
        """
        Returns
        -------
        str
            "IDLE" | "MOVING" | "ARRIVED" | "DONE"
        """
        if not self._active:
            return "IDLE"
        if not self.points:
            self._active = False
            return "DONE"

        if self._idx >= len(self.points):
            if self.loop:
                self._idx = 0
                self._start_t = time.monotonic()
            else:
                self._active = False
                return "DONE"

        wp = self.points[self._idx]
        x = int(wp.get("x", 0)) if isinstance(wp, dict) else int(wp[0])
        y = int(wp.get("y", 0)) if isinstance(wp, dict) else int(wp[1])

        now = time.monotonic()
        if now - self._start_t == 0.0:
            # 첫 틱: 클릭 전송
            pico.click(x, y, pulse_ms=50)
            self._start_t = now
            return "MOVING"

        elapsed = now - self._start_t
        if elapsed >= self.timeout_s:
            # 타임아웃 → 다음 웨이포인트
            wait_ms = int(wp.get("wait_ms", 0)) if isinstance(wp, dict) else 0
            if wait_ms > 0:
                time.sleep(wait_ms / 1000.0)
            self._idx += 1
            self._start_t = time.monotonic()
            if self._idx >= len(self.points) and not self.loop:
                self._active = False
                return "DONE"
            # 다음 WP 클릭
            if self._idx < len(self.points):
                nwp = self.points[self._idx]
                nx = int(nwp.get("x", 0)) if isinstance(nwp, dict) else int(nwp[0])
                ny = int(nwp.get("y", 0)) if isinstance(nwp, dict) else int(nwp[1])
                pico.click(nx, ny, pulse_ms=50)
            return "ARRIVED"

        return "MOVING"
