"""
modes/field.py — 필드 자유 사냥 모드 (NearestNeighborTracker 주입).

전체 흐름
─────────
    IDLE
      │ start() 호출
      ▼
    PATROLLING   ← patrol_waypoints 루프 이동 + 적 탐지 대기
      │ 적 감지 → ENGAGING
      ▼
    ENGAGING     ← tracker.SequentialTargetSM 이 공격 담당
      │ 공격 SM 이 IDLE 복귀 (또는 타임아웃) → 루팅 체크 → LOOTING or PATROLLING
      ▼
    LOOTING      ← 아데나 줍기 (재스캔 최대 N회, timeout_ms)
      │ 완료 → PATROLLING
      ▼
    DONE         ← max_kills 또는 stop() 호출

어느 상태에서든:
    stop() → IDLE
    HP < threshold → F5 물약
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from core.state import BaseFSM

log = logging.getLogger(__name__)


class FieldState(Enum):
    IDLE       = auto()
    PATROLLING = auto()
    ENGAGING   = auto()
    LOOTING    = auto()
    DONE       = auto()


class FieldMode(BaseFSM):
    """
    필드 자유 사냥 FSM.

    Parameters
    ----------
    settings : Settings
    pico : PicoWorker | NullPicoWorker
    frame_grabber : Callable[[], np.ndarray]
    tracker : NearestNeighborTracker
        적 추적기. ENGAGING 에서 tracker SM 이 공격 담당.
        None 이면 PATROLLING 에서 적 발견 시 직접 클릭.
    hp_reader : HpReader | None
    loot_detector : LootDetector | None
    max_kills : int
        목표 킬 수. 0 = 무한.
    """

    def __init__(
        self,
        settings,
        pico,
        frame_grabber: Callable[[], np.ndarray],
        tracker=None,
        hp_reader=None,
        loot_detector=None,
        max_kills: int = 0,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.pico = pico
        self.grab = frame_grabber
        self.tracker = tracker
        self.hp_reader = hp_reader
        self.loot_detector = loot_detector
        self.max_kills = max_kills

        # ── 키 설정 ───────────────────────────────────────────────
        keys = settings.keys
        self._key_potion = getattr(keys, "potion", "F5")
        self._potion_cd  = getattr(keys, "potion_cooldown_ms", 3000) / 1000.0
        self._last_potion = 0.0

        # ── 순찰 무버 ─────────────────────────────────────────────
        from modes.leveling import _build_waypoint_mover, _SimpleMover
        self._patrol = _build_waypoint_mover(settings, "patrol_waypoints", loop=True)

        # ── 전투 타임아웃 ─────────────────────────────────────────
        pc_cfg = getattr(settings, "patrol_combat", None)
        self._engage_timeout = (
            getattr(pc_cfg, "engage_timeout_ms", 15000) if pc_cfg else 15000
        ) / 1000.0

        # ── 루팅 설정 ─────────────────────────────────────────────
        loot_cfg = settings.loot
        self._loot_timeout = getattr(loot_cfg, "timeout_ms",  8000) / 1000.0
        self._loot_rescan  = getattr(loot_cfg, "rescan_max",  3)
        self._loot_rescan_cnt = 0

        # ── 카운터 ────────────────────────────────────────────────
        self._kill_count = 0
        self._running    = False

    # ────────────────────────────── 공개 API ────────────────────────

    def start(self) -> None:
        if self.is_state(FieldState.IDLE):
            self._running = True
            self._kill_count = 0
            self._patrol.start()
            self.transition(FieldState.PATROLLING)

    def stop(self) -> None:
        self._running = False
        if self.tracker:
            self.tracker.set_active(False)
        self.transition(FieldState.IDLE, force=True)

    @property
    def kill_count(self) -> int:
        return self._kill_count

    @property
    def is_done(self) -> bool:
        return self.is_state(FieldState.DONE)

    # ────────────────────────────── BaseFSM ──────────────────────────

    def _initial_state(self):
        return FieldState.IDLE

    def tick(self, dt: float) -> None:
        if not self._running and not self.is_state(FieldState.IDLE):
            return

        s = self.state
        if s == FieldState.IDLE:
            pass
        elif s == FieldState.PATROLLING:
            self._tick_patrolling()
        elif s == FieldState.ENGAGING:
            self._tick_engaging()
        elif s == FieldState.LOOTING:
            self._tick_looting()
        elif s == FieldState.DONE:
            pass

        self._check_hp()

    # ────────────────────────────── on_enter 훅 ─────────────────────

    def on_enter_PATROLLING(self):
        log.info("[Field] PATROLLING")
        if self.tracker:
            self.tracker.set_active(True)

    def on_enter_ENGAGING(self):
        log.info("[Field] ENGAGING")
        if self.tracker:
            self.tracker.set_active(True)

    def on_enter_LOOTING(self):
        log.info("[Field] LOOTING")
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
        log.info("[Field] DONE — 총 %d킬", self._kill_count)
        self._running = False
        if self.tracker:
            self.tracker.set_active(False)

    # ────────────────────────────── 상태별 tick ──────────────────────

    def _tick_patrolling(self):
        frame = self.grab()

        # 루팅 우선 체크
        if self.loot_detector:
            items = self.loot_detector.find(frame)
            if items:
                self.transition(FieldState.LOOTING)
                return

        # tracker 가 적을 발견하면 ENGAGING
        if self.tracker and self.tracker.attack_active:
            self.transition(FieldState.ENGAGING)
            return

        # 순찰 이동
        self._patrol.tick(self.pico)

    def _tick_engaging(self):
        frame = self.grab()

        # 루팅 체크
        if self.loot_detector:
            items = self.loot_detector.find(frame)
            if items:
                self._kill_count += 1
                self._check_done()
                self.transition(FieldState.LOOTING)
                return

        # tracker SM 이 IDLE 로 복귀 → 순찰 재개
        if self.tracker and not self.tracker.attack_active:
            self._kill_count += 1
            log.debug("[Field] 적 처치 — 누적 %d킬", self._kill_count)
            self._check_done()
            if not self.is_state(FieldState.DONE):
                self.transition(FieldState.PATROLLING)
            return

        # 교전 타임아웃
        if self.state_elapsed_s >= self._engage_timeout:
            log.warning("[Field] 교전 타임아웃 → PATROLLING")
            self.transition(FieldState.PATROLLING)

    def _tick_looting(self):
        if self.state_elapsed_s >= self._loot_timeout:
            self.transition(FieldState.PATROLLING)
            return

        frame = self.grab()
        if not self.loot_detector:
            self.transition(FieldState.PATROLLING)
            return

        item = self.loot_detector.find_nearest(frame, 0, 0)
        if item:
            self.pico.click(item.cx, item.cy, pulse_ms=50)
            self.loot_detector.invalidate()
        else:
            self._loot_rescan_cnt += 1
            if self._loot_rescan_cnt >= self._loot_rescan:
                self.transition(FieldState.PATROLLING)

    # ────────────────────────────── 공통 헬퍼 ───────────────────────

    def _check_done(self):
        if self.max_kills > 0 and self._kill_count >= self.max_kills:
            self.transition(FieldState.DONE)

    def _check_hp(self):
        if self.hp_reader is None:
            return
        now = time.monotonic()
        if now - self._last_potion < self._potion_cd:
            return
        frame = self.grab()
        self.hp_reader.read(frame)
        if self.hp_reader.is_low():
            self.pico.key_tap_name(self._key_potion, hold_ms=50)
            self._last_potion = now
            log.info("[Field] HP 낮음 → %s 물약 사용", self._key_potion)
