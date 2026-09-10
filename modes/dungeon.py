"""
modes/dungeon.py — 던전 반복 공략 모드.

전체 흐름
─────────
    IDLE
      │ start() 호출
      ▼
    ENTER_DUNGEON   ← 던전 입장 (텔레포트 or 직접 이동)
      ▼
    CLEARING        ← 적 탐지·공격 (tracker 주입 시) + 순찰 이동
      ▼
    LOOTING         ← 아데나/드롭 수집
      ▼
    EXIT_DUNGEON    ← 던전 퇴장 (텔레포트 or 대기)
      ▼
    COOLDOWN        ← 재입장 대기 (cooldown_ms)
      │ 반복 횟수 초과 시 → DONE
      ▼
    (ENTER_DUNGEON 반복 또는 DONE)

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


class DungeonState(Enum):
    IDLE          = auto()
    ENTER_DUNGEON = auto()
    CLEARING      = auto()
    LOOTING       = auto()
    EXIT_DUNGEON  = auto()
    COOLDOWN      = auto()
    DONE          = auto()


class DungeonMode(BaseFSM):
    """
    던전 반복 공략 FSM.

    Parameters
    ----------
    settings : Settings
        통합 설정 객체.
    pico : PicoWorker | NullPicoWorker
        키/클릭 입력 디바이스.
    frame_grabber : Callable[[], np.ndarray]
        BGR 프레임 반환 함수.
    tracker : NearestNeighborTracker | None
        적 추적기. 주입 시 CLEARING 에서 SM 이 공격 담당.
    hp_reader : HpReader | None
    loot_detector : LootDetector | None
    max_runs : int
        최대 반복 횟수. 0 = 무한.
    """

    def __init__(
        self,
        settings,
        pico,
        frame_grabber: Callable[[], np.ndarray],
        tracker=None,
        hp_reader=None,
        loot_detector=None,
        max_runs: int = 0,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.pico = pico
        self.grab = frame_grabber
        self.tracker = tracker
        self.hp_reader = hp_reader
        self.loot_detector = loot_detector
        self.max_runs = max_runs

        # ── 키 설정 ───────────────────────────────────────────────
        keys = settings.keys
        self._key_potion  = getattr(keys, "potion",  "F5")
        self._potion_cd   = getattr(keys, "potion_cooldown_ms", 3000) / 1000.0
        self._last_potion = 0.0

        # ── 던전 설정 ─────────────────────────────────────────────
        # settings 에 dungeon 섹션이 없으면 기본값 사용
        dungeon_cfg = getattr(settings, "dungeon", None)
        self._enter_timeout  = getattr(dungeon_cfg, "enter_timeout_ms",   8000) if dungeon_cfg else 8000
        self._enter_timeout /= 1000.0
        self._clear_timeout  = getattr(dungeon_cfg, "clear_timeout_ms",  60000) if dungeon_cfg else 60000
        self._clear_timeout /= 1000.0
        self._exit_timeout   = getattr(dungeon_cfg, "exit_timeout_ms",    5000) if dungeon_cfg else 5000
        self._exit_timeout  /= 1000.0
        self._cooldown_ms    = getattr(dungeon_cfg, "cooldown_ms",       10000) if dungeon_cfg else 10000
        self._cooldown_s     = self._cooldown_ms / 1000.0

        # 던전 입구/출구 좌표
        def _pt(d, dx=960, dy=540):
            if d is None:
                return (dx, dy)
            if isinstance(d, dict):
                return int(d.get("x", dx)), int(d.get("y", dy))
            return int(d[0]), int(d[1])

        self._enter_coord = _pt(getattr(dungeon_cfg, "enter_coord", None) if dungeon_cfg else None)
        self._exit_coord  = _pt(getattr(dungeon_cfg, "exit_coord",  None) if dungeon_cfg else None)

        # ── 루팅 설정 ─────────────────────────────────────────────
        loot_cfg = settings.loot
        self._loot_timeout  = getattr(loot_cfg, "timeout_ms",  8000) / 1000.0
        self._loot_rescan   = getattr(loot_cfg, "rescan_max",  3)
        self._loot_rescan_cnt = 0

        # ── 카운터 ────────────────────────────────────────────────
        self._run_count = 0
        self._running   = False

    # ────────────────────────────── 공개 API ────────────────────────

    def start(self) -> None:
        if self.is_state(DungeonState.IDLE):
            self._running = True
            self._run_count = 0
            self.transition(DungeonState.ENTER_DUNGEON)

    def stop(self) -> None:
        self._running = False
        if self.tracker:
            self.tracker.set_active(False)
        self.transition(DungeonState.IDLE, force=True)

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def is_done(self) -> bool:
        return self.is_state(DungeonState.DONE)

    # ────────────────────────────── BaseFSM ──────────────────────────

    def _initial_state(self):
        return DungeonState.IDLE

    def tick(self, dt: float) -> None:
        if not self._running and not self.is_state(DungeonState.IDLE):
            return

        s = self.state
        if s == DungeonState.IDLE:
            pass
        elif s == DungeonState.ENTER_DUNGEON:
            self._tick_enter()
        elif s == DungeonState.CLEARING:
            self._tick_clearing()
        elif s == DungeonState.LOOTING:
            self._tick_looting()
        elif s == DungeonState.EXIT_DUNGEON:
            self._tick_exit()
        elif s == DungeonState.COOLDOWN:
            self._tick_cooldown()
        elif s == DungeonState.DONE:
            pass

        self._check_hp()

    # ────────────────────────────── on_enter 훅 ─────────────────────

    def on_enter_ENTER_DUNGEON(self):
        log.info("[Dungeon] ENTER_DUNGEON (run #%d)", self._run_count + 1)
        ex, ey = self._enter_coord
        self.pico.click(ex, ey, pulse_ms=80)

    def on_enter_CLEARING(self):
        log.info("[Dungeon] CLEARING 시작")
        if self.tracker:
            self.tracker.set_active(True)

    def on_enter_LOOTING(self):
        log.info("[Dungeon] LOOTING 시작")
        self._loot_rescan_cnt = 0
        if self.tracker:
            self.tracker.set_active(False)
        if self.loot_detector:
            self.loot_detector.invalidate()

    def on_exit_LOOTING(self):
        if self.loot_detector:
            self.loot_detector.invalidate()

    def on_enter_EXIT_DUNGEON(self):
        log.info("[Dungeon] EXIT_DUNGEON")
        ex, ey = self._exit_coord
        self.pico.click(ex, ey, pulse_ms=80)

    def on_enter_COOLDOWN(self):
        self._run_count += 1
        log.info("[Dungeon] COOLDOWN — 완료 %d회 / 최대 %d회", self._run_count, self.max_runs)

    def on_enter_DONE(self):
        log.info("[Dungeon] DONE — %d회 완료", self._run_count)
        self._running = False
        if self.tracker:
            self.tracker.set_active(False)

    # ────────────────────────────── 상태별 tick ──────────────────────

    def _tick_enter(self):
        if self.state_elapsed_s >= self._enter_timeout:
            log.info("[Dungeon] 던전 입장 완료 (타임아웃)")
            self.transition(DungeonState.CLEARING)

    def _tick_clearing(self):
        frame = self.grab()

        # 루팅 전환 체크
        if self.loot_detector:
            items = self.loot_detector.find(frame)
            if items:
                self.transition(DungeonState.LOOTING)
                return

        # 클리어 타임아웃 → 퇴장
        if self.state_elapsed_s >= self._clear_timeout:
            log.info("[Dungeon] 클리어 타임아웃 → EXIT")
            self.transition(DungeonState.EXIT_DUNGEON)

    def _tick_looting(self):
        if self.state_elapsed_s >= self._loot_timeout:
            self.transition(DungeonState.EXIT_DUNGEON)
            return

        frame = self.grab()
        if not self.loot_detector:
            self.transition(DungeonState.EXIT_DUNGEON)
            return

        item = self.loot_detector.find_nearest(frame, 0, 0)
        if item:
            self.pico.click(item.cx, item.cy, pulse_ms=50)
            self.loot_detector.invalidate()
        else:
            self._loot_rescan_cnt += 1
            if self._loot_rescan_cnt >= self._loot_rescan:
                self.transition(DungeonState.EXIT_DUNGEON)

    def _tick_exit(self):
        if self.state_elapsed_s >= self._exit_timeout:
            self.transition(DungeonState.COOLDOWN)

    def _tick_cooldown(self):
        if self.state_elapsed_s >= self._cooldown_s:
            if self.max_runs > 0 and self._run_count >= self.max_runs:
                self.transition(DungeonState.DONE)
            else:
                self.transition(DungeonState.ENTER_DUNGEON)

    # ────────────────────────────── 공통 헬퍼 ───────────────────────

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
            log.info("[Dungeon] HP 낮음 → %s 물약 사용", self._key_potion)
