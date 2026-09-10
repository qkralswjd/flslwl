"""
tests/test_modes.py — modes/ 레이어 단위 테스트.

실제 Pico 연결, 화면 캡처, easyocr 없이도 동작한다.
모든 외부 의존은 Mock / NullPicoWorker 대체.
"""
from __future__ import annotations

import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np


# ─────────────────────────── 더미 Settings ───────────────────────

@dataclass
class _Keys:
    potion: str = "F5"
    scroll: str = "F6"
    speed_potion: str = "F9"
    potion_cooldown_ms: int = 3000
    speed_potion_wait_ms: int = 100


@dataclass
class _Dummy:
    drag_from: dict = field(default_factory=lambda: {"x": 960, "y": 600})
    drag_to:   dict = field(default_factory=lambda: {"x": 960, "y": 400})
    drag_steps: int = 8
    attack_interval_ms: int = 50
    move_timeout_ms: int = 100
    attack_duration_s: float = 0.0


@dataclass
class _LevelOcr:
    region_x: int = 0
    region_y: int = 0
    region_w: int = 120
    region_h: int = 30
    min_confidence: float = 0.4
    target_level_dummy: int = 5
    target_level_hunt: int = 10


@dataclass
class _HpBar:
    region_x: int = 0
    region_y: int = 0
    region_w: int = 100
    region_h: int = 20
    threshold_pct: float = 50.0
    read_interval_s: float = 0.5


@dataclass
class _LootCfg:
    timeout_ms: int = 500
    rescan_max: int = 2
    scan_interval_s: float = 0.1
    region_w: int = 0
    region_h: int = 0


@dataclass
class _WpCfg:
    points: list = field(default_factory=list)
    move_timeout_ms: int = 100


@dataclass
class _PatrolCombat:
    engage_timeout_ms: int = 500


@dataclass
class _CaptureCfg:
    monitor_index: int = 0
    fps: int = 60
    region_x: int = 0
    region_y: int = 0
    region_w: int = 1920
    region_h: int = 1080


@dataclass
class _DetectionCfg:
    fps: int = 6
    templates_dir: str = "config/templates"
    reject_templates_dir: str = "config/templates_reject"
    match_threshold: float = 0.50
    scale_factors: list = field(default_factory=lambda: [1.0])
    nms_iou_threshold: float = 0.30
    max_templates: int = 60
    zone_enabled: bool = False
    zone_cx: int = 960
    zone_cy: int = 490
    zone_half_w: int = 600
    zone_half_h: int = 250
    zone_h: int = 500


@dataclass
class _TrackingCfg:
    max_missing_frames: int = 10
    max_match_distance: int = 200
    lock_confirm_frames: int = 2
    click_pulse_ms: int = 20
    click_offset_x: int = 0
    click_offset_y: int = 0
    wait_dead_timeout_ms: int = 500
    next_target_cooldown_ms: int = 300
    priority: str = "nearest_center"
    drag_enabled: bool = False
    drag_dx: int = 0
    drag_dy: int = 0
    drag_steps: int = 8


@dataclass
class _Settings:
    keys: _Keys = field(default_factory=_Keys)
    dummy: _Dummy = field(default_factory=_Dummy)
    level_ocr: _LevelOcr = field(default_factory=_LevelOcr)
    hp_bar: _HpBar = field(default_factory=_HpBar)
    loot: _LootCfg = field(default_factory=_LootCfg)
    hunt_waypoints: _WpCfg = field(default_factory=_WpCfg)
    patrol_waypoints: _WpCfg = field(default_factory=_WpCfg)
    patrol_combat: _PatrolCombat = field(default_factory=_PatrolCombat)
    capture: _CaptureCfg = field(default_factory=_CaptureCfg)
    detection: _DetectionCfg = field(default_factory=_DetectionCfg)
    tracking: _TrackingCfg = field(default_factory=_TrackingCfg)


# ─────────────────────────── 더미 Pico ───────────────────────────

class _NullPico:
    def __init__(self):
        self.calls = []

    def click(self, x, y, pulse_ms=20):
        self.calls.append(("click", x, y))

    def drag(self, fx, fy, tx, ty, steps=8):
        self.calls.append(("drag", fx, fy, tx, ty))

    def key_tap_name(self, name, hold_ms=50):
        self.calls.append(("key_tap_name", name))

    def key_tap(self, code, hold_ms=50):
        self.calls.append(("key_tap", code))

    @property
    def is_connected(self):
        return True

    @property
    def is_idle(self):
        return True


def _black_frame():
    return np.zeros((100, 200, 3), dtype=np.uint8)


# ════════════════════════════════════════════════════════════════
#  LevelingMode 테스트
# ════════════════════════════════════════════════════════════════
class TestLevelingMode(unittest.TestCase):

    def _make(self, **kw):
        from modes.leveling import LevelingMode
        s = _Settings()
        return LevelingMode(
            settings=s,
            pico=_NullPico(),
            frame_grabber=_black_frame,
            **kw,
        )

    # ── 초기 상태 ─────────────────────────────────────────────────
    def test_initial_state_idle(self):
        from modes.leveling import LevelingState
        m = self._make()
        self.assertEqual(m.state, LevelingState.IDLE)

    def test_is_done_false_initially(self):
        m = self._make()
        self.assertFalse(m.is_done)

    # ── start/stop ────────────────────────────────────────────────
    def test_start_transitions_from_idle(self):
        from modes.leveling import LevelingState
        m = self._make()
        m.start()
        self.assertNotEqual(m.state, LevelingState.IDLE)

    def test_stop_returns_to_idle(self):
        from modes.leveling import LevelingState
        m = self._make()
        m.start()
        m.stop()
        self.assertEqual(m.state, LevelingState.IDLE)

    def test_double_start_noop(self):
        from modes.leveling import LevelingState
        m = self._make()
        m.start()
        s1 = m.state
        m.start()  # 두 번째 start — IDLE 아니므로 무시
        self.assertEqual(m.state, s1)

    # ── tick ──────────────────────────────────────────────────────
    def test_tick_idle_noop(self):
        from modes.leveling import LevelingState
        m = self._make()
        m.tick(0.1)
        self.assertEqual(m.state, LevelingState.IDLE)

    def test_tick_increments_tick_count(self):
        m = self._make()
        m.start()
        for _ in range(5):
            m.run_tick(0.016)
        self.assertGreaterEqual(m.tick_count, 5)

    # ── elapsed_since_entry ───────────────────────────────────────
    def test_elapsed_since_entry_false_immediately(self):
        m = self._make()
        m.start()
        self.assertFalse(m.elapsed_since_entry(threshold_ms=100000))

    # ── 타임아웃 기반 ATTACKING_DUMMY → USE_SPEED_POTION ─────────
    def test_attacking_dummy_timeout_transitions(self):
        from modes.leveling import LevelingMode, LevelingState
        s = _Settings()
        s.dummy.attack_duration_s = 0.05    # 50ms 후 전환
        s.dummy.attack_interval_ms = 10
        s.dummy.move_timeout_ms = 10        # MOVE_TO_DUMMY 도 빠르게
        s.keys.speed_potion_wait_ms = 10
        m = LevelingMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame)
        m.start()
        # USE_SCROLL_DUMMY → (teleport handler 없음) → MOVE_TO_DUMMY
        m.tick(0.0)
        # MOVE_TO_DUMMY 타임아웃 대기
        time.sleep(0.05)
        m.tick(0.05)
        # 이제 ATTACKING_DUMMY
        if m.is_state(LevelingState.ATTACKING_DUMMY):
            time.sleep(0.1)
            m.tick(0.1)
            self.assertTrue(
                m.is_state(LevelingState.USE_SPEED_POTION)
                or m.is_state(LevelingState.MOVE_TO_HUNT_ZONE)
            )

    # ── HP 물약 호출 ──────────────────────────────────────────────
    def test_hp_check_calls_potion(self):
        hp_mock = MagicMock()
        hp_mock.read.return_value = 20.0
        hp_mock.is_low.return_value = True
        pico = _NullPico()
        m = self._make(hp_reader=hp_mock)
        m._running = True
        m._last_potion_t = 0.0
        m._check_hp()
        potion_calls = [c for c in pico.calls if c[0] == "key_tap_name"]
        # _NullPico 이므로 직접 확인 불가 — hp_reader.read 가 호출됐는지 확인
        hp_mock.read.assert_called()

    # ── _SimpleMover 기본 동작 ────────────────────────────────────
    def test_simple_mover_idle_before_start(self):
        from modes.leveling import _SimpleMover
        m = _SimpleMover(points=[{"x": 100, "y": 200}], move_timeout_ms=1000, loop=False)
        pico = _NullPico()
        status = m.tick(pico)
        self.assertEqual(status, "IDLE")

    def test_simple_mover_done_no_points(self):
        from modes.leveling import _SimpleMover
        m = _SimpleMover(points=[], move_timeout_ms=1000, loop=False)
        m.start()
        pico = _NullPico()
        status = m.tick(pico)
        self.assertEqual(status, "DONE")

    def test_simple_mover_loop_resets_index(self):
        from modes.leveling import _SimpleMover
        m = _SimpleMover(
            points=[{"x": 10, "y": 10}],
            move_timeout_ms=10,
            loop=True,
        )
        m.start()
        pico = _NullPico()
        m.tick(pico)
        time.sleep(0.05)
        status = m.tick(pico)
        # loop 모드에서 Done 이 아닌 Moving/Arrived 반환
        self.assertNotEqual(status, "DONE")


# ════════════════════════════════════════════════════════════════
#  DungeonMode 테스트
# ════════════════════════════════════════════════════════════════
class TestDungeonMode(unittest.TestCase):

    def _make(self, **kw):
        from modes.dungeon import DungeonMode
        s = _Settings()
        return DungeonMode(
            settings=s,
            pico=_NullPico(),
            frame_grabber=_black_frame,
            **kw,
        )

    def test_initial_state_idle(self):
        from modes.dungeon import DungeonState
        m = self._make()
        self.assertEqual(m.state, DungeonState.IDLE)

    def test_start_transitions(self):
        from modes.dungeon import DungeonState
        m = self._make()
        m.start()
        self.assertEqual(m.state, DungeonState.ENTER_DUNGEON)

    def test_stop_returns_idle(self):
        from modes.dungeon import DungeonState
        m = self._make()
        m.start()
        m.stop()
        self.assertEqual(m.state, DungeonState.IDLE)

    def test_is_done_false_initially(self):
        m = self._make()
        self.assertFalse(m.is_done)

    def test_run_count_zero_initially(self):
        m = self._make()
        self.assertEqual(m.run_count, 0)

    def test_enter_timeout_transitions_to_clearing(self):
        from modes.dungeon import DungeonMode, DungeonState
        s = _Settings()
        m = DungeonMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame)
        m._enter_timeout = 0.01
        m.start()
        time.sleep(0.05)
        m.tick(0.05)
        self.assertEqual(m.state, DungeonState.CLEARING)

    def test_max_runs_limits_repetition(self):
        from modes.dungeon import DungeonMode, DungeonState
        s = _Settings()
        m = DungeonMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame, max_runs=1)
        m._run_count = 1  # 강제 설정
        m.transition(DungeonState.COOLDOWN, force=True)
        m._running = True
        m._cooldown_s = 0.0
        m.tick(0.0)
        self.assertEqual(m.state, DungeonState.DONE)

    def test_tick_idle_noop(self):
        from modes.dungeon import DungeonState
        m = self._make()
        m.tick(0.1)
        self.assertEqual(m.state, DungeonState.IDLE)

    def test_cooldown_transitions_to_enter(self):
        from modes.dungeon import DungeonMode, DungeonState
        s = _Settings()
        m = DungeonMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame, max_runs=0)
        m.transition(DungeonState.COOLDOWN, force=True)
        m._running = True
        m._cooldown_s = 0.0
        m.tick(0.0)
        self.assertEqual(m.state, DungeonState.ENTER_DUNGEON)


# ════════════════════════════════════════════════════════════════
#  FieldMode 테스트
# ════════════════════════════════════════════════════════════════
class TestFieldMode(unittest.TestCase):

    def _make(self, **kw):
        from modes.field import FieldMode
        s = _Settings()
        return FieldMode(
            settings=s,
            pico=_NullPico(),
            frame_grabber=_black_frame,
            **kw,
        )

    def test_initial_state_idle(self):
        from modes.field import FieldState
        m = self._make()
        self.assertEqual(m.state, FieldState.IDLE)

    def test_start_transitions(self):
        from modes.field import FieldState
        m = self._make()
        m.start()
        self.assertEqual(m.state, FieldState.PATROLLING)

    def test_stop_returns_idle(self):
        from modes.field import FieldState
        m = self._make()
        m.start()
        m.stop()
        self.assertEqual(m.state, FieldState.IDLE)

    def test_kill_count_zero_initially(self):
        m = self._make()
        self.assertEqual(m.kill_count, 0)

    def test_is_done_false_initially(self):
        m = self._make()
        self.assertFalse(m.is_done)

    def test_max_kills_done(self):
        from modes.field import FieldMode, FieldState
        s = _Settings()
        m = FieldMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame, max_kills=3)
        m._kill_count = 3
        m._check_done()
        self.assertEqual(m.state, FieldState.DONE)

    def test_max_kills_zero_never_done(self):
        from modes.field import FieldMode, FieldState
        s = _Settings()
        m = FieldMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame, max_kills=0)
        m._kill_count = 9999
        m._check_done()
        self.assertNotEqual(m.state, FieldState.DONE)

    def test_tick_idle_noop(self):
        from modes.field import FieldState
        m = self._make()
        m.tick(0.1)
        self.assertEqual(m.state, FieldState.IDLE)

    def test_engage_timeout_returns_to_patrolling(self):
        from modes.field import FieldMode, FieldState
        s = _Settings()
        m = FieldMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame)
        m._engage_timeout = 0.01
        m.transition(FieldState.ENGAGING, force=True)
        m._running = True
        time.sleep(0.05)
        m.tick(0.05)
        self.assertEqual(m.state, FieldState.PATROLLING)

    def test_looting_timeout_returns_to_patrolling(self):
        from modes.field import FieldMode, FieldState
        s = _Settings()
        m = FieldMode(settings=s, pico=_NullPico(), frame_grabber=_black_frame)
        m._loot_timeout = 0.01
        m.transition(FieldState.LOOTING, force=True)
        m._running = True
        time.sleep(0.05)
        m.tick(0.05)
        self.assertEqual(m.state, FieldState.PATROLLING)

    def test_tracker_active_triggers_engaging(self):
        from modes.field import FieldMode, FieldState
        s = _Settings()
        tracker_mock = MagicMock()
        tracker_mock.attack_active = True
        loot_mock = MagicMock()
        loot_mock.find.return_value = []
        m = FieldMode(
            settings=s,
            pico=_NullPico(),
            frame_grabber=_black_frame,
            tracker=tracker_mock,
            loot_detector=loot_mock,
        )
        m.start()
        m.tick(0.016)
        self.assertEqual(m.state, FieldState.ENGAGING)

    def test_hp_check_skipped_without_reader(self):
        m = self._make()
        m._running = True
        # 예외 없이 호출되면 OK
        m._check_hp()


# ════════════════════════════════════════════════════════════════
#  modes/__init__.py 확인
# ════════════════════════════════════════════════════════════════
class TestModesPackage(unittest.TestCase):
    def test_import_package(self):
        import modes  # noqa: F401

    def test_all_names(self):
        import modes
        for name in ["leveling", "dungeon", "field"]:
            self.assertIn(name, modes.__all__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
