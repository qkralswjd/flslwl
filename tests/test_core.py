"""tests/test_core.py — core 레이어 유닛 테스트.

cv2 / numpy 없이 실행 가능한 순수 로직 테스트:
  - Detection 데이터 클래스
  - _nms_boxes 알고리즘
  - Enemy 상태 갱신 / 예측
  - SequentialTargetStateMachine 상태 전환 흐름
  - NearestNeighborTracker 매칭 / 소실 / 신규 등록
  - BaseFSM 전환·훅·편의 메서드
"""
from __future__ import annotations

import sys
import time
import types
import unittest
from enum import Enum, auto
from unittest.mock import MagicMock, patch


# ── cv2 / numpy 스텁 ─────────────────────────────────────────────────────
def _install_cv2_numpy_stubs():
    np_mod = types.ModuleType("numpy")
    import numpy as _np  # 실제 numpy 는 있음
    sys.modules.setdefault("numpy", _np)

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.COLOR_BGR2GRAY   = 6
    cv2_mod.TM_CCOEFF_NORMED = 5
    cv2_mod.INTER_LINEAR     = 1
    cv2_mod.FONT_HERSHEY_SIMPLEX = 0
    cv2_mod.LINE_AA          = 16
    cv2_mod.cvtColor         = MagicMock(return_value=_np.zeros((100, 100), dtype="uint8"))
    cv2_mod.matchTemplate    = MagicMock(return_value=_np.zeros((10, 10), dtype="float32"))
    cv2_mod.resize           = MagicMock(side_effect=lambda img, size, **kw: img)
    cv2_mod.rectangle        = MagicMock()
    cv2_mod.putText          = MagicMock()
    cv2_mod.imdecode         = MagicMock(return_value=None)
    sys.modules.setdefault("cv2", cv2_mod)

_install_cv2_numpy_stubs()


# ════════════════════════════════════════════════════════════════════════════
# Detection 테스트
# ════════════════════════════════════════════════════════════════════════════

class TestDetection(unittest.TestCase):

    def _make(self, x=10, y=20, w=30, h=40, conf=0.8):
        from core.detection import Detection
        return Detection(x, y, w, h, conf)

    def test_center_computed(self):
        d = self._make(x=10, y=20, w=30, h=40)
        self.assertEqual(d.center_x, 25)   # 10 + 30//2
        self.assertEqual(d.center_y, 40)   # 20 + 40//2

    def test_area_computed(self):
        d = self._make(w=30, h=40)
        self.assertEqual(d.area, 1200)

    def test_confidence_stored(self):
        d = self._make(conf=0.75)
        self.assertAlmostEqual(d.confidence, 0.75)

    def test_default_confidence(self):
        from core.detection import Detection
        d = Detection(0, 0, 10, 10)
        self.assertAlmostEqual(d.confidence, 1.0)

    def test_repr_contains_class_name(self):
        d = self._make()
        self.assertIn("Detection", repr(d))


# ════════════════════════════════════════════════════════════════════════════
# NMS 알고리즘 테스트
# ════════════════════════════════════════════════════════════════════════════

class TestNmsBoxes(unittest.TestCase):

    def _nms(self, boxes, iou=0.3):
        from core.detection import _nms_boxes
        return _nms_boxes(boxes, iou)

    def test_empty_input(self):
        self.assertEqual(self._nms([]), [])

    def test_single_box_kept(self):
        boxes = [(0, 0, 10, 10, 0.9)]
        result = self._nms(boxes)
        self.assertEqual(len(result), 1)

    def test_non_overlapping_both_kept(self):
        # 두 박스가 전혀 겹치지 않음
        boxes = [(0, 0, 10, 10, 0.9), (100, 100, 10, 10, 0.8)]
        result = self._nms(boxes)
        self.assertEqual(len(result), 2)

    def test_identical_boxes_one_kept(self):
        # 완전히 겹치는 두 박스 → 높은 score 만 남음
        boxes = [(5, 5, 10, 10, 0.9), (5, 5, 10, 10, 0.5)]
        result = self._nms(boxes, iou=0.3)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][4], 0.9)

    def test_partial_overlap_suppressed(self):
        # IoU > 0.3 인 박스는 억제됨
        boxes = [(0, 0, 100, 100, 0.9), (10, 10, 100, 100, 0.7)]
        result = self._nms(boxes, iou=0.3)
        self.assertEqual(len(result), 1)

    def test_sorted_by_score_descending(self):
        boxes = [(0, 0, 5, 5, 0.3), (50, 0, 5, 5, 0.9), (100, 0, 5, 5, 0.6)]
        result = self._nms(boxes, iou=0.3)
        scores = [b[4] for b in result]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ════════════════════════════════════════════════════════════════════════════
# Enemy 테스트
# ════════════════════════════════════════════════════════════════════════════

class TestEnemy(unittest.TestCase):

    def _make(self, **kw):
        from core.tracking import Enemy
        defaults = dict(id=1, x=100, y=200, width=30, height=40,
                        center_x=115, center_y=220, confidence=0.9)
        defaults.update(kw)
        return Enemy(**defaults)

    def test_initial_state(self):
        e = self._make()
        self.assertEqual(e.missing_frames, 0)
        self.assertFalse(e.predicted)

    def test_update_position_updates_center(self):
        e = self._make()
        e.update_position(200, 300, 30, 40, dt=0.1)
        self.assertEqual(e.center_x, 215)
        self.assertEqual(e.center_y, 320)
        self.assertEqual(e.missing_frames, 0)
        self.assertFalse(e.predicted)

    def test_update_position_computes_velocity(self):
        e = self._make(center_x=100, center_y=200)
        e.update_position(110, 200, 10, 10, dt=0.1)
        # center 는 (115, 205), velocity_x = 115-100=15
        self.assertAlmostEqual(e.velocity_x, 15)

    def test_apply_prediction_increments_missing(self):
        e = self._make()
        e.apply_prediction()
        self.assertEqual(e.missing_frames, 1)
        self.assertTrue(e.predicted)

    def test_apply_prediction_decays_velocity(self):
        e = self._make()
        e.velocity_x = 10.0
        e.apply_prediction()
        self.assertAlmostEqual(e.velocity_x, 10.0 * 0.75)

    def test_history_appended_on_update(self):
        e = self._make()
        e.update_position(0, 0, 10, 10, dt=0.1)
        self.assertEqual(len(e.history), 1)

    def test_velocity_decay_field_default(self):
        e = self._make()
        self.assertAlmostEqual(e.VELOCITY_DECAY, 0.75)


# ════════════════════════════════════════════════════════════════════════════
# TargetState 열거형
# ════════════════════════════════════════════════════════════════════════════

class TestTargetState(unittest.TestCase):

    def test_all_states_exist(self):
        from core.tracking import TargetState
        names = {s.name for s in TargetState}
        self.assertIn("IDLE", names)
        self.assertIn("LOCKING", names)
        self.assertIn("CLICKING", names)
        self.assertIn("WAITING_DEAD", names)
        self.assertIn("COOLDOWN", names)


# ════════════════════════════════════════════════════════════════════════════
# SequentialTargetStateMachine 테스트
# ════════════════════════════════════════════════════════════════════════════

class TestSequentialTargetSM(unittest.TestCase):

    def _make_sm(self, click_cb=None):
        from core.tracking import SequentialTargetStateMachine
        return SequentialTargetStateMachine(
            pico_click_callback  = click_cb or MagicMock(),
            to_screen_fn         = lambda x, y: (x, y),
            lock_confirm_frames  = 2,
            wait_dead_timeout_ms = 5000.0,
            next_target_cooldown_ms = 200.0,
        )

    def _make_enemy(self, eid=1, cx=100, cy=200):
        from core.tracking import Enemy
        return Enemy(
            id=eid, x=cx-5, y=cy-5, width=10, height=10,
            center_x=cx, center_y=cy,
        )

    def test_initial_state_idle(self):
        from core.tracking import TargetState
        sm = self._make_sm()
        self.assertEqual(sm.state, TargetState.IDLE)
        self.assertIsNone(sm.target_id)

    def test_idle_to_locking_on_enemy_appear(self):
        from core.tracking import TargetState
        sm = self._make_sm()
        e = self._make_enemy()
        sm.update({e.id: e})
        self.assertEqual(sm.state, TargetState.LOCKING)
        self.assertEqual(sm.target_id, e.id)

    def test_locking_to_clicking_after_confirm_frames(self):
        from core.tracking import TargetState
        cb = MagicMock()
        sm = self._make_sm(click_cb=cb)
        e = self._make_enemy()
        enemies = {e.id: e}
        sm.update(enemies)  # IDLE → LOCKING (streak=0)
        sm.update(enemies)  # LOCKING streak=1
        sm.update(enemies)  # LOCKING streak=2 → CLICKING (fire)
        self.assertEqual(sm.state, TargetState.CLICKING)
        cb.assert_called_once()

    def test_clicking_to_waiting_dead(self):
        from core.tracking import TargetState
        sm = self._make_sm()
        e = self._make_enemy()
        enemies = {e.id: e}
        for _ in range(3):
            sm.update(enemies)
        # 3번 호출 후 CLICKING
        sm.update(enemies)  # CLICKING → WAITING_DEAD
        self.assertEqual(sm.state, TargetState.WAITING_DEAD)

    def test_waiting_dead_to_cooldown_when_enemy_gone(self):
        from core.tracking import TargetState
        sm = self._make_sm()
        e = self._make_enemy()
        enemies = {e.id: e}
        for _ in range(4):
            sm.update(enemies)
        # 이제 WAITING_DEAD 상태 — 적 제거
        sm.update({})  # enemy gone → COOLDOWN
        self.assertEqual(sm.state, TargetState.COOLDOWN)

    def test_inactive_blocks_attack(self):
        from core.tracking import TargetState
        cb = MagicMock()
        sm = self._make_sm(click_cb=cb)
        sm.set_active(False)
        e = self._make_enemy()
        sm.update({e.id: e})
        self.assertEqual(sm.state, TargetState.IDLE)
        cb.assert_not_called()

    def test_set_active_false_resets(self):
        from core.tracking import TargetState
        sm = self._make_sm()
        e = self._make_enemy()
        sm.update({e.id: e})
        self.assertEqual(sm.state, TargetState.LOCKING)
        sm.set_active(False)
        self.assertEqual(sm.state, TargetState.IDLE)
        self.assertIsNone(sm.target_id)

    def test_reset_clears_state(self):
        from core.tracking import TargetState
        sm = self._make_sm()
        e = self._make_enemy()
        sm.update({e.id: e})
        sm.reset()
        self.assertEqual(sm.state, TargetState.IDLE)
        self.assertIsNone(sm.target_id)
        self.assertEqual(sm._lock_streak, 0)


# ════════════════════════════════════════════════════════════════════════════
# NearestNeighborTracker 테스트
# ════════════════════════════════════════════════════════════════════════════

class TestNearestNeighborTracker(unittest.TestCase):

    def _make_tracker(self, click_cb=None):
        from core.tracking import NearestNeighborTracker
        return NearestNeighborTracker(
            max_missing_frames = 3,
            max_match_distance = 50,
            pico_click_callback = click_cb or MagicMock(),
        )

    def _det(self, x, y, w=20, h=20, conf=0.9):
        from core.detection import Detection
        return Detection(x, y, w, h, conf)

    def test_new_detection_creates_enemy(self):
        t = self._make_tracker()
        result = t.update([self._det(100, 200)], dt=0.1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(t.enemies), 1)

    def test_matching_detection_updates_position(self):
        t = self._make_tracker()
        t.update([self._det(100, 200)], dt=0.1)
        eid = list(t.enemies.keys())[0]
        t.update([self._det(105, 205)], dt=0.1)
        self.assertEqual(t.enemies[eid].center_x, 115)   # 105+20//2

    def test_missing_enemy_increments_missing_frames(self):
        t = self._make_tracker()
        t.update([self._det(100, 200)], dt=0.1)
        eid = list(t.enemies.keys())[0]
        t.update([], dt=0.1)  # 탐지 없음
        self.assertEqual(t.enemies[eid].missing_frames, 1)
        self.assertTrue(t.enemies[eid].predicted)

    def test_enemy_removed_after_max_missing(self):
        t = self._make_tracker()
        t.update([self._det(100, 200)], dt=0.1)
        eid = list(t.enemies.keys())[0]
        for _ in range(4):  # max_missing_frames=3, 4번 누락 → 제거
            t.update([], dt=0.1)
        self.assertNotIn(eid, t.enemies)

    def test_two_detections_two_enemies(self):
        t = self._make_tracker()
        result = t.update([self._det(0, 0), self._det(200, 200)], dt=0.1)
        self.assertEqual(len(result), 2)

    def test_far_detection_creates_new_enemy(self):
        t = self._make_tracker()
        t.update([self._det(0, 0)], dt=0.1)
        # max_match_distance=50 보다 먼 탐지 → 신규 적
        t.update([self._det(200, 200)], dt=0.1)
        self.assertEqual(len(t.enemies), 2)

    def test_set_active_propagates_to_sm(self):
        t = self._make_tracker()
        t.set_active(False)
        self.assertFalse(t.attack_active)
        t.set_active(True)
        self.assertTrue(t.attack_active)

    def test_to_screen_offset(self):
        from core.tracking import NearestNeighborTracker
        t = NearestNeighborTracker(
            roi_offset=(10, 20),
            capture_region_offset=(100, 200),
        )
        sx, sy = t._to_screen(5, 10)
        self.assertEqual(sx, 5 + 10 + 100)
        self.assertEqual(sy, 10 + 20 + 200)


# ════════════════════════════════════════════════════════════════════════════
# BaseFSM 테스트
# ════════════════════════════════════════════════════════════════════════════

class TestBaseFSM(unittest.TestCase):

    class _DemoState(Enum):
        IDLE    = auto()
        RUNNING = auto()
        DONE    = auto()

    class _DemoFSM(  # noqa: N801
        # 로컬 import 후 정의
    ):
        pass

    def setUp(self):
        from core.state import BaseFSM

        class _DemoFSM(BaseFSM):
            S = TestBaseFSM._DemoState

            def _initial_state(self):
                return self.S.IDLE

            def tick(self, dt):
                pass

        self.FSM = _DemoFSM

    def test_initial_state(self):
        fsm = self.FSM()
        self.assertEqual(fsm.state, self.FSM.S.IDLE)

    def test_transition_changes_state(self):
        fsm = self.FSM()
        changed = fsm.transition(self.FSM.S.RUNNING)
        self.assertTrue(changed)
        self.assertEqual(fsm.state, self.FSM.S.RUNNING)

    def test_transition_same_state_returns_false(self):
        fsm = self.FSM()
        changed = fsm.transition(self.FSM.S.IDLE)
        self.assertFalse(changed)

    def test_transition_force_same_state(self):
        fsm = self.FSM()
        changed = fsm.transition(self.FSM.S.IDLE, force=True)
        self.assertTrue(changed)

    def test_prev_state_updated(self):
        fsm = self.FSM()
        fsm.transition(self.FSM.S.RUNNING)
        self.assertEqual(fsm.prev_state, self.FSM.S.IDLE)

    def test_state_elapsed_increases(self):
        fsm = self.FSM()
        time.sleep(0.05)
        self.assertGreater(fsm.state_elapsed_ms, 0)

    def test_elapsed_resets_on_transition(self):
        fsm = self.FSM()
        time.sleep(0.05)
        fsm.transition(self.FSM.S.RUNNING)
        self.assertLess(fsm.state_elapsed_ms, 50)

    def test_is_state(self):
        fsm = self.FSM()
        self.assertTrue(fsm.is_state(self.FSM.S.IDLE))
        self.assertFalse(fsm.is_state(self.FSM.S.RUNNING))

    def test_tick_count_increments(self):
        fsm = self.FSM()
        for _ in range(5):
            fsm.run_tick(0.1)
        self.assertEqual(fsm.tick_count, 5)

    def test_on_enter_hook_called(self):
        from core.state import BaseFSM

        entered = []

        class HookFSM(BaseFSM):
            S = TestBaseFSM._DemoState

            def _initial_state(self):
                return self.S.IDLE

            def tick(self, dt):
                pass

            def on_enter_RUNNING(self):
                entered.append(True)

        fsm = HookFSM()
        fsm.transition(HookFSM.S.RUNNING)
        self.assertEqual(len(entered), 1)

    def test_on_exit_hook_called(self):
        from core.state import BaseFSM

        exited = []

        class HookFSM(BaseFSM):
            S = TestBaseFSM._DemoState

            def _initial_state(self):
                return self.S.IDLE

            def tick(self, dt):
                pass

            def on_exit_IDLE(self):
                exited.append(True)

        fsm = HookFSM()
        fsm.transition(HookFSM.S.RUNNING)
        self.assertEqual(len(exited), 1)

    def test_elapsed_since_entry_true(self):
        fsm = self.FSM()
        time.sleep(0.06)
        self.assertTrue(fsm.elapsed_since_entry(50))

    def test_elapsed_since_entry_false(self):
        fsm = self.FSM()
        self.assertFalse(fsm.elapsed_since_entry(10000))

    def test_repr_contains_state_name(self):
        fsm = self.FSM()
        self.assertIn("IDLE", repr(fsm))


if __name__ == "__main__":
    unittest.main(verbosity=2)
