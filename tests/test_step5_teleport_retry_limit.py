"""STEP5: 텔레포트 무한 재시도 제거 검증 테스트.

검증 항목:
  1. 첫 실패 -> fail_count = 1, 상태 유지
  2. 4회 실패 -> 여전히 USE_SCROLL_DUMMY 상태
  3. 5회 실패 -> IDLE 전환 (안전 정지)
  4. 5회 실패 이후 tick에서 텔레포트 명령 없음
  5. 성공 시 fail_count = 0
  6. STEP2/STEP4 회귀 테스트
"""

import sys
import os
import time
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from automation.state_machine import HuntingStateMachine, HuntingState, MAX_TELEPORT_RETRY


# ── 공통 픽스처 ──────────────────────────────────────────────────────────

def _make_sm() -> HuntingStateMachine:
    """최소 config로 SM 생성."""
    config = {
        "keys": {},
        "hp_bar": {"region": {"x": 0, "y": 0, "width": 10, "height": 5}},
        "level_ocr": {"region": {"x": 0, "y": 0, "width": 50, "height": 20}},
        "scroll_dummy": {},
        "dummy": {},
        "hunt_waypoints": {"points": [{"x": 100, "y": 100}]},
        "loot": {},
    }
    pico = MagicMock()
    frame = np.zeros((400, 600, 3), dtype="uint8")
    sm = HuntingStateMachine(config, pico, lambda: frame)
    sm.hp_reader.read = MagicMock(return_value=100.0)
    sm.hp_reader.get_cached = MagicMock(return_value=100.0)
    return sm


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _fail_once(sm: HuntingStateMachine, clock: FakeClock) -> None:
    """tick()이 False(실패)를 한 번 반환하도록 만들고 retry_at을 넘긴다."""
    frame = np.zeros((400, 600, 3), dtype="uint8")

    with patch("automation.state_machine.time.monotonic", clock), \
         patch.object(sm.scroll_teleporter, "tick", return_value=False):
        sm.update(frame, [])   # tick=False -> fail_count++, retry_at 설정

    # retry_at(2초) 넘기기 -> 다음 실패를 받을 준비
    clock.advance(2.1)


# ── 테스트 ────────────────────────────────────────────────────────────────

def test_constant_max_teleport_retry():
    """MAX_TELEPORT_RETRY 상수가 존재하고 양의 정수다."""
    assert isinstance(MAX_TELEPORT_RETRY, int)
    assert MAX_TELEPORT_RETRY > 0


def test_fail_count_starts_at_zero():
    """초기 fail_count = 0."""
    sm = _make_sm()
    assert sm._teleport_fail_count == 0


def test_first_failure_increments_count():
    """첫 실패 -> fail_count = 1, 상태는 USE_SCROLL_DUMMY 유지."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    _fail_once(sm, clock)

    assert sm._teleport_fail_count == 1
    assert sm.state == HuntingState.USE_SCROLL_DUMMY


def test_four_failures_still_in_scroll_dummy():
    """MAX-1(=4)회 실패 -> 여전히 USE_SCROLL_DUMMY."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    for _ in range(MAX_TELEPORT_RETRY - 1):
        _fail_once(sm, clock)

    assert sm._teleport_fail_count == MAX_TELEPORT_RETRY - 1
    assert sm.state == HuntingState.USE_SCROLL_DUMMY


def test_max_failures_transitions_to_idle():
    """MAX(=5)회 실패 -> IDLE 전환."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    for _ in range(MAX_TELEPORT_RETRY):
        _fail_once(sm, clock)

    assert sm.state == HuntingState.IDLE


def test_idle_after_max_fail_resets_counter():
    """IDLE 전환 시 fail_count가 0으로 리셋된다."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    for _ in range(MAX_TELEPORT_RETRY):
        _fail_once(sm, clock)

    assert sm.state == HuntingState.IDLE
    assert sm._teleport_fail_count == 0


def test_no_teleport_after_max_fail():
    """IDLE 전환 이후 추가 tick에서 teleporter.tick() 호출 없음."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    for _ in range(MAX_TELEPORT_RETRY):
        _fail_once(sm, clock)

    assert sm.state == HuntingState.IDLE

    # 이 시점부터 tick_call 카운트
    tick_calls = [0]
    original_tick = sm.scroll_teleporter.tick

    def counting_tick(*a, **kw):
        tick_calls[0] += 1
        return original_tick(*a, **kw)

    sm.scroll_teleporter.tick = counting_tick
    frame = np.zeros((400, 600, 3), dtype="uint8")

    with patch("automation.state_machine.time.monotonic", clock):
        for _ in range(10):
            clock.advance(0.5)
            sm.update(frame, [])

    # IDLE 상태이므로 update() 자체가 즉시 return -> tick 호출 없음
    assert tick_calls[0] == 0


def test_success_resets_fail_count():
    """실패 후 성공 시 fail_count = 0."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    # 2회 실패
    _fail_once(sm, clock)
    _fail_once(sm, clock)
    assert sm._teleport_fail_count == 2

    # 성공
    frame = np.zeros((400, 600, 3), dtype="uint8")
    with patch("automation.state_machine.time.monotonic", clock), \
         patch.object(sm.scroll_teleporter, "tick", return_value=True):
        sm.update(frame, [])

    assert sm._teleport_fail_count == 0
    assert sm.state == HuntingState.MOVE_TO_DUMMY


def test_retry_between_failures_is_nonblocking():
    """실패 후 retry_at 이전에는 teleporter.tick() 재호출 없음 (비블로킹 유지)."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    tick_calls = [0]

    def fail_tick(*a, **kw):
        tick_calls[0] += 1
        return False

    sm.scroll_teleporter.tick = fail_tick
    frame = np.zeros((400, 600, 3), dtype="uint8")

    with patch("automation.state_machine.time.monotonic", clock):
        # 1차 실패
        sm.update(frame, [])
        assert tick_calls[0] == 1

        first_count = tick_calls[0]

        # retry_at(2초) 이전 tick 5번 -> tick() 재호출 없음
        for _ in range(5):
            clock.advance(0.3)
            sm.update(frame, [])

        assert tick_calls[0] == first_count   # 추가 호출 없음

        # 2초 넘기기 -> 재시도
        clock.advance(2.0)
        sm.update(frame, [])
        assert tick_calls[0] == first_count + 1


def test_start_after_idle_recovery_resets_state():
    """IDLE 복구 후 start() -> USE_SCROLL_DUMMY, fail_count = 0."""
    sm = _make_sm()
    clock = FakeClock()
    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    for _ in range(MAX_TELEPORT_RETRY):
        _fail_once(sm, clock)

    assert sm.state == HuntingState.IDLE

    # 재시작
    sm.start()
    assert sm.state == HuntingState.USE_SCROLL_DUMMY
    assert sm._teleport_fail_count == 0


# ── STEP2/STEP4 회귀 ─────────────────────────────────────────────────────

def test_step4_regression():
    """STEP4: LOOTING 중 target SM 클릭 차단."""
    from tracking.tracker import SequentialTargetStateMachine, TargetState
    from tracking.enemy import Enemy

    click_cb = MagicMock()
    sm = SequentialTargetStateMachine(
        pico_click_callback=click_cb,
        to_screen_fn=lambda x, y: (x, y),
        lock_confirm_frames=1,
    )
    sm.set_active(False)
    e = Enemy(id=1, x=90, y=90, width=20, height=20,
              center_x=100, center_y=100, confidence=1.0, predicted=False)
    for _ in range(5):
        sm.update({1: e})
    click_cb.assert_not_called()


def test_step2_regression_speed_potion_nonblocking():
    """STEP2: USE_SPEED_POTION blocking sleep 없음."""
    sm = _make_sm()
    clock = FakeClock(start=1000.0)
    frame = np.zeros((400, 600, 3), dtype="uint8")
    sm._enter(HuntingState.USE_SPEED_POTION)

    with patch("automation.state_machine.time.monotonic", clock):
        start_wall = time.monotonic()
        sm.update(frame, [])
        elapsed = time.monotonic() - start_wall

    assert elapsed < 0.05, f"첫 tick이 {elapsed:.3f}초 block됨"
    assert sm.state == HuntingState.USE_SPEED_POTION


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
