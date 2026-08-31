"""STEP2: blocking sleep 제거 검증 테스트.

검증 항목:
  1. 텔레포트 실패 시 호출 thread가 2초 block되지 않는다 (retry_at 방식)
  2. retry_at 이전에는 TeleportHandler.tick() 재호출 없음
  3. retry_at 이후 다음 tick에서 재시도된다
  4. 성공 시 기존 성공 흐름 유지
  5. 실패 retry 횟수 동작이 기존과 동일
  6. speed_potion 대기도 비블로킹

모든 시간 의존 테스트는 가짜 clock(monkeypatch)을 사용해 실제 대기 없음.
"""

import sys
import os
import time
from typing import Optional
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from automation.teleport_handler import TeleportHandler, _TPhase


# ────────────────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────────────────

def _make_handler(
    wait_after_key_ms: int = 800,
    wait_after_click_ms: int = 2000,
    max_retries: int = 3,
) -> TeleportHandler:
    return TeleportHandler(
        key="F6",
        destination_region={"x": 0, "y": 0, "width": 300, "height": 400},
        destination_text="허수아비",
        capture_offset=(0, 0),
        wait_after_key_ms=wait_after_key_ms,
        wait_after_click_ms=wait_after_click_ms,
        max_retries=max_retries,
    )


def _fake_frame():
    import numpy as np
    return np.zeros((400, 300, 3), dtype="uint8")


# ────────────────────────────────────────────────────────────
# TeleportHandler.tick() 비블로킹 단계 테스트
# ────────────────────────────────────────────────────────────

class FakeClock:
    """monotonic 시계를 수동으로 제어하는 가짜 클락."""
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_tick_idle_sends_key_and_returns_none():
    """IDLE tick: 키 입력 후 None 반환 (WAIT_WINDOW 진입)."""
    h = _make_handler(wait_after_key_ms=800)
    pico = MagicMock()
    clock = FakeClock()

    with patch("automation.teleport_handler.time.monotonic", clock):
        result = h.tick(_fake_frame, pico)

    assert result is None
    pico.key_tap_name.assert_called_once_with("F6", hold_ms=80)
    assert h._phase == _TPhase.WAIT_WINDOW


def test_tick_wait_window_blocks_until_time():
    """WAIT_WINDOW: phase_until 전이면 계속 None."""
    h = _make_handler(wait_after_key_ms=800)
    pico = MagicMock()
    clock = FakeClock(start=1000.0)

    with patch("automation.teleport_handler.time.monotonic", clock):
        h.tick(_fake_frame, pico)          # IDLE -> WAIT_WINDOW (phase_until=1000.8)

        # 시간 진행 없이 tick -> 여전히 대기 중
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.WAIT_WINDOW

        # 0.5초만 진행 (아직 대기 중)
        clock.advance(0.5)
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.WAIT_WINDOW

        # 0.4초 더 진행 -> 0.9초 총 경과 -> WAIT_WINDOW 통과
        clock.advance(0.4)
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.FIND_DESTINATION


def test_tick_find_destination_success():
    """OCR 성공 시 클릭 후 WAIT_COMPLETE로 이동."""
    h = _make_handler(wait_after_key_ms=0, wait_after_click_ms=2000)
    pico = MagicMock()
    clock = FakeClock(start=1000.0)

    # _find_destination이 (100, 200) 반환하도록 패치
    with patch("automation.teleport_handler.time.monotonic", clock), \
         patch.object(h, "_find_destination", return_value=(100, 200)):

        h.tick(_fake_frame, pico)   # IDLE -> WAIT_WINDOW (wait=0이므로 즉시)
        h.tick(_fake_frame, pico)   # WAIT_WINDOW -> FIND_DESTINATION
        result = h.tick(_fake_frame, pico)  # FIND_DESTINATION -> WAIT_COMPLETE

    assert result is None
    pico.click.assert_called_once_with(100, 200)
    assert h._phase == _TPhase.WAIT_COMPLETE


def test_tick_wait_complete_then_success():
    """WAIT_COMPLETE 경과 후 True 반환."""
    h = _make_handler(wait_after_key_ms=0, wait_after_click_ms=2000)
    pico = MagicMock()
    clock = FakeClock(start=1000.0)

    with patch("automation.teleport_handler.time.monotonic", clock), \
         patch.object(h, "_find_destination", return_value=(100, 200)):

        h.tick(_fake_frame, pico)   # IDLE -> WAIT_WINDOW
        h.tick(_fake_frame, pico)   # WAIT_WINDOW -> FIND_DESTINATION
        h.tick(_fake_frame, pico)   # FIND_DESTINATION -> WAIT_COMPLETE (phase_until=1002)

        # 1초 경과 - 아직 대기 중
        clock.advance(1.0)
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.WAIT_COMPLETE

        # 1.1초 더 경과 (총 2.1초) -> SUCCESS
        clock.advance(1.1)
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.SUCCESS

        # SUCCESS tick -> True
        result = h.tick(_fake_frame, pico)
        assert result is True


def test_tick_ocr_retry_nonblocking():
    """OCR 실패 시 WAIT_RETRY -> 0.5초 비블로킹 대기 후 재시도."""
    h = _make_handler(wait_after_key_ms=0, max_retries=3)
    pico = MagicMock()
    clock = FakeClock(start=1000.0)

    fail_then_succeed = [None, (100, 200)]  # 첫 번째 실패, 두 번째 성공
    call_count = [0]

    def mock_find(frame):
        result = fail_then_succeed[min(call_count[0], 1)]
        call_count[0] += 1
        return result

    with patch("automation.teleport_handler.time.monotonic", clock), \
         patch.object(h, "_find_destination", side_effect=mock_find):

        h.tick(_fake_frame, pico)   # IDLE -> WAIT_WINDOW
        h.tick(_fake_frame, pico)   # WAIT_WINDOW -> FIND_DESTINATION

        # 1차 OCR 실패 -> WAIT_RETRY
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.WAIT_RETRY
        assert h._attempt == 1

        # 0.3초만 경과 - 아직 WAIT_RETRY
        clock.advance(0.3)
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.WAIT_RETRY

        # 0.3초 더 경과 -> FIND_DESTINATION 재진입
        clock.advance(0.3)
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.FIND_DESTINATION

        # 2차 OCR 성공 -> WAIT_COMPLETE
        result = h.tick(_fake_frame, pico)
        assert result is None
        assert h._phase == _TPhase.WAIT_COMPLETE


def test_tick_max_retries_returns_false():
    """max_retries 소진 시 False 반환."""
    h = _make_handler(wait_after_key_ms=0, max_retries=2)
    pico = MagicMock()
    clock = FakeClock(start=1000.0)

    with patch("automation.teleport_handler.time.monotonic", clock), \
         patch.object(h, "_find_destination", return_value=None):

        h.tick(_fake_frame, pico)   # IDLE -> WAIT_WINDOW
        h.tick(_fake_frame, pico)   # WAIT_WINDOW -> FIND_DESTINATION

        # 1차 실패
        h.tick(_fake_frame, pico)   # -> WAIT_RETRY
        clock.advance(1.0)
        h.tick(_fake_frame, pico)   # -> FIND_DESTINATION

        # 2차 실패 -> FAILURE (attempt=2 >= max_retries=2)
        h.tick(_fake_frame, pico)   # -> FAILURE
        result = h.tick(_fake_frame, pico)
        assert result is False


def test_tick_does_not_block_caller():
    """tick() 호출이 실제로 수십 ms 이상 block하지 않는다."""
    h = _make_handler(wait_after_key_ms=800)
    pico = MagicMock()

    start = time.monotonic()
    # IDLE tick (키 입력만 하고 즉시 반환)
    with patch.object(pico, "key_tap_name"):
        h.tick(_fake_frame, pico)
    elapsed = time.monotonic() - start

    # 실제로 800ms 대기가 발생하면 안 됨 (10ms 미만이어야 정상)
    assert elapsed < 0.05, f"tick()이 {elapsed:.3f}초 동안 block됨!"

    # WAIT_WINDOW tick도 즉시 반환
    start = time.monotonic()
    h.tick(_fake_frame, pico)
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"WAIT_WINDOW tick이 {elapsed:.3f}초 block됨!"


# ────────────────────────────────────────────────────────────
# HuntingStateMachine 비블로킹 재시도 테스트
# ────────────────────────────────────────────────────────────

def test_state_machine_teleport_retry_nonblocking():
    """SM: 텔레포트 실패 후 _teleport_retry_at 설정 -> tick이 즉시 return."""
    from automation.state_machine import HuntingStateMachine, HuntingState

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
    import numpy as np
    frame = np.zeros((400, 600, 3), dtype="uint8")
    sm = HuntingStateMachine(config, pico, lambda: frame)

    # TeleportHandler.tick()이 False(실패)를 반환하도록 패치
    sm.scroll_teleporter.tick = MagicMock(return_value=False)
    sm.hp_reader.read = MagicMock(return_value=100.0)
    sm.hp_reader.get_cached = MagicMock(return_value=100.0)

    sm._enter(HuntingState.USE_SCROLL_DUMMY)

    clock = FakeClock(start=1000.0)
    tick_call_count = [0]
    original_tick = sm.scroll_teleporter.tick

    def counting_tick(*args, **kwargs):
        tick_call_count[0] += 1
        return original_tick(*args, **kwargs)

    sm.scroll_teleporter.tick = counting_tick

    with patch("automation.state_machine.time.monotonic", clock):
        # 1차 tick: 실패 -> retry_at = 1002.0 설정
        sm.update(frame, [])
        assert tick_call_count[0] == 1
        assert sm._teleport_retry_at > 0

        first_call_count = tick_call_count[0]

        # retry_at 이전 tick들: teleporter.tick 호출되면 안 됨
        for _ in range(5):
            clock.advance(0.1)
            sm.update(frame, [])

        # retry_at 이전에는 추가 tick 없음
        assert tick_call_count[0] == first_call_count

        # retry_at 넘기기
        clock.advance(2.0)
        sm.update(frame, [])

        # 이제 tick 재호출됨
        assert tick_call_count[0] == first_call_count + 1


def test_state_machine_speed_potion_nonblocking():
    """SM: USE_SPEED_POTION 상태에서 blocking sleep 없이 tick마다 return."""
    from automation.state_machine import HuntingStateMachine, HuntingState

    config = {
        "keys": {"speed_potion": "F9"},
        "hp_bar": {"region": {"x": 0, "y": 0, "width": 10, "height": 5}},
        "level_ocr": {"region": {"x": 0, "y": 0, "width": 50, "height": 20}},
        "scroll_dummy": {},
        "dummy": {},
        "hunt_waypoints": {"points": [{"x": 100, "y": 100}]},
        "loot": {},
    }
    pico = MagicMock()
    import numpy as np
    frame = np.zeros((400, 600, 3), dtype="uint8")
    sm = HuntingStateMachine(config, pico, lambda: frame)
    sm.hp_reader.read = MagicMock(return_value=100.0)
    sm.hp_reader.get_cached = MagicMock(return_value=100.0)

    sm._enter(HuntingState.USE_SPEED_POTION)

    clock = FakeClock(start=1000.0)

    with patch("automation.state_machine.time.monotonic", clock):
        # 1차 tick: 키 입력 + sent_at 설정 -> MOVE_TO_HUNT_ZONE 진입 안 함
        start_wall = time.monotonic()
        sm.update(frame, [])
        elapsed = time.monotonic() - start_wall
        assert elapsed < 0.05, f"첫 tick이 {elapsed:.3f}초 block됨"
        assert sm.state == HuntingState.USE_SPEED_POTION
        pico.key_tap_name.assert_called_with("F9", hold_ms=80)

        # 대기 중 tick들: 상태 유지
        for _ in range(3):
            clock.advance(0.2)
            sm.update(frame, [])
        assert sm.state == HuntingState.USE_SPEED_POTION

        # 1.1초 경과 -> MOVE_TO_HUNT_ZONE 전환
        clock.advance(1.1)
        sm.update(frame, [])
        assert sm.state == HuntingState.MOVE_TO_HUNT_ZONE


# ────────────────────────────────────────────────────────────
# STEP4 회귀 테스트
# ────────────────────────────────────────────────────────────

def test_step4_still_passes():
    """STEP4: active=False -> click 차단 회귀 테스트."""
    from tracking.tracker import SequentialTargetStateMachine, TargetState
    from tracking.enemy import Enemy

    click_cb = MagicMock()
    sm = SequentialTargetStateMachine(
        pico_click_callback=click_cb,
        to_screen_fn=lambda x, y: (x, y),
        lock_confirm_frames=1,
    )
    sm.set_active(False)

    enemy = Enemy(id=1, x=90, y=90, width=20, height=20,
                  center_x=100, center_y=100, confidence=1.0, predicted=False)
    for _ in range(5):
        sm.update({1: enemy})

    click_cb.assert_not_called()
    assert sm.state == TargetState.IDLE


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
