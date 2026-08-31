"""HuntingStateMachine — 허수아비 공격 흐름 (1~5레벨) 테스트.

start_at_dummy() 호출 후:
  - 즉시 ATTACKING_DUMMY 진입
  - drag() 호출 확인
  - Lv.5 달성 시 USE_SPEED_POTION 전환
  - HP < 50% 시 F5 물약 사용
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from automation.state_machine import HuntingStateMachine, HuntingState


# ── 공통 픽스처 ───────────────────────────────────────────────────────────────

def _make_cfg():
    return {
        "capture_offset": {"x": 0, "y": 0},
        "roi_offset":     {"x": 0, "y": 0},
        "keys": {
            "potion": "F5",
            "scroll": "F6",
            "speed_potion": "F9",
            "potion_cooldown_ms": 3000,
        },
        "hp_bar": {
            "region": {"x": 0, "y": 0, "width": 200, "height": 10},
            "threshold_pct": 50.0,
            "read_interval_s": 0.1,
        },
        "level_ocr": {
            "region": {"x": 0, "y": 0, "width": 80, "height": 25},
            "read_interval_s": 0.1,
            "target_level_dummy": 5,
            "target_level_hunt":  10,
        },
        "scroll_dummy": {
            "destination_region": {"x": 400, "y": 150, "width": 300, "height": 400},
            "destination_text": "허수아비",
            "wait_after_key_ms": 100,
            "wait_after_click_ms": 100,
        },
        "dummy": {
            "drag_from": {"x": 960, "y": 600},
            "drag_to":   {"x": 960, "y": 400},
            "drag_steps": 4,
            "attack_interval_ms": 50,   # 테스트용 — 빠른 간격
            "move_timeout_ms": 100,
        },
        "hunt_waypoints": {
            "points": [{"x": 800, "y": 400, "label": "A", "wait_ms": 100}],
            "move_timeout_ms": 1000,
        },
        "loot": {
            "scan_region": {"x": 0, "y": 0, "width": 100, "height": 100},
            "keywords": ["아데나"],
            "scan_interval_s": 999,
            "click_interval_ms": 100,
            "timeout_ms": 500,
        },
    }


def _make_sm(hp_pct=100.0, level=None):
    """HuntingStateMachine + mock HpReader/LevelReader/LootDetector."""
    pico = MagicMock()
    pico.is_connected = True
    frame_mock = np.zeros((100, 200, 3), dtype=np.uint8)
    sm = HuntingStateMachine(
        config=_make_cfg(),
        pico_worker=pico,
        frame_grabber=lambda: frame_mock,
    )
    # HpReader mock
    sm.hp_reader.read    = MagicMock(return_value=hp_pct)
    sm.hp_reader.get_cached = MagicMock(return_value=hp_pct)
    # LevelReader mock
    sm.level_reader.read     = MagicMock(return_value=level)
    sm.level_reader.get_cached = MagicMock(return_value=level)
    # LootDetector mock — 아데나 없음
    sm.loot_detector.find = MagicMock(return_value=[])
    return sm, pico, frame_mock


# ── 테스트 ───────────────────────────────────────────────────────────────────

class TestStartAtDummy:
    """start_at_dummy() 흐름 검증."""

    def test_start_at_dummy_enters_attacking_dummy(self):
        """start_at_dummy() 호출 시 즉시 ATTACKING_DUMMY 진입."""
        sm, pico, frame = _make_sm()
        sm.start_at_dummy()
        assert sm.state == HuntingState.ATTACKING_DUMMY

    def test_start_at_dummy_not_use_scroll(self):
        """start_at_dummy() 는 USE_SCROLL_DUMMY를 거치지 않는다."""
        sm, pico, frame = _make_sm()
        sm.start_at_dummy()
        # key_tap_name이 F6 텔레포트 키로 호출되지 않아야 함
        pico.key_tap_name.assert_not_called()

    def test_dummy_attack_calls_drag(self):
        """update() 호출 시 pico.drag()가 실행된다."""
        sm, pico, frame = _make_sm(hp_pct=100.0, level=1)
        sm.start_at_dummy()
        # 공격 간격(50ms) 이후 update 호출
        sm._last_dummy_atk = 0.0   # 강제 초기화로 즉시 실행
        sm.update(frame, [])
        pico.drag.assert_called_once_with(960, 600, 960, 400, 4)

    def test_dummy_drag_uses_configured_coords(self):
        """drag_from / drag_to 좌표가 config 설정과 일치한다."""
        sm, pico, frame = _make_sm(hp_pct=100.0, level=3)
        sm.start_at_dummy()
        sm._last_dummy_atk = 0.0
        sm.update(frame, [])
        args = pico.drag.call_args[0]
        assert args[0] == 960  # fx
        assert args[1] == 600  # fy
        assert args[2] == 960  # tx
        assert args[3] == 400  # ty

    def test_level_5_triggers_speed_potion(self):
        """레벨 5 달성 시 USE_SPEED_POTION 전환."""
        sm, pico, frame = _make_sm(hp_pct=100.0, level=5)
        sm.start_at_dummy()
        sm.update(frame, [])
        assert sm.state == HuntingState.USE_SPEED_POTION

    def test_level_below_target_stays_attacking(self):
        """레벨 4 이하 → ATTACKING_DUMMY 유지."""
        sm, pico, frame = _make_sm(hp_pct=100.0, level=4)
        sm.start_at_dummy()
        sm.update(frame, [])
        assert sm.state == HuntingState.ATTACKING_DUMMY

    def test_level_none_stays_attacking(self):
        """레벨 인식 실패(None) → ATTACKING_DUMMY 유지 (포기 안 함)."""
        sm, pico, frame = _make_sm(hp_pct=100.0, level=None)
        sm.start_at_dummy()
        sm.update(frame, [])
        assert sm.state == HuntingState.ATTACKING_DUMMY

    def test_hp_low_uses_potion(self):
        """HP < 50% → F5 물약 사용."""
        sm, pico, frame = _make_sm(hp_pct=49.0, level=3)
        sm._last_potion_t = 0.0   # 쿨타임 초기화
        sm.start_at_dummy()
        sm.update(frame, [])
        pico.key_tap_name.assert_any_call("F5", hold_ms=80)

    def test_hp_ok_no_potion(self):
        """HP >= 50% → 물약 사용 안 함."""
        sm, pico, frame = _make_sm(hp_pct=80.0, level=3)
        sm.start_at_dummy()
        sm._last_dummy_atk = time.time()  # 공격은 쿨타임 중
        sm.update(frame, [])
        # F5는 호출되지 않아야 함
        for call in pico.key_tap_name.call_args_list:
            assert call[0][0] != "F5", "HP 정상인데 물약 사용됨"

    def test_stop_returns_to_idle(self):
        """stop() 호출 시 IDLE로 복귀."""
        sm, pico, frame = _make_sm()
        sm.start_at_dummy()
        sm.stop()
        assert sm.state == HuntingState.IDLE


class TestStartAtHuntZone:
    """start_at_hunt_zone() 흐름 검증."""

    def test_enters_move_to_hunt_zone(self):
        """start_at_hunt_zone() 호출 시 MOVE_TO_HUNT_ZONE 진입."""
        sm, pico, frame = _make_sm()
        sm.start_at_hunt_zone()
        assert sm.state == HuntingState.MOVE_TO_HUNT_ZONE
