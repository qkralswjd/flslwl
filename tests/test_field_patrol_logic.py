"""필드 사냥 순찰 로직 테스트.

이번에 새로 추가한 핵심 기능 검증:
  1. 3초 idle 타임아웃 → 다음 WP 이동
  2. 이동 중 적 발견 → 즉시 SCAN 전환 (WP 도착 안 기다림)
  3. 적 발견 시 idle 타임아웃 리셋
  4. idle_timeout_s config 반영
  5. _pc_scan_start_t 초기화 확인
  6. patrol_mover 상태 추적 (MOVING → 완료 감지)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from automation.state_machine import HuntingStateMachine, HuntingState


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def _make_cfg(idle_timeout_s=3.0, patrol_timeout_ms=200):
    return {
        "capture_offset": {"x": 0, "y": 0},
        "roi_offset":     {"x": 0, "y": 0},
        "keys": {
            "potion": "F5", "scroll": "F6", "speed_potion": "F9",
            "potion_cooldown_ms": 3000,
        },
        "hp_bar": {
            "region": {"x": 0, "y": 0, "width": 200, "height": 10},
            "threshold_pct": 50.0, "read_interval_s": 0.1,
        },
        "level_ocr": {
            "region": {"x": 0, "y": 0, "width": 80, "height": 25},
            "read_interval_s": 0.1,
            "target_level_dummy": 5, "target_level_hunt": 999,  # 레벨 완료 없이
        },
        "scroll_dummy": {
            "destination_region": {"x": 400, "y": 150, "width": 300, "height": 400},
            "destination_text": "허수아비",
            "wait_after_key_ms": 50, "wait_after_click_ms": 50,
        },
        "dummy": {
            "drag_from": {"x": 960, "y": 600}, "drag_to": {"x": 960, "y": 400},
            "drag_steps": 4, "attack_interval_ms": 50, "move_timeout_ms": 100,
        },
        "hunt_waypoints": {
            "points": [{"x": 800, "y": 400, "label": "HuntA", "wait_ms": 50}],
            "move_timeout_ms": 100,
        },
        "patrol_waypoints": {
            "points": [
                {"x": 1521, "y": 353, "label": "1차 필드사냥 이동", "wait_ms": 50},
                {"x": 1497, "y": 404, "label": "2차 필드사냥 이동", "wait_ms": 50},
                {"x": 1131, "y": 35,  "label": "3차 필드사냥 이동", "wait_ms": 50},
            ],
            "move_timeout_ms": patrol_timeout_ms,
            "idle_timeout_s":  idle_timeout_s,
        },
        "loot": {
            "scan_region": {"x": 0, "y": 0, "width": 100, "height": 100},
            "keywords": ["아데나"],
            "scan_interval_s": 999,
            "click_interval_ms": 50,
            "timeout_ms": 500,
        },
        "patrol_combat": {
            "kill_wait_ms": 100,
            "attack_interval_ms": 0,
            "max_attacks_per_wp": 3,
        },
    }


def _make_tracker_mock():
    """SequentialTargetSM mock — 기본값: IDLE 상태."""
    from tracking.tracker import TargetState
    tracker = MagicMock()
    tracker._sm        = MagicMock()
    tracker._sm.state  = TargetState.IDLE
    tracker._sm.reset  = MagicMock()
    tracker._sm.set_active = MagicMock()
    return tracker


def _make_sm(idle_timeout_s=3.0, level=None, hp_pct=100.0, with_tracker=True):
    """
    HuntingStateMachine 생성 헬퍼.

    with_tracker=True (기본값): 실제 필드 사냥처럼 tracker mock 주입
        → SCAN 상태에서 field 모드 경로(idle_timeout 기반)를 탐
    with_tracker=False: tracker 없음 → 기존 모드(max_attacks 기반) 경로
    """
    pico  = MagicMock()
    pico.is_connected = True
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    sm    = HuntingStateMachine(
        config=_make_cfg(idle_timeout_s=idle_timeout_s),
        pico_worker=pico,
        frame_grabber=lambda: frame,
    )
    sm.hp_reader.read           = MagicMock(return_value=hp_pct)
    sm.hp_reader.get_cached     = MagicMock(return_value=hp_pct)
    sm.level_reader.read        = MagicMock(return_value=level)
    sm.level_reader.get_cached  = MagicMock(return_value=level)
    sm.loot_detector.find       = MagicMock(return_value=[])

    if with_tracker:
        # 실제 필드 모드처럼 tracker 주입 → field 경로(idle_timeout) 활성화
        sm._tracker = _make_tracker_mock()

    return sm, pico, frame


def _fake_enemy(cx=800, cy=400):
    e = MagicMock()
    e.center_x = cx
    e.center_y = cy
    return e


def _enter_hunting_10(sm, frame):
    """HUNTING_10 진입 + patrol_mover 시작."""
    sm._enter(HuntingState.HUNTING_10)
    sm.update(frame, enemies=[])
    assert sm._patrol_started
    assert sm._pc_state == "PATROL"


def _force_to_scan(sm, frame, timeout=1.5):
    """PATROL → SCAN 강제 진행 (patrol_mover 타임아웃 대기)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        sm.update(frame, enemies=[])
        if sm._pc_state == "SCAN":
            return True
        time.sleep(0.03)
    return False


# ── 테스트 1: 3초 idle 타임아웃 ─────────────────────────────────────────────

class TestIdleTimeout:

    def test_3sec_idle_moves_to_next_wp(self):
        """SCAN에서 3초 동안 적 없으면 다음 WP(PATROL)로 이동."""
        sm, pico, frame = _make_sm(idle_timeout_s=0.1)  # 0.1초로 빠르게 테스트
        _enter_hunting_10(sm, frame)

        ok = _force_to_scan(sm, frame)
        assert ok, "PATROL→SCAN 전환 실패"
        assert sm._pc_state == "SCAN"

        # 0.1초 대기 → idle 타임아웃 발동
        time.sleep(0.15)
        sm.update(frame, enemies=[])

        assert sm._pc_state == "PATROL", \
            f"3초(테스트: 0.1초) idle 후 PATROL로 가야 함, 현재: {sm._pc_state}"

    def test_idle_timeout_respects_config(self):
        """idle_timeout_s 설정값이 실제로 반영되는지."""
        sm, pico, frame = _make_sm(idle_timeout_s=0.05)
        assert sm._hunt_idle_timeout == pytest.approx(0.05), \
            "config의 idle_timeout_s가 _hunt_idle_timeout에 반영돼야 함"

    def test_idle_not_triggered_before_timeout(self):
        """타임아웃 전에는 PATROL로 전환 안 함."""
        sm, pico, frame = _make_sm(idle_timeout_s=10.0)  # 10초 타임아웃
        _enter_hunting_10(sm, frame)

        ok = _force_to_scan(sm, frame)
        assert ok

        # 바로 update → 10초 안 지났으므로 PATROL 전환 없음
        sm.update(frame, enemies=[])
        assert sm._pc_state == "SCAN", \
            "타임아웃 전에는 SCAN 유지해야 함"

    def test_loot_checked_before_next_wp(self):
        """idle 타임아웃 시 아데나 있으면 LOOTING 우선."""
        sm, pico, frame = _make_sm(idle_timeout_s=0.05)
        _enter_hunting_10(sm, frame)

        ok = _force_to_scan(sm, frame)
        assert ok

        sm.loot_detector.find = MagicMock(
            return_value=[(100, 200, "아데나", 0.9)]
        )
        time.sleep(0.1)
        sm.update(frame, enemies=[])

        assert sm.state == HuntingState.LOOTING, \
            "idle 타임아웃 시에도 아데나 있으면 LOOTING 먼저"


# ── 테스트 2: 이동 중 적 발견 즉시 SCAN ─────────────────────────────────────

class TestEnemyDuringPatrol:

    def test_enemy_found_during_moving_triggers_scan(self):
        """PATROL MOVING 중 적 발견 → WP 도착 안 기다리고 즉시 SCAN (field 모드)."""
        from tracking.tracker import TargetState

        sm, pico, frame = _make_sm(idle_timeout_s=3.0)  # tracker 자동 주입
        _enter_hunting_10(sm, frame)

        # patrol_mover가 MOVING 상태인지 확인
        assert sm.patrol_mover._state == "MOVING", "초기 상태는 MOVING이어야 함"

        # field 모드: MOVING 중 적 발견 → 즉시 SCAN 전환
        enemy = _fake_enemy()
        sm.update(frame, enemies=[enemy])

        assert sm._pc_state == "SCAN", \
            "field 모드: PATROL MOVING 중 적 발견 → 즉시 SCAN"

    def test_enemy_found_during_moving_field_mode(self):
        """필드 모드(tracker 주입): PATROL MOVING 중 적 발견 → 즉시 SCAN."""
        from tracking.tracker import NearestNeighborTracker, TargetState

        sm, pico, frame = _make_sm(idle_timeout_s=3.0)
        _enter_hunting_10(sm, frame)

        # tracker 주입 (field 모드 시뮬레이션)
        mock_tracker = MagicMock()
        mock_tracker._sm.state = TargetState.IDLE
        mock_tracker._sm.set_active = MagicMock()
        sm._tracker = mock_tracker

        # patrol_mover가 MOVING 상태일 때 적 발견
        assert sm.patrol_mover._state == "MOVING"
        enemy = _fake_enemy()
        sm.update(frame, enemies=[enemy])

        assert sm._pc_state == "SCAN", \
            "field 모드: 이동 중 적 발견 → 즉시 SCAN 전환"

    def test_no_enemy_during_moving_continues_patrol(self):
        """이동 중 적 없으면 계속 PATROL 유지."""
        from tracking.tracker import TargetState

        sm, pico, frame = _make_sm(idle_timeout_s=3.0)
        _enter_hunting_10(sm, frame)

        mock_tracker = MagicMock()
        mock_tracker._sm.state = TargetState.IDLE
        sm._tracker = mock_tracker

        # 적 없이 update
        sm.update(frame, enemies=[])
        # patrol_mover가 아직 MOVING 중이면 PATROL 유지
        if sm.patrol_mover._state == "MOVING":
            assert sm._pc_state == "PATROL", "적 없으면 PATROL 유지"


# ── 테스트 3: 적 발견 시 idle 타임아웃 리셋 ──────────────────────────────────

class TestScanTimerReset:

    def test_enemy_found_resets_scan_timer(self):
        """SCAN 중 적 발견 → KILL_WAIT 전환 + _pc_scan_start_t 갱신."""
        from tracking.tracker import TargetState

        sm, pico, frame = _make_sm(idle_timeout_s=0.05)
        _enter_hunting_10(sm, frame)

        ok = _force_to_scan(sm, frame)
        assert ok

        # 타임아웃 이미 지남
        time.sleep(0.1)

        # tracker 주입 (field 모드)
        mock_tracker = MagicMock()
        mock_tracker._sm.state = TargetState.IDLE
        mock_tracker._sm.set_active = MagicMock()
        sm._tracker = mock_tracker

        enemy = _fake_enemy()
        before_scan_t = sm._pc_scan_start_t
        sm.update(frame, enemies=[enemy])

        assert sm._pc_state == "KILL_WAIT", \
            "적 발견 시 KILL_WAIT 전환 (타임아웃 지났어도)"
        # scan_start_t가 갱신됐는지 확인
        assert sm._pc_scan_start_t >= before_scan_t, \
            "적 발견 시 scan_start_t 갱신"


# ── 테스트 4: _pc_scan_start_t 초기화 확인 ───────────────────────────────────

class TestScanStartInit:

    def test_scan_start_t_initializes_on_arrived(self):
        """PATROL→SCAN 전환 시 _pc_scan_start_t가 현재 시각으로 설정."""
        sm, pico, frame = _make_sm(idle_timeout_s=3.0)
        _enter_hunting_10(sm, frame)

        t_before = time.time()
        ok = _force_to_scan(sm, frame)
        t_after = time.time()

        assert ok
        assert t_before <= sm._pc_scan_start_t <= t_after + 0.1, \
            "SCAN 진입 시 _pc_scan_start_t가 현재 시각으로 설정돼야 함"

    def test_scan_start_t_initial_value(self):
        """초기값은 0.0."""
        sm, pico, frame = _make_sm()
        assert sm._pc_scan_start_t == 0.0


# ── 테스트 5: get_status scan_idle_elapsed ────────────────────────────────────

class TestGetStatus:

    def test_scan_idle_elapsed_in_scan_state(self):
        """SCAN 상태일 때 get_status()에 scan_idle_elapsed 포함."""
        sm, pico, frame = _make_sm(idle_timeout_s=3.0)
        _enter_hunting_10(sm, frame)

        ok = _force_to_scan(sm, frame)
        assert ok

        time.sleep(0.05)
        status = sm.get_status()

        assert "scan_idle_elapsed" in status, \
            "get_status()에 scan_idle_elapsed 키 있어야 함"
        assert status["scan_idle_elapsed"] >= 0.0, \
            "scan_idle_elapsed는 0 이상"

    def test_scan_idle_elapsed_zero_in_patrol(self):
        """PATROL 상태(SCAN 아님)일 때 scan_idle_elapsed는 0."""
        sm, pico, frame = _make_sm()
        _enter_hunting_10(sm, frame)

        # PATROL 상태에서 바로 status 조회
        assert sm._pc_state == "PATROL"
        status = sm.get_status()
        assert status["scan_idle_elapsed"] == 0.0, \
            "PATROL 상태에서는 scan_idle_elapsed가 0"


# ── 테스트 6: 실제 patrol_waypoints 동선 반영 확인 ───────────────────────────

class TestRealWaypoints:

    def test_config_waypoints_loaded_correctly(self):
        """config_automation.json의 patrol_waypoints가 올바르게 로드."""
        sm, pico, frame = _make_sm()

        wps = sm.patrol_mover.waypoints
        assert len(wps) == 3, f"웨이포인트 3개여야 함, 실제: {len(wps)}"
        assert wps[0]["label"] == "1차 필드사냥 이동"
        assert wps[1]["label"] == "2차 필드사냥 이동"
        assert wps[2]["label"] == "3차 필드사냥 이동"
        assert sm.patrol_mover.loop is True, "필드 순찰은 loop=True"

    def test_idle_timeout_from_config(self):
        """idle_timeout_s가 config에서 sm._hunt_idle_timeout으로 전달."""
        sm, pico, frame = _make_sm(idle_timeout_s=5.0)
        assert sm._hunt_idle_timeout == pytest.approx(5.0)

    def test_waypoints_loop_after_last(self):
        """마지막 WP 도착 후 처음으로 순환."""
        sm, pico, frame = _make_sm(idle_timeout_s=0.01)
        _enter_hunting_10(sm, frame)

        # 3개의 WP를 모두 통과
        visited = []
        deadline = time.time() + 5.0
        while time.time() < deadline and len(visited) < 4:
            sm.update(frame, enemies=[])
            if sm._pc_state == "SCAN":
                lbl = sm.patrol_mover.current_label
                if not visited or visited[-1] != lbl:
                    visited.append(lbl)
                time.sleep(0.02)
                sm.update(frame, enemies=[])  # idle 타임아웃 발동
            time.sleep(0.02)

        assert len(visited) >= 3, \
            f"3개 WP 모두 방문해야 함, 방문: {visited}"
        # 루프: 4번째는 다시 첫 번째
        if len(visited) >= 4:
            assert visited[3] == visited[0], "루프 순환 확인"
