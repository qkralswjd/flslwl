"""HUNTING_10 PATROL_COMBAT 서브플로우 테스트.

검증 항목:
  1. PATROL → ARRIVED → SCAN 전환
  2. SCAN + 적 있음 → 클릭+드래그 → KILL_WAIT 전환
  3. KILL_WAIT → kill_wait_ms 경과 → LOOT_SCAN 전환
  4. LOOT_SCAN + 아데나 없음 → SCAN 복귀
  5. LOOT_SCAN + 아데나 있음 → LOOTING → 복귀 시 SCAN 유지
  6. SCAN + 적 없음 + 아데나 없음 → PATROL (다음 WP)
  7. max_attacks_per_wp 초과 → PATROL (다음 WP)
  8. attack_interval_ms 쿨타임 중에는 공격 안 함
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from automation.state_machine import HuntingStateMachine, HuntingState


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def _make_cfg():
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
            "target_level_dummy": 5, "target_level_hunt": 10,
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
            "points": [{"x": 800, "y": 400, "label": "A", "wait_ms": 50}],
            "move_timeout_ms": 200,
        },
        "patrol_waypoints": {
            "points": [
                {"x": 900, "y": 300, "label": "P1", "wait_ms": 50},
                {"x": 1000, "y": 350, "label": "P2", "wait_ms": 50},
            ],
            "move_timeout_ms": 200,   # 테스트용 짧은 타임아웃
        },
        "loot": {
            "scan_region": {"x": 0, "y": 0, "width": 100, "height": 100},
            "keywords": ["아데나"],
            "scan_interval_s": 999,   # 캐시 고정 (테스트에서 직접 주입)
            "click_interval_ms": 50,
            "timeout_ms": 500,
        },
        "patrol_combat": {
            "kill_wait_ms": 100,        # 테스트용 짧은 대기
            "attack_interval_ms": 0,    # 쿨타임 없음 (기본 테스트)
            "max_attacks_per_wp": 3,
        },
    }


def _make_sm(level=None, hp_pct=100.0):
    pico  = MagicMock()
    pico.is_connected = True
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    sm    = HuntingStateMachine(
        config=_make_cfg(),
        pico_worker=pico,
        frame_grabber=lambda: frame,
    )
    sm.hp_reader.read       = MagicMock(return_value=hp_pct)
    sm.hp_reader.get_cached = MagicMock(return_value=hp_pct)
    sm.level_reader.read        = MagicMock(return_value=level)
    sm.level_reader.get_cached  = MagicMock(return_value=level)
    sm.loot_detector.find       = MagicMock(return_value=[])   # 기본: 아데나 없음
    return sm, pico, frame


def _fake_enemy(cx=800, cy=400):
    e = MagicMock()
    e.center_x = cx
    e.center_y = cy
    return e


def _enter_hunting_10(sm, pico, frame):
    """HUNTING_10 상태로 직접 진입 + patrol_mover 시작."""
    sm._enter(HuntingState.HUNTING_10)
    sm.update(frame, enemies=[])   # 순찰 시작 트리거
    assert sm._patrol_started
    assert sm._pc_state == "PATROL"


# ── 테스트 ──────────────────────────────────────────────────────────────────

class TestPatrolToCombat:
    """PATROL → SCAN 전환."""

    def test_patrol_arrived_transitions_to_scan(self):
        """patrol_mover 타임아웃 경과 후 _pc_state가 SCAN으로 전환."""
        sm, pico, frame = _make_sm()
        _enter_hunting_10(sm, pico, frame)

        # patrol_mover 타임아웃까지 tick 반복
        deadline = time.time() + 1.0
        while time.time() < deadline:
            sm.update(frame, enemies=[])
            if sm._pc_state == "SCAN":
                break
            time.sleep(0.05)

        assert sm._pc_state == "SCAN", "ARRIVED 후 SCAN으로 전환돼야 함"
        assert sm.state == HuntingState.HUNTING_10

    def test_pc_attack_count_reset_on_new_wp(self):
        """새 WP 도착 시 _pc_attack_count가 0으로 리셋."""
        sm, pico, frame = _make_sm()
        _enter_hunting_10(sm, pico, frame)
        sm._pc_attack_count = 2   # 이전 WP에서 공격했다고 가정

        deadline = time.time() + 1.0
        while time.time() < deadline:
            sm.update(frame, enemies=[])
            if sm._pc_state == "SCAN":
                break
            time.sleep(0.05)

        assert sm._pc_attack_count == 0, "새 WP 도착 시 attack_count 리셋"


class TestScanWithEnemy:
    """SCAN + 적 발견 → 공격 → KILL_WAIT."""

    def _force_scan(self, sm, pico, frame):
        _enter_hunting_10(sm, pico, frame)
        # PATROL 타임아웃 강제
        deadline = time.time() + 1.0
        while time.time() < deadline:
            sm.update(frame, enemies=[])
            if sm._pc_state == "SCAN":
                break
            time.sleep(0.05)
        assert sm._pc_state == "SCAN"

    def test_attack_on_enemy_found(self):
        """적 발견 시 click + drag 호출, KILL_WAIT 전환."""
        sm, pico, frame = _make_sm()
        self._force_scan(sm, pico, frame)

        enemy = _fake_enemy(800, 400)
        sm.update(frame, enemies=[enemy])

        assert pico.click.called, "click() 호출돼야 함"
        assert pico.drag.called,  "drag() 호출돼야 함"
        assert sm._pc_state == "KILL_WAIT"
        assert sm._pc_attack_count == 1

    def test_no_attack_when_empty(self):
        """적 없고 아데나 없으면 PATROL로 이동 (다음 WP)."""
        sm, pico, frame = _make_sm()
        self._force_scan(sm, pico, frame)
        sm.loot_detector.find = MagicMock(return_value=[])

        sm.update(frame, enemies=[])
        assert sm._pc_state == "PATROL", "적+아데나 없으면 다음 WP"

    def test_loot_trigger_in_scan(self):
        """적 없지만 아데나 있으면 LOOTING 전환."""
        sm, pico, frame = _make_sm()
        self._force_scan(sm, pico, frame)
        sm.loot_detector.find = MagicMock(
            return_value=[(100, 200, "아데나", 0.9)]
        )

        sm.update(frame, enemies=[])
        assert sm.state == HuntingState.LOOTING


class TestKillWait:
    """KILL_WAIT → kill_wait_ms 경과 → LOOT_SCAN."""

    def test_transitions_to_loot_scan_after_wait(self):
        sm, pico, frame = _make_sm()
        sm._enter(HuntingState.HUNTING_10)
        sm.update(frame, enemies=[])

        # 강제로 KILL_WAIT 상태 진입
        sm._pc_state        = "KILL_WAIT"
        sm._pc_kill_start_t = time.time() - 999   # 이미 경과

        sm.update(frame, enemies=[])
        assert sm._pc_state == "LOOT_SCAN"

    def test_stays_in_kill_wait_before_timeout(self):
        sm, pico, frame = _make_sm()
        sm._enter(HuntingState.HUNTING_10)
        sm.update(frame, enemies=[])

        sm._pc_state        = "KILL_WAIT"
        sm._pc_kill_start_t = time.time() + 999   # 아직 안 경과

        sm.update(frame, enemies=[])
        assert sm._pc_state == "KILL_WAIT", "타임아웃 전엔 KILL_WAIT 유지"


class TestLootScan:
    """LOOT_SCAN → 아데나 있으면 LOOTING, 없으면 SCAN."""

    def test_transitions_to_looting_when_loot_found(self):
        sm, pico, frame = _make_sm()
        sm._enter(HuntingState.HUNTING_10)
        sm.update(frame, enemies=[])
        sm._pc_state = "LOOT_SCAN"
        sm.loot_detector.find = MagicMock(
            return_value=[(300, 400, "아데나", 0.95)]
        )

        sm.update(frame, enemies=[])
        assert sm.state == HuntingState.LOOTING

    def test_transitions_to_scan_when_no_loot(self):
        sm, pico, frame = _make_sm()
        sm._enter(HuntingState.HUNTING_10)
        sm.update(frame, enemies=[])
        sm._pc_state = "LOOT_SCAN"
        sm.loot_detector.find = MagicMock(return_value=[])

        sm.update(frame, enemies=[])
        assert sm._pc_state == "SCAN", "아데나 없으면 SCAN으로 복귀"

    def test_looting_return_to_hunting_10_keeps_scan(self):
        """LOOTING 완료 후 HUNTING_10 복귀 시 _pc_state = SCAN 유지."""
        sm, pico, frame = _make_sm()
        sm._enter(HuntingState.HUNTING_10)
        sm.update(frame, enemies=[])

        # LOOTING 강제 진입
        sm._loot_targets      = [(100, 200, "아데나", 0.9)]
        sm._loot_idx          = 0
        sm._loot_start_t      = time.time()
        sm._loot_all_clicked_t = 0.0
        sm._loot_return_state = HuntingState.HUNTING_10
        sm._enter(HuntingState.LOOTING)
        sm._pc_state          = "LOOT_SCAN"   # 공격 후 아데나 발견 케이스

        # 루팅 완료 처리 — post_click_wait 스킵을 위해 클릭 완료 시각을 과거로 설정
        sm._loot_idx = 99   # 모두 처리된 것으로
        sm._loot_all_clicked_t = time.time() - 10.0  # 이미 대기 완료
        sm.update(frame, enemies=[])

        assert sm.state == HuntingState.HUNTING_10
        assert sm._pc_state == "SCAN", "LOOTING 복귀 시 SCAN 유지"


class TestMaxAttacks:
    """max_attacks_per_wp 초과 시 PATROL로 전환."""

    def test_max_attacks_forces_patrol(self):
        sm, pico, frame = _make_sm()
        sm._enter(HuntingState.HUNTING_10)
        sm.update(frame, enemies=[])
        sm._pc_state        = "SCAN"
        sm._pc_attack_count = sm._pc_max_attacks   # 이미 최대

        enemy = _fake_enemy()
        sm.update(frame, enemies=[enemy])

        assert sm._pc_state == "PATROL", "최대 공격 초과 시 PATROL"
        assert not pico.drag.called, "최대 공격 초과 시 추가 공격 없음"


class TestAttackCooltime:
    """attack_interval_ms 쿨타임 중 공격 안 함."""

    def test_no_attack_during_cooltime(self):
        cfg = _make_cfg()
        cfg["patrol_combat"]["attack_interval_ms"] = 60_000   # 60초 쿨타임
        pico  = MagicMock()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        sm    = HuntingStateMachine(cfg, pico, lambda: frame)
        sm.hp_reader.read       = MagicMock(return_value=100.0)
        sm.hp_reader.get_cached = MagicMock(return_value=100.0)
        sm.level_reader.read        = MagicMock(return_value=None)
        sm.level_reader.get_cached  = MagicMock(return_value=None)
        sm.loot_detector.find       = MagicMock(return_value=[])

        sm._enter(HuntingState.HUNTING_10)
        sm.update(frame, enemies=[])
        sm._pc_state        = "SCAN"
        sm._pc_last_atk_t   = time.time()   # 방금 공격

        pico.reset_mock()
        sm.update(frame, enemies=[_fake_enemy()])
        assert not pico.drag.called, "쿨타임 중에는 공격 없음"
