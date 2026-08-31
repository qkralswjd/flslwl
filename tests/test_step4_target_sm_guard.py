"""STEP4: SequentialTargetSM active 플래그 검증 테스트.

검증 항목:
  1. HUNTING_10 + 적 존재 → 공격 가능 (active=True)
  2. LOOTING + 기존 target 존재 → 공격 불가 (active=False)
  3. MOVE_TO_HUNT_ZONE + 적 존재 → 공격 불가
  4. ATTACKING_DUMMY → target SM 공격 불가
  5. LOOTING 진입 시 target_id 초기화 확인
  6. set_active(False) 후 update() 호출해도 pico 콜백 미호출
"""

import time
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
from typing import Optional, Dict
from unittest.mock import MagicMock, patch
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracking.tracker import SequentialTargetStateMachine, TargetState
from tracking.enemy import Enemy


# ── 헬퍼 ──────────────────────────────────────────────────────────────────

def _make_enemy(eid: int, cx: int = 100, cy: int = 100) -> Enemy:
    """테스트용 Enemy 생성."""
    return Enemy(
        id=eid, x=cx - 10, y=cy - 10,
        width=20, height=20,
        center_x=cx, center_y=cy,
        confidence=1.0,
        predicted=False,
    )


def _make_sm(click_cb=None, drag_cb=None) -> SequentialTargetStateMachine:
    """테스트용 SM 생성."""
    return SequentialTargetStateMachine(
        pico_click_callback=click_cb,
        pico_drag_callback=drag_cb,
        to_screen_fn=lambda x, y: (x, y),
        lock_confirm_frames=1,       # 1프레임에 바로 CLICKING
        wait_dead_timeout_ms=9999,
        next_target_cooldown_ms=0,
    )


def _advance_to_clicking(sm: SequentialTargetStateMachine, enemies: Dict[int, Enemy]) -> None:
    """IDLE → LOCKING → CLICKING 까지 진행 (lock_confirm_frames=1 가정)."""
    sm.update(enemies)   # IDLE → LOCKING (타겟 선택)
    sm.update(enemies)   # LOCKING → CLICKING (streak=1 >= lock_confirm_frames=1, 클릭 발사)


# ── 테스트 1: active=True → 공격 발생 ─────────────────────────────────────

def test_active_true_fires_click():
    """active=True(기본값)이면 정상적으로 click 콜백이 호출된다."""
    click_cb = MagicMock()
    sm = _make_sm(click_cb=click_cb)

    assert sm.active is True

    enemies = {1: _make_enemy(1)}
    _advance_to_clicking(sm, enemies)

    assert sm.state == TargetState.CLICKING
    click_cb.assert_called_once()


# ── 테스트 2: LOOTING → set_active(False) → 공격 불가 ────────────────────

def test_looting_no_click():
    """set_active(False) 이후 update()는 클릭 콜백을 호출하지 않는다."""
    click_cb = MagicMock()
    sm = _make_sm(click_cb=click_cb)

    enemies = {1: _make_enemy(1)}

    # LOOTING 진입 시뮬레이션: set_active(False)
    sm.set_active(False)

    # update()를 여러 번 호출해도 클릭 없음
    for _ in range(5):
        sm.update(enemies)

    click_cb.assert_not_called()
    assert sm.state == TargetState.IDLE


# ── 테스트 3: LOOTING 진입 시 target_id 초기화 ───────────────────────────

def test_looting_resets_target_id():
    """set_active(False)는 기존 target_id를 None으로 초기화한다."""
    click_cb = MagicMock()
    sm = _make_sm(click_cb=click_cb)

    enemies = {1: _make_enemy(1)}

    # LOCKING 상태 + target_id 설정
    sm.update(enemies)   # → LOCKING, target_id=1
    assert sm.target_id == 1

    # LOOTING 진입 → set_active(False)
    sm.set_active(False)

    assert sm.target_id is None
    assert sm.state == TargetState.IDLE


# ── 테스트 4: MOVE_TO_HUNT_ZONE → 공격 불가 ─────────────────────────────

def test_move_to_hunt_zone_no_click():
    """active=False (MOVE_TO_HUNT_ZONE 등) 상태에서 적이 있어도 클릭 없음."""
    click_cb = MagicMock()
    sm = _make_sm(click_cb=click_cb)

    sm.set_active(False)

    enemies = {1: _make_enemy(1), 2: _make_enemy(2, 200, 200)}
    for _ in range(10):
        sm.update(enemies)

    click_cb.assert_not_called()


# ── 테스트 5: ATTACKING_DUMMY → 공격 불가 ────────────────────────────────

def test_attacking_dummy_no_target_sm_click():
    """active=False (ATTACKING_DUMMY) 상태에서 target SM 클릭 없음."""
    click_cb = MagicMock()
    sm = _make_sm(click_cb=click_cb)

    sm.set_active(False)

    enemies = {1: _make_enemy(1)}
    sm.update(enemies)
    sm.update(enemies)

    click_cb.assert_not_called()
    assert sm.state == TargetState.IDLE


# ── 테스트 6: active 재활성화 후 정상 공격 ───────────────────────────────

def test_reactivate_after_looting():
    """LOOTING 종료 → HUNTING_10 복귀 → set_active(True) → 다시 공격 가능."""
    click_cb = MagicMock()
    sm = _make_sm(click_cb=click_cb)

    enemies = {1: _make_enemy(1)}

    # LOOTING 진입
    sm.set_active(False)
    sm.update(enemies)
    click_cb.assert_not_called()

    # HUNTING_10 복귀
    sm.set_active(True)
    _advance_to_clicking(sm, enemies)

    assert sm.state == TargetState.CLICKING
    click_cb.assert_called_once()


# ── 테스트 7: CLICKING 도중 set_active(False) → 즉시 reset ───────────────

def test_set_active_false_during_clicking_resets():
    """CLICKING 상태 중 set_active(False) → state=IDLE, target_id=None."""
    click_cb = MagicMock()
    sm = _make_sm(click_cb=click_cb)

    enemies = {1: _make_enemy(1)}
    _advance_to_clicking(sm, enemies)  # → CLICKING

    assert sm.state == TargetState.CLICKING

    # LOOTING 진입
    sm.set_active(False)

    assert sm.state == TargetState.IDLE
    assert sm.target_id is None

    # 이후 update() 해도 클릭 없음
    click_cb.reset_mock()
    sm.update(enemies)
    click_cb.assert_not_called()


# ── 테스트 8: drag 콜백도 차단 ───────────────────────────────────────────

def test_drag_also_blocked_when_inactive():
    """drag 모드에서도 active=False 이면 drag 콜백 미호출."""
    drag_cb = MagicMock()
    sm = SequentialTargetStateMachine(
        pico_click_callback=None,
        pico_drag_callback=drag_cb,
        to_screen_fn=lambda x, y: (x, y),
        lock_confirm_frames=1,
        drag_enabled=True,
        drag_dx=50,
        drag_dy=0,
    )
    sm.set_active(False)

    enemies = {1: _make_enemy(1)}
    for _ in range(5):
        sm.update(enemies)

    drag_cb.assert_not_called()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
