"""core/state.py — 상태머신 베이스 클래스.

이 모듈은 HuntingStateMachine(modes/) 과 그 외 상위 레벨 상태머신이
공통으로 사용하는 BaseFSM 을 제공한다.

설계 원칙:
  - 의존성 없음 (순수 Python)
  - 상태 전환 로깅은 BaseFSM 이 자동 처리
  - tick() 추상 메서드로 매 루프 갱신
  - on_enter_* / on_exit_* 훅 오버라이드 가능
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("core.state")


class BaseFSM(ABC):
    """유한 상태머신 베이스 클래스.

    하위 클래스 최소 구현:
        1. 상태 Enum 정의 (권장: inner class 또는 모듈 레벨)
        2. tick(dt) 구현 — 매 루프 호출
        3. _initial_state() 반환값 지정

    선택적 훅:
        on_enter_<STATE_NAME>(self) — 해당 상태 진입 시 1회 호출
        on_exit_<STATE_NAME>(self)  — 해당 상태 탈출 시 1회 호출

    사용 예::

        class MyFSM(BaseFSM):
            class S(Enum):
                IDLE    = auto()
                RUNNING = auto()

            def _initial_state(self):
                return self.S.IDLE

            def tick(self, dt: float):
                if self.state == self.S.IDLE:
                    ...

            def on_enter_RUNNING(self):
                print("running started")
    """

    def __init__(self) -> None:
        self._state: Any              = self._initial_state()
        self._prev_state: Any         = None
        self._state_entered_at: float = time.monotonic()
        self._tick_count: int         = 0

    # ── 추상 메서드 ────────────────────────────────────────────────────────

    @abstractmethod
    def _initial_state(self) -> Any:
        """초기 상태를 반환한다."""

    @abstractmethod
    def tick(self, dt: float) -> None:
        """매 루프마다 호출된다.

        Args:
            dt: 이전 tick 과의 시간 간격 (초)
        """

    # ── 상태 프로퍼티 ──────────────────────────────────────────────────────

    @property
    def state(self) -> Any:
        return self._state

    @property
    def prev_state(self) -> Any:
        return self._prev_state

    @property
    def state_elapsed_s(self) -> float:
        """현재 상태에 머문 시간(초)."""
        return time.monotonic() - self._state_entered_at

    @property
    def state_elapsed_ms(self) -> float:
        """현재 상태에 머문 시간(밀리초)."""
        return self.state_elapsed_s * 1000.0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # ── 상태 전환 ──────────────────────────────────────────────────────────

    def transition(self, new_state: Any, *, force: bool = False) -> bool:
        """상태를 전환한다.

        Args:
            new_state : 전환할 상태
            force     : True 이면 현재 상태와 동일해도 전환(재진입)

        Returns:
            실제로 전환이 일어났으면 True.
        """
        if not force and new_state == self._state:
            return False

        # on_exit 훅
        exit_hook = f"on_exit_{self._state.name}"
        if callable(getattr(self, exit_hook, None)):
            getattr(self, exit_hook)()

        logger.info(
            "[%s] %s → %s",
            type(self).__name__,
            self._state.name,
            new_state.name,
        )

        self._prev_state       = self._state
        self._state            = new_state
        self._state_entered_at = time.monotonic()

        # on_enter 훅
        enter_hook = f"on_enter_{new_state.name}"
        if callable(getattr(self, enter_hook, None)):
            getattr(self, enter_hook)()

        return True

    def run_tick(self, dt: float) -> None:
        """tick() 호출 + tick_count 증가. 주 루프에서 이 메서드를 호출한다."""
        self._tick_count += 1
        self.tick(dt)

    # ── 편의 메서드 ────────────────────────────────────────────────────────

    def is_state(self, *states: Any) -> bool:
        """현재 상태가 주어진 상태 중 하나인지 확인한다."""
        return self._state in states

    def elapsed_since_entry(self, threshold_ms: float) -> bool:
        """현재 상태 진입 후 threshold_ms 이상 경과했으면 True."""
        return self.state_elapsed_ms >= threshold_ms

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(state={self._state.name}, "
            f"elapsed={self.state_elapsed_ms:.0f}ms, ticks={self._tick_count})"
        )
