"""NullPicoWorker — Pico 명령을 실제로 실행하지 않는 더미 워커.

Dry-Run / Diagnostic 모드에서 PicoSerialWorker 대신 이 객체를 주입한다.
모든 메서드는 호출 사실을 로그에 기록하고 즉시 반환한다.
실제 COM 포트를 열지 않으므로 게임 외부에서 안전하게 사용할 수 있다.

인터페이스는 PicoSerialWorker와 동일하게 유지한다.
(duck-typing — isinstance 체크 없이 교체 가능)
"""

import logging
from typing import Optional

logger = logging.getLogger("null_pico")


class NullPicoWorker:
    """PicoSerialWorker 인터페이스를 구현하지만 실제 명령을 전송하지 않는다.

    Dry-Run 모드에서 기존 코드(main.py, HuntingStateMachine 등)를
    수정 없이 재사용하기 위해 완전한 duck-typing 인터페이스를 제공한다.
    """

    # ── 접속 상태 (항상 True 처럼 보이게 — SM 로직이 연결 체크를 우회하도록) ──
    @property
    def is_connected(self) -> bool:
        return True

    # ── 생명 주기 ────────────────────────────────────────────────────────────
    def start(self) -> None:
        logger.info("[NullPico] start() — no-op (dry-run)")

    def stop(self) -> None:
        logger.info("[NullPico] stop() — no-op (dry-run)")

    def stop_target(self) -> None:
        logger.info("[NullPico] stop_target() — no-op (dry-run)")

    # ── 마우스 명령 (차단됨) ─────────────────────────────────────────────────
    def click(
        self,
        x: int,
        y: int,
        pulse_ms: int = 20,
    ) -> None:
        logger.debug(f"[NullPico] BLOCKED click({x}, {y}, pulse_ms={pulse_ms})")

    def drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        steps: int = 8,
        step_delay_ms: int = 10,
    ) -> None:
        logger.debug(
            f"[NullPico] BLOCKED drag("
            f"{from_x},{from_y} → {to_x},{to_y}, steps={steps})"
        )

    # ── 키보드 명령 (차단됨) ─────────────────────────────────────────────────
    def key_tap_name(
        self,
        name: str,
        hold_ms: int = 50,
    ) -> None:
        logger.debug(f"[NullPico] BLOCKED key_tap_name({name!r}, hold_ms={hold_ms})")

    def key_tap(
        self,
        keycode: int,
        hold_ms: int = 50,
    ) -> None:
        logger.debug(f"[NullPico] BLOCKED key_tap(keycode={keycode}, hold_ms={hold_ms})")

    def key_down(self, name: str) -> None:
        logger.debug(f"[NullPico] BLOCKED key_down({name!r})")

    def key_up(self, name: str) -> None:
        logger.debug(f"[NullPico] BLOCKED key_up({name!r})")

    # ── 기타 헬퍼 (PicoSerialWorker 호환) ───────────────────────────────────
    def move_to(self, x: int, y: int) -> None:
        logger.debug(f"[NullPico] BLOCKED move_to({x}, {y})")

    def scroll(self, delta: int) -> None:
        logger.debug(f"[NullPico] BLOCKED scroll({delta})")
