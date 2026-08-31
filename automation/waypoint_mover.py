"""좌표 기반 웨이포인트 이동 모듈.

미리 지정한 좌표 리스트를 순서대로 Pico 클릭으로 이동합니다.
각 웨이포인트 도착 판정은 캐릭터 위치가 충분히 바뀌지 않으면 도착으로 판정.

사용법:
    mover = WaypointMover(
        waypoints=[
            {"x": 960, "y": 540, "label": "사냥터A", "wait_ms": 1500},
            {"x": 800, "y": 400, "label": "사냥터B", "wait_ms": 1000},
        ],
        capture_offset=(0, 0),
    )
    mover.start()
    while not mover.done:
        mover.tick(pico_worker)
        time.sleep(0.1)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("waypoint_mover")


class WaypointMover:
    """웨이포인트 리스트를 순환하며 이동합니다."""

    def __init__(
        self,
        waypoints: list[dict],
        capture_offset: tuple[int, int] = (0, 0),
        move_timeout_ms: float          = 5000.0,
        loop: bool                      = True,
        click_pulse_ms: int             = 20,
    ):
        """
        Args:
            waypoints: [{"x","y","label"(optional),"wait_ms"(optional)}, ...]
                wait_ms: 이 웨이포인트 도착 후 대기 시간 (기본 1000ms)
            capture_offset: 캡처 영역 오프셋 (ox, oy) — 절대좌표 변환에 사용
            move_timeout_ms: 한 웨이포인트 이동 최대 대기 시간
            loop: True면 마지막 웨이포인트 후 처음으로 순환
            click_pulse_ms: Pico 클릭 pulse 시간
        """
        if not waypoints:
            raise ValueError("waypoints가 비어 있습니다.")

        self.waypoints       = waypoints
        self.capture_offset  = capture_offset
        self.move_timeout_ms = move_timeout_ms
        self.loop            = loop
        self.click_pulse_ms  = click_pulse_ms

        self._idx            = 0       # 현재 목표 웨이포인트 인덱스
        self._state          = "IDLE"  # IDLE / MOVING / WAITING
        self._move_start_t   = 0.0
        self._wait_until_t   = 0.0

    # ── 공개 API ─────────────────────────────────────────────────────

    def start(self) -> None:
        """순환 시작 (처음 웨이포인트로)."""
        self._idx    = 0
        self._state  = "IDLE"
        logger.info(f"[WaypointMover] 시작: {len(self.waypoints)}개 웨이포인트")

    def reset(self) -> None:
        """처음으로 리셋."""
        self.start()

    @property
    def done(self) -> bool:
        """loop=False일 때 모든 웨이포인트 완료 여부."""
        return self._state == "DONE"

    @property
    def current_label(self) -> str:
        """현재 목표 웨이포인트 이름."""
        if self._idx < len(self.waypoints):
            return self.waypoints[self._idx].get("label", f"WP{self._idx}")
        return "DONE"

    @property
    def current_index(self) -> int:
        return self._idx

    def tick(self, pico_worker) -> str:
        """매 루프마다 호출. 현재 상태를 반환합니다.

        Returns:
            "MOVING"   : 이동 중
            "ARRIVED"  : 방금 도착
            "WAITING"  : 도착 후 대기 중
            "DONE"     : 모든 웨이포인트 완료 (loop=False)
            "IDLE"     : 시작 전
        """
        now = time.time()

        # ── IDLE → 첫 웨이포인트로 이동 시작 ─────────────────────────
        if self._state == "IDLE":
            self._move_to_current(pico_worker, now)
            return "MOVING"

        # ── MOVING → 타임아웃 체크 후 도착 처리 ──────────────────────
        if self._state == "MOVING":
            elapsed_ms = (now - self._move_start_t) * 1000.0
            if elapsed_ms >= self.move_timeout_ms:
                # 타임아웃 = 도착으로 간주
                logger.info(
                    f"[WaypointMover] '{self.current_label}' 도착 "
                    f"(타임아웃 {self.move_timeout_ms:.0f}ms)"
                )
                self._on_arrived(now)
                return "ARRIVED"
            return "MOVING"

        # ── WAITING → 대기 완료 후 다음 웨이포인트 ───────────────────
        if self._state == "WAITING":
            if now >= self._wait_until_t:
                self._advance(pico_worker, now)
            return "WAITING"

        # ── DONE ─────────────────────────────────────────────────────
        if self._state == "DONE":
            return "DONE"

        return self._state

    def force_next(self, pico_worker) -> None:
        """현재 웨이포인트를 건너뛰고 다음으로 강제 이동합니다."""
        logger.info(f"[WaypointMover] 강제 다음: '{self.current_label}' 스킵")
        self._advance(pico_worker, time.time())

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _move_to_current(self, pico_worker, now: float) -> None:
        wp    = self.waypoints[self._idx]
        ax    = wp["x"] + self.capture_offset[0]
        ay    = wp["y"] + self.capture_offset[1]
        label = wp.get("label", f"WP{self._idx}")
        clicks      = wp.get("clicks", 1)
        click_delay = wp.get("click_delay_ms", 0) / 1000.0
        logger.info(f"[WaypointMover] → '{label}' ({ax},{ay}) x{clicks} delay={click_delay:.1f}s")
        for i in range(clicks):
            if i > 0 and click_delay > 0:
                time.sleep(click_delay)
            pico_worker.click(ax, ay, self.click_pulse_ms)
        self._move_start_t = now
        self._state = "MOVING"

    def _on_arrived(self, now: float) -> None:
        wp       = self.waypoints[self._idx]
        wait_ms  = wp.get("wait_ms", 1000)
        self._wait_until_t = now + wait_ms / 1000.0
        self._state = "WAITING"

    def _advance(self, pico_worker, now: float) -> None:
        self._idx += 1
        if self._idx >= len(self.waypoints):
            if self.loop:
                self._idx = 0
                logger.info("[WaypointMover] 순환 반복 시작")
                self._move_to_current(pico_worker, now)
            else:
                logger.info("[WaypointMover] 모든 웨이포인트 완료")
                self._state = "DONE"
        else:
            self._move_to_current(pico_worker, now)
