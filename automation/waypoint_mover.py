"""좌표 기반 웨이포인트 이동 모듈.

미리 지정한 좌표 리스트를 순서대로 Pico 클릭으로 이동합니다.
각 웨이포인트 도착 판정은 move_timeout_ms 경과 시 도착으로 간주.

픽셀 diff 기반 stuck 감지는 제거되었습니다.
리니지 클래식 클릭 이동 방식은 이동 중/멈춤 중 화면 diff가
거의 동일하여 구분이 불가능합니다 (diff 범위 0.0~0.8로 겹침).
장애물 회피는 move_timeout_ms 조정으로 대응하세요.

사용법:
    mover = WaypointMover(
        waypoints=[
            {"x": 960, "y": 540, "label": "사냥터A", "wait_ms": 1500},
            {"x": 800, "y": 400, "label": "사냥터B", "wait_ms": 1000},
        ],
        capture_offset=(0, 0),
        move_timeout_ms=8000,
    )
    mover.start()
    while not mover.done:
        mover.tick(pico_worker)
        time.sleep(0.1)
"""

import logging
import time
from typing import Optional

import numpy as np

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
        # ── 하위 호환 파라미터 (모두 무시됨) ────────────────────────
        stuck_check_ms: float           = 1500.0,
        stuck_threshold: float          = 2.0,
        stuck_skip: bool                = False,
        baseline_collect_ms: float      = 1500.0,
        stuck_start_ratio: float        = 0.60,
        stuck_ratio: float              = 0.25,
        stuck_min_baseline: float       = 0.30,
        stuck_max: int                  = 3,
    ):
        """
        Args:
            waypoints: [{"x","y","label"(optional),"wait_ms"(optional),
                         "clicks"(optional),"click_delay_ms"(optional)}, ...]
                wait_ms: 이 웨이포인트 도착 후 대기 시간 (기본 1000ms)
                clicks: 클릭 횟수 (기본 1)
                click_delay_ms: 클릭 사이 딜레이 ms (기본 0)
            capture_offset: 캡처 영역 오프셋 (ox, oy) — 절대좌표 변환에 사용
            move_timeout_ms: 한 웨이포인트 이동 최대 대기 시간 (경과 시 도착 간주)
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

        self._idx          = 0       # 현재 목표 웨이포인트 인덱스
        self._state        = "IDLE"  # IDLE / MOVING / WAITING / DONE
        self._move_start_t = 0.0
        self._wait_until_t = 0.0

    # ── 공개 API ────────────────────────────────────────────────────

    def start(self) -> None:
        """순환 시작 (처음 웨이포인트로)."""
        self._idx   = 0
        self._state = "IDLE"
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

    def tick(self, pico_worker, frame=None) -> str:
        """매 루프마다 호출. 현재 상태를 반환합니다.

        Args:
            pico_worker: Pico 워커 (클릭 명령 전송)
            frame: 사용하지 않음 (하위 호환용 — 무시됨)

        Returns:
            "MOVING"   : 이동 중
            "ARRIVED"  : 방금 도착 (타임아웃)
            "WAITING"  : 도착 후 대기 중
            "DONE"     : 모든 웨이포인트 완료 (loop=False)
            "IDLE"     : 시작 전
        """
        now = time.time()

        # ── IDLE → 첫 웨이포인트로 이동 시작 ───────────────────────
        if self._state == "IDLE":
            self._move_to_current(pico_worker, now)
            return "MOVING"

        # ── MOVING → 타임아웃 체크 ──────────────────────────────────
        if self._state == "MOVING":
            elapsed_ms = (now - self._move_start_t) * 1000.0
            if elapsed_ms >= self.move_timeout_ms:
                logger.info(
                    f"[WaypointMover] '{self.current_label}' 도착 "
                    f"(타임아웃 {self.move_timeout_ms:.0f}ms)"
                )
                self._on_arrived(now)
                return "ARRIVED"
            return "MOVING"

        # ── WAITING → 대기 완료 후 다음 웨이포인트 ─────────────────
        if self._state == "WAITING":
            if now >= self._wait_until_t:
                self._advance(pico_worker, now)
            return "WAITING"

        # ── DONE ────────────────────────────────────────────────────
        if self._state == "DONE":
            return "DONE"

        return self._state

    def force_next(self, pico_worker) -> None:
        """현재 웨이포인트를 건너뛰고 다음으로 강제 이동합니다."""
        logger.info(f"[WaypointMover] 강제 다음: '{self.current_label}' 스킵")
        self._advance(pico_worker, time.time())

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _move_to_current(self, pico_worker, now: float) -> None:
        wp          = self.waypoints[self._idx]
        ax          = wp["x"] + self.capture_offset[0]
        ay          = wp["y"] + self.capture_offset[1]
        label       = wp.get("label", f"WP{self._idx}")
        clicks      = wp.get("clicks", 1)
        click_delay = wp.get("click_delay_ms", 0) / 1000.0
        logger.info(
            f"[WaypointMover] → '{label}' ({ax},{ay}) "
            f"x{clicks} delay={click_delay:.1f}s"
        )
        for i in range(clicks):
            if i > 0 and click_delay > 0:
                time.sleep(click_delay)
            pico_worker.click(ax, ay, self.click_pulse_ms)
        self._move_start_t = now
        self._state        = "MOVING"

    def _on_arrived(self, now: float) -> None:
        wp                 = self.waypoints[self._idx]
        wait_ms            = wp.get("wait_ms", 1000)
        self._wait_until_t = now + wait_ms / 1000.0
        self._state        = "WAITING"

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
