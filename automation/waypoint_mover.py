"""좌표 기반 웨이포인트 이동 모듈.

미리 지정한 좌표 리스트를 순서대로 Pico 클릭으로 이동합니다.
각 웨이포인트 도착 판정은 move_timeout_ms 경과 시 도착으로 간주.

장애물 감지:
    tick()에 frame을 넘기면 이동 중 화면 변화량을 감시합니다.
    stuck_check_ms 경과 후 픽셀 변화량이 stuck_threshold 미만이면
    장애물에 걸린 것으로 판단 → 다음 웨이포인트로 강제 스킵.

사용법:
    mover = WaypointMover(
        waypoints=[
            {"x": 960, "y": 540, "label": "사냥터A", "wait_ms": 1500},
            {"x": 800, "y": 400, "label": "사냥터B", "wait_ms": 1000},
        ],
        capture_offset=(0, 0),
        stuck_check_ms=1500,    # 클릭 후 이 시간 뒤 멈춤 체크
        stuck_threshold=2.0,    # 픽셀 변화량 이하 = 멈춤
    )
    mover.start()
    while not mover.done:
        frame = capturer.grab()
        mover.tick(pico_worker, frame=frame)
        time.sleep(0.1)
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("waypoint_mover")


def _frame_diff(f1: np.ndarray, f2: np.ndarray) -> float:
    """두 프레임의 평균 픽셀 변화량 (0.0 ~ 255.0)."""
    if f1 is None or f2 is None:
        return 999.0
    # 화면 중앙 50% 영역만 비교 (UI 제외)
    h, w = f1.shape[:2]
    y0, y1 = h // 4, h * 3 // 4
    x0, x1 = w // 4, w * 3 // 4
    crop1 = cv2.cvtColor(f1[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop2 = cv2.cvtColor(f2[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(crop1.astype(np.int16) - crop2.astype(np.int16))))


class WaypointMover:
    """웨이포인트 리스트를 순환하며 이동합니다."""

    def __init__(
        self,
        waypoints: list[dict],
        capture_offset: tuple[int, int] = (0, 0),
        move_timeout_ms: float          = 5000.0,
        loop: bool                      = True,
        click_pulse_ms: int             = 20,
        stuck_check_ms: float           = 1500.0,  # 클릭 후 멈춤 체크 시작까지 대기
        stuck_threshold: float          = 2.0,     # 픽셀 변화량 이하 = 멈춤
        stuck_skip: bool                = True,    # 멈춤 감지 시 다음 WP 스킵
    ):
        """
        Args:
            waypoints: [{"x","y","label"(optional),"wait_ms"(optional)}, ...]
                wait_ms: 이 웨이포인트 도착 후 대기 시간 (기본 1000ms)
            capture_offset: 캡처 영역 오프셋 (ox, oy) — 절대좌표 변환에 사용
            move_timeout_ms: 한 웨이포인트 이동 최대 대기 시간
            loop: True면 마지막 웨이포인트 후 처음으로 순환
            click_pulse_ms: Pico 클릭 pulse 시간
            stuck_check_ms: 클릭 후 이 시간(ms) 뒤부터 멈춤 감지 시작
            stuck_threshold: 픽셀 평균 변화량이 이 값 이하면 멈춤으로 판정
            stuck_skip: True면 멈춤 감지 시 다음 웨이포인트로 스킵
        """
        if not waypoints:
            raise ValueError("waypoints가 비어 있습니다.")

        self.waypoints       = waypoints
        self.capture_offset  = capture_offset
        self.move_timeout_ms = move_timeout_ms
        self.loop            = loop
        self.click_pulse_ms  = click_pulse_ms
        self.stuck_check_ms  = stuck_check_ms
        self.stuck_threshold = stuck_threshold
        self.stuck_skip      = stuck_skip

        self._idx            = 0       # 현재 목표 웨이포인트 인덱스
        self._state          = "IDLE"  # IDLE / MOVING / WAITING
        self._move_start_t   = 0.0
        self._wait_until_t   = 0.0

        # ── 장애물 감지용 ────────────────────────────────────────────
        self._prev_frame: Optional[np.ndarray] = None   # 멈춤 체크용 이전 프레임
        self._stuck_check_t  = 0.0    # 이 시각 이후부터 멈춤 체크
        self._stuck_count    = 0      # 연속 멈춤 횟수
        self._stuck_max      = 3      # 이 횟수 연속 멈춤 → 스킵

    # ── 공개 API ─────────────────────────────────────────────────────

    def start(self) -> None:
        """순환 시작 (처음 웨이포인트로)."""
        self._idx        = 0
        self._state      = "IDLE"
        self._prev_frame = None
        self._stuck_count = 0
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

    def tick(self, pico_worker, frame: Optional[np.ndarray] = None) -> str:
        """매 루프마다 호출. 현재 상태를 반환합니다.

        Args:
            pico_worker: Pico 워커 (클릭 명령 전송)
            frame: 현재 화면 프레임 (장애물 감지용, None이면 감지 비활성)

        Returns:
            "MOVING"   : 이동 중
            "ARRIVED"  : 방금 도착
            "STUCK"    : 장애물 감지 → 다음 WP 스킵
            "WAITING"  : 도착 후 대기 중
            "DONE"     : 모든 웨이포인트 완료 (loop=False)
            "IDLE"     : 시작 전
        """
        now = time.time()

        # ── IDLE → 첫 웨이포인트로 이동 시작 ─────────────────────────
        if self._state == "IDLE":
            self._move_to_current(pico_worker, now)
            self._prev_frame  = frame.copy() if frame is not None else None
            self._stuck_check_t = now + self.stuck_check_ms / 1000.0
            self._stuck_count   = 0
            return "MOVING"

        # ── MOVING → 장애물 감지 + 타임아웃 체크 ────────────────────
        if self._state == "MOVING":
            elapsed_ms = (now - self._move_start_t) * 1000.0

            # 장애물 감지 (frame 있을 때만)
            if (frame is not None
                    and self.stuck_skip
                    and now >= self._stuck_check_t):

                diff = _frame_diff(self._prev_frame, frame)

                if diff < self.stuck_threshold:
                    self._stuck_count += 1
                    logger.warning(
                        f"[WaypointMover] '{self.current_label}' "
                        f"멈춤 감지 (diff={diff:.2f} < {self.stuck_threshold}) "
                        f"[{self._stuck_count}/{self._stuck_max}]"
                    )
                    if self._stuck_count >= self._stuck_max:
                        logger.warning(
                            f"[WaypointMover] '{self.current_label}' "
                            f"장애물 — 다음 웨이포인트로 스킵"
                        )
                        self._stuck_count = 0
                        self._advance(pico_worker, now)
                        return "STUCK"
                else:
                    # 움직임 있음 → 카운터 리셋
                    if self._stuck_count > 0:
                        logger.debug(
                            f"[WaypointMover] '{self.current_label}' "
                            f"이동 재개 (diff={diff:.2f})"
                        )
                    self._stuck_count = 0

                # 다음 체크까지 500ms 대기
                self._prev_frame    = frame.copy()
                self._stuck_check_t = now + 0.5

            # 타임아웃 = 도착으로 간주
            if elapsed_ms >= self.move_timeout_ms:
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
                if frame is not None:
                    self._prev_frame    = frame.copy()
                    self._stuck_check_t = now + self.stuck_check_ms / 1000.0
                    self._stuck_count   = 0
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
