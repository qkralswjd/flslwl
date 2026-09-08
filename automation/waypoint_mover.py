"""좌표 기반 웨이포인트 이동 모듈.

미리 지정한 좌표 리스트를 순서대로 Pico 클릭으로 이동합니다.
각 웨이포인트 도착 판정은 move_timeout_ms 경과 시 도착으로 간주.

장애물 감지 (baseline-relative 방식):
    이동 시작 직후 baseline_collect_ms 동안 diff를 수집해 평균 baseline을 산출.
    그 후 타임아웃의 stuck_start_ratio 비율 이상 경과하면 체크 시작.
    current_diff <= baseline * stuck_ratio 이면 장애물로 판단 → 다음 웨이포인트로 스킵.

    이 방식은 게임마다 절대 diff 값이 달라도 자동 적응합니다.
    baseline이 충분히 수집되지 않으면 (이동이 너무 짧거나 화면이 정지 상태)
    오탐지를 방지하기 위해 stuck 체크를 하지 않습니다.

    이전 파라미터(stuck_check_ms, stuck_threshold)는 하위 호환을 위해 시그니처에
    남아있지만 실제 동작에 영향을 주지 않습니다.

사용법:
    mover = WaypointMover(
        waypoints=[
            {"x": 960, "y": 540, "label": "사냥터A", "wait_ms": 1500},
            {"x": 800, "y": 400, "label": "사냥터B", "wait_ms": 1000},
        ],
        capture_offset=(0, 0),
        baseline_collect_ms=1500,  # 이동 직후 baseline 수집 구간 (ms)
        stuck_start_ratio=0.60,    # 타임아웃의 이 비율 이후부터 stuck 체크
        stuck_ratio=0.25,          # baseline 대비 이 배율 이하면 멈춤
        stuck_min_baseline=0.30,   # baseline 평균이 이 값 미만이면 체크 스킵
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
    """두 프레임의 평균 픽셀 변화량 (0.0 ~ 255.0).

    화면 중앙 50% 영역만 비교 (상단 UI 및 하단 HUD 제외).
    """
    if f1 is None or f2 is None:
        return 0.0
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
        # ── 하위 호환 파라미터 (동작에 영향 없음) ───────────────────
        stuck_check_ms: float           = 1500.0,
        stuck_threshold: float          = 2.0,
        stuck_skip: bool                = True,
        # ── baseline-relative 장애물 감지 파라미터 ──────────────────
        baseline_collect_ms: float      = 1500.0,
        stuck_start_ratio: float        = 0.60,
        stuck_ratio: float              = 0.25,
        stuck_min_baseline: float       = 0.30,
        stuck_max: int                  = 3,
    ):
        """
        Args:
            waypoints: [{"x","y","label"(optional),"wait_ms"(optional)}, ...]
                wait_ms: 이 웨이포인트 도착 후 대기 시간 (기본 1000ms)
            capture_offset: 캡처 영역 오프셋 (ox, oy) — 절대좌표 변환에 사용
            move_timeout_ms: 한 웨이포인트 이동 최대 대기 시간
            loop: True면 마지막 웨이포인트 후 처음으로 순환
            click_pulse_ms: Pico 클릭 pulse 시간
            stuck_skip: False면 장애물 감지 전체 비활성화
            baseline_collect_ms: 이동 직후 diff baseline 수집 구간 (ms)
                기본값 1500ms. move_timeout_ms * stuck_start_ratio 보다 작아야 함.
            stuck_start_ratio: 타임아웃 비율 이후부터 stuck 체크 (기본 0.6 = 60%)
                move_timeout_ms=8000 이면 4800ms(4.8초) 이후부터 체크.
            stuck_ratio: baseline 대비 이 배율 이하면 멈춤 판정 (기본 0.25)
                이동 시 diff가 0.5, 멈춤 시 diff가 0.05 → ratio=0.25면 0.5*0.25=0.125 → 0.05 ≤ 0.125 판정.
            stuck_min_baseline: baseline 평균이 이 값 미만이면 체크 스킵 (기본 0.30)
                화면이 원래 정지 상태(캐릭터 이동 없음)이면 baseline 자체가 낮아 오탐지 방지.
            stuck_max: 연속 stuck 판정 횟수 이상 시 다음 WP 스킵 (기본 3)
        """
        if not waypoints:
            raise ValueError("waypoints가 비어 있습니다.")

        self.waypoints           = waypoints
        self.capture_offset      = capture_offset
        self.move_timeout_ms     = move_timeout_ms
        self.loop                = loop
        self.click_pulse_ms      = click_pulse_ms
        self.stuck_skip          = stuck_skip
        self.baseline_collect_ms = baseline_collect_ms
        self.stuck_start_ratio   = stuck_start_ratio
        self.stuck_ratio         = stuck_ratio
        self.stuck_min_baseline  = stuck_min_baseline
        self._stuck_max          = stuck_max

        self._idx          = 0       # 현재 목표 웨이포인트 인덱스
        self._state        = "IDLE"  # IDLE / MOVING / WAITING / DONE
        self._move_start_t = 0.0    # 큐 투입 시각 (타임아웃 기준)
        self._wait_until_t = 0.0

        # ── 장애물 감지 내부 상태 ──────────────────────────────────
        self._prev_frame: Optional[np.ndarray] = None
        self._baseline_samples: list[float]   = []
        self._baseline_avg: float             = 0.0
        self._baseline_ready: bool            = False
        self._stuck_count: int                = 0
        self._next_check_t: float             = 0.0
        # 클릭 ACK 대기: pico 큐가 비워진 후에 baseline 수집 시작
        self._waiting_pico_idle: bool         = True  # True = 아직 큐 처리 대기 중
        self._baseline_start_t: float         = 0.0   # 큐 비워진 시각

    # ── 공개 API ────────────────────────────────────────────────────

    def start(self) -> None:
        """순환 시작 (처음 웨이포인트로)."""
        self._idx   = 0
        self._state = "IDLE"
        self._reset_stuck_state()
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

        # ── IDLE → 첫 웨이포인트로 이동 시작 ───────────────────────
        if self._state == "IDLE":
            self._move_to_current(pico_worker, now)
            self._init_stuck_for_move(frame, now)
            return "MOVING"

        # ── MOVING → baseline 수집 + stuck 체크 + 타임아웃 ─────────
        if self._state == "MOVING":
            elapsed_ms = (now - self._move_start_t) * 1000.0

            if frame is not None and self.stuck_skip:
                diff = _frame_diff(self._prev_frame, frame)

                # 단계 0: pico 큐가 빌 때까지 대기 (실제 클릭 ACK 완료 시점 감지)
                if self._waiting_pico_idle:
                    pico_idle = getattr(pico_worker, "is_idle", True)
                    if pico_idle:
                        # 큐 처리 완료 → 이 시점부터 baseline 수집 시작
                        self._waiting_pico_idle = False
                        self._baseline_start_t  = now
                        self._prev_frame        = frame.copy()
                        logger.debug(
                            f"[WaypointMover] '{self.current_label}' "
                            f"Pico 처리 완료 → baseline 수집 시작 "
                            f"(큐대기 {elapsed_ms:.0f}ms)"
                        )
                    # 아직 대기 중: frame만 갱신하고 체크 스킵
                    else:
                        self._prev_frame = frame.copy()

                # 단계 1: baseline 수집 (Pico idle 후 baseline_collect_ms 동안)
                elif not self._baseline_ready:
                    baseline_elapsed = (now - self._baseline_start_t) * 1000.0
                    if baseline_elapsed <= self.baseline_collect_ms:
                        if diff > 0.0:
                            self._baseline_samples.append(diff)
                    else:
                        # 수집 완료 → 평균 산출
                        if self._baseline_samples:
                            self._baseline_avg = float(
                                np.mean(self._baseline_samples)
                            )
                        else:
                            self._baseline_avg = 0.0
                        self._baseline_ready = True
                        logger.debug(
                            f"[WaypointMover] '{self.current_label}' "
                            f"baseline 확정: {self._baseline_avg:.3f} "
                            f"(n={len(self._baseline_samples)})"
                        )

                # 단계 2: stuck 체크 (baseline 확정 후 + 타임아웃 비율 이상 경과)
                elif now >= self._next_check_t:
                    stuck_start_ms = self.move_timeout_ms * self.stuck_start_ratio
                    if elapsed_ms >= stuck_start_ms:
                        self._run_stuck_check(diff, pico_worker, now)
                        if self._state != "MOVING":
                            return "STUCK"
                    self._next_check_t = now + 0.5  # 다음 체크 0.5초 후

                self._prev_frame = frame.copy()

            # 타임아웃 = 도착으로 간주
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
                self._init_stuck_for_move(frame, now)
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

    def _reset_stuck_state(self) -> None:
        """장애물 감지 관련 상태 전체 초기화."""
        self._prev_frame          = None
        self._baseline_samples    = []
        self._baseline_avg        = 0.0
        self._baseline_ready      = False
        self._stuck_count         = 0
        self._next_check_t        = 0.0
        self._waiting_pico_idle   = True   # 새 이동 시작 → 큐 처리 대기 모드로
        self._baseline_start_t    = 0.0

    def _init_stuck_for_move(
        self,
        frame: Optional[np.ndarray],
        now: float,
    ) -> None:
        """새 웨이포인트 이동 시작 시 stuck 상태 초기화."""
        self._reset_stuck_state()
        if frame is not None:
            self._prev_frame = frame.copy()
        self._next_check_t = now

    def _run_stuck_check(
        self,
        diff: float,
        pico_worker,
        now: float,
    ) -> None:
        """diff vs baseline 비교로 stuck 판정.

        baseline이 stuck_min_baseline 미만이면 (화면이 원래 정지 상태)
        오탐지 방지를 위해 체크하지 않습니다.
        """
        if self._baseline_avg < self.stuck_min_baseline:
            logger.debug(
                f"[WaypointMover] '{self.current_label}' "
                f"baseline 낮음 ({self._baseline_avg:.3f} < "
                f"{self.stuck_min_baseline}) — stuck 체크 스킵"
            )
            return

        threshold = self._baseline_avg * self.stuck_ratio
        if diff <= threshold:
            self._stuck_count += 1
            logger.warning(
                f"[WaypointMover] '{self.current_label}' "
                f"멈춤 감지 (diff={diff:.3f} ≤ "
                f"baseline×{self.stuck_ratio}={threshold:.3f}) "
                f"[{self._stuck_count}/{self._stuck_max}]"
            )
            if self._stuck_count >= self._stuck_max:
                logger.warning(
                    f"[WaypointMover] '{self.current_label}' "
                    f"장애물 확정 (baseline={self._baseline_avg:.3f}) "
                    f"— 다음 웨이포인트로 스킵"
                )
                self._stuck_count = 0
                self._advance(pico_worker, now)
        else:
            if self._stuck_count > 0:
                logger.debug(
                    f"[WaypointMover] '{self.current_label}' "
                    f"이동 재개 (diff={diff:.3f} > threshold={threshold:.3f})"
                )
            self._stuck_count = 0

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
