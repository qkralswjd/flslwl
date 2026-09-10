"""ui/preview.py — 실시간 cv2 오버레이 미리보기 창.

화면 캡처 → 탐지 박스 / HP / 레벨 / FSM 상태를 별도 cv2 창에
실시간으로 표시한다. BotApp 의 "시작" 버튼과 연동해 백그라운드
스레드로 실행된다.

사용법 (단독):
    from ui.preview import PreviewWindow
    pw = PreviewWindow(capturer=capturer, detector=detector, tracker=tracker)
    pw.start()        # 별도 스레드로 cv2 창 열기
    ...
    pw.stop()         # 창 닫기

BotApp 연동:
    - _start_mode() → pw.set_mode(mode)
    - _emergency_stop() → pw.stop()
    - preview 스레드는 daemon=True 이므로 앱 종료 시 자동 정리됨

Notes:
    - cv2.imshow 는 반드시 창을 만든 같은 스레드에서 호출해야 한다
      (macOS/Linux 모두 동일). 이 모듈은 자체 스레드에서 창을 소유한다.
    - ScreenCapturer.grab() 이 None 을 반환하면 이전 프레임을 유지한다.
    - detection 없이도 원본 캡처 화면만 표시할 수 있다 (detector=None).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, List, Optional

import cv2
import numpy as np

from core.overlay import (
    draw_detection_zone,
    draw_enemies,
    draw_hud,
    draw_mode_state_banner,
    draw_roi,
)

if TYPE_CHECKING:
    from config.settings import Settings
    from core.detection import RealtimeTemplateDetector
    from core.tracking import Enemy, NearestNeighborTracker
    from hardware.pico import PicoWorker
    from hardware.screen import ScreenCapturer
    from perception.hp import HpReader

logger = logging.getLogger("ui.preview")

WINDOW_NAME = "Bot2 — 실시간 미리보기"

# 목표 표시 FPS (실제 탐지 FPS 와 무관하게 화면 갱신)
_DISPLAY_FPS = 30
_DISPLAY_INTERVAL = 1.0 / _DISPLAY_FPS


# ════════════════════════════════════════════════════════════════════════════
class PreviewWindow:
    """cv2 실시간 오버레이 창 관리자.

    Parameters
    ----------
    capturer    : ScreenCapturer — grab() 메서드로 프레임 획득
    detector    : RealtimeTemplateDetector | None
    tracker     : NearestNeighborTracker | None
    hp_reader   : HpReader | None
    pico        : PicoWorker | None — 연결 상태 표시용
    settings    : Settings | None — ROI / detection_zone 표시용
    frame_grabber : grab_fn | None — capturer 대신 함수 직접 주입 가능
    win_x, win_y : 창 초기 위치
    """

    def __init__(
        self,
        capturer=None,
        detector=None,
        tracker=None,
        hp_reader: Optional["HpReader"] = None,
        pico=None,
        settings=None,
        frame_grabber: Optional[Callable[[], Optional[np.ndarray]]] = None,
        win_x: int = 50,
        win_y: int = 50,
    ):
        self._capturer     = capturer
        self._detector     = detector
        self._tracker      = tracker
        self._hp_reader    = hp_reader
        self._pico         = pico
        self._settings     = settings
        self._win_x        = win_x
        self._win_y        = win_y

        # frame_grabber 우선순위: 인자 > capturer.grab
        if frame_grabber is not None:
            self._grab = frame_grabber
        elif capturer is not None and hasattr(capturer, "grab"):
            self._grab = capturer.grab
        else:
            # 빈 검정 프레임 fallback
            self._grab = lambda: np.zeros((480, 854, 3), dtype=np.uint8)

        # 공유 상태 (스레드에서 갱신)
        self._enemies: List["Enemy"]  = []
        self._hp_ratio: Optional[float] = None
        self._level:    Optional[int]   = None
        self._mode_state: str           = ""
        self._lock = threading.Lock()

        # 제어
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # 통계
        self._capture_fps   = 0.0
        self._detection_fps = 0.0
        self._proc_ms       = 0.0

    # ── 외부 상태 업데이트 API ────────────────────────────────────────

    def update_enemies(self, enemies: List["Enemy"]) -> None:
        """트래커 결과를 미리보기에 반영한다 (스레드 안전)."""
        with self._lock:
            self._enemies = list(enemies)

    def update_hp(self, hp_ratio: float) -> None:
        """HP 비율 갱신 (0.0–1.0)."""
        with self._lock:
            self._hp_ratio = hp_ratio

    def update_level(self, level: int) -> None:
        """레벨 갱신."""
        with self._lock:
            self._level = level

    def update_mode_state(self, state_name: str) -> None:
        """현재 FSM 상태 이름 갱신."""
        with self._lock:
            self._mode_state = state_name

    def set_mode(self, mode) -> None:
        """모드 객체를 연결 — tick 마다 상태를 자동 동기화한다."""
        self._attached_mode = mode

    # ── 생명주기 ─────────────────────────────────────────────────────

    def start(self) -> None:
        """백그라운드 스레드로 cv2 창을 연다."""
        if self._thread and self._thread.is_alive():
            logger.debug("PreviewWindow 이미 실행 중 — start() 무시")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="PreviewWindow",
            daemon=True,
        )
        self._thread.start()
        logger.info("PreviewWindow 시작")

    def stop(self) -> None:
        """미리보기 창을 닫고 스레드를 종료한다."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("PreviewWindow 정지")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── 메인 루프 ────────────────────────────────────────────────────

    def _run(self) -> None:
        """cv2 창 소유 스레드 — 캡처 → 탐지 → 오버레이 → imshow."""
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 854, 480)
            cv2.moveWindow(WINDOW_NAME, self._win_x, self._win_y)
        except Exception as e:
            logger.error(f"cv2 창 생성 실패: {e}")
            return

        last_capture_t   = 0.0
        last_detection_t = 0.0
        frame: Optional[np.ndarray] = None

        # 통계용 지수 이동 평균
        cap_fps_smooth   = 0.0
        det_fps_smooth   = 0.0

        logger.info(f"cv2 오버레이 창 열림: '{WINDOW_NAME}'")

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            # ── 1. 프레임 캡처 ───────────────────────────────────────
            new_frame = self._grab()
            if new_frame is not None and new_frame.size > 0:
                elapsed_cap = time.monotonic() - last_capture_t
                if elapsed_cap > 0:
                    cap_fps_smooth = 0.9 * cap_fps_smooth + 0.1 * (1.0 / elapsed_cap)
                last_capture_t = time.monotonic()
                frame = new_frame.copy()
            
            if frame is None:
                # 캡처 실패 시 검정 화면 + 안내 텍스트
                placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, "캡처 대기 중...",
                    (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (100, 100, 100), 2, cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, placeholder)
                key = cv2.waitKey(int(_DISPLAY_INTERVAL * 1000)) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            display = frame.copy()

            # ── 2. 탐지 + 트래킹 ────────────────────────────────────
            det_start = time.monotonic()

            if self._detector is not None:
                try:
                    detections = self._detector.detect(frame)
                except Exception:
                    detections = []
            else:
                detections = []

            if self._tracker is not None and detections:
                try:
                    dt = max(time.monotonic() - last_detection_t, 1e-3)
                    enemies = self._tracker.update(detections, dt)
                    self.update_enemies(enemies)
                except Exception:
                    pass
            last_detection_t = time.monotonic()

            det_elapsed = time.monotonic() - det_start
            if det_elapsed > 0:
                det_fps_smooth = 0.9 * det_fps_smooth + 0.1 * (1.0 / max(det_elapsed, 1e-3))
            proc_ms = det_elapsed * 1000.0

            # ── 3. 현재 공유 상태 읽기 ──────────────────────────────
            with self._lock:
                enemies      = list(self._enemies)
                hp_ratio     = self._hp_ratio
                level        = self._level
                mode_state   = self._mode_state

            # 모드 객체 연결 시 자동 상태 동기화
            if hasattr(self, "_attached_mode") and self._attached_mode is not None:
                try:
                    mode_state = self._attached_mode.state.name
                except Exception:
                    pass

            # ── 4. HP 읽기 ───────────────────────────────────────────
            if self._hp_reader is not None:
                try:
                    ratio = self._hp_reader.read(frame)
                    if ratio is not None:
                        hp_ratio = ratio
                        self.update_hp(ratio)
                except Exception:
                    pass

            # ── 5. 오버레이 그리기 ───────────────────────────────────
            # Pico 상태
            pico_connected = getattr(self._pico, "is_connected", False) if self._pico else False
            pico_port      = getattr(self._pico, "port",         None)  if self._pico else None

            # 트래커에서 현재 타겟 정보
            current_target_id  = None
            target_state_name  = ""
            if self._tracker is not None:
                current_target_id = getattr(self._tracker, "current_target_id", None)
                ts = getattr(self._tracker, "target_state", None)
                if ts is not None:
                    target_state_name = getattr(ts, "name", "")

            # ROI 정보
            roi            = None
            detection_zone = None
            roi_offset     = (0, 0)
            if self._settings is not None:
                try:
                    import dataclasses
                    d = dataclasses.asdict(self._settings)
                    roi            = d.get("roi")
                    detection_zone = d.get("detection_zone")
                    if roi:
                        roi_offset = (roi.get("x", 0), roi.get("y", 0))
                except Exception:
                    pass

            draw_roi(display, roi)
            draw_detection_zone(display, detection_zone, roi_offset)
            draw_enemies(
                display, enemies, roi_offset,
                current_target_id=current_target_id,
                target_state_name=target_state_name,
            )
            draw_hud(
                display,
                cap_fps_smooth,
                det_fps_smooth,
                proc_ms,
                len(enemies),
                pico_connected=pico_connected,
                pico_port=pico_port,
                target_state_name=target_state_name,
                current_target_id=current_target_id,
                hp_ratio=hp_ratio,
                level=level,
                mode_state=mode_state,
            )
            if mode_state:
                draw_mode_state_banner(display, mode_state)

            # ── 6. 표시 ──────────────────────────────────────────────
            cv2.imshow(WINDOW_NAME, display)

            # 'q' 또는 창 X 버튼 → 중단
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("'q' 키 입력 — 미리보기 종료")
                self._stop_event.set()
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("미리보기 창 닫힘")
                    self._stop_event.set()
                    break
            except cv2.error:
                break

            # FPS 제한 (남은 시간 대기)
            elapsed = time.monotonic() - loop_start
            sleep_t = _DISPLAY_INTERVAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # 정리
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass
        logger.info("PreviewWindow 루프 종료")


# ════════════════════════════════════════════════════════════════════════════
# standalone 실행 (python -m ui.preview)
# ════════════════════════════════════════════════════════════════════════════

def _run_standalone():
    """설정 파일 없이 화면 캡처만 테스트한다."""
    import sys

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    capturer = None
    try:
        from hardware.screen import ScreenCapturer
        capturer = ScreenCapturer()
        logger.info("ScreenCapturer 초기화 성공")
    except Exception as e:
        logger.warning(f"ScreenCapturer 초기화 실패: {e} — 검정 화면으로 대체")

    pw = PreviewWindow(capturer=capturer)
    pw.start()

    print("미리보기 창이 열렸습니다.")
    print("  q  — 창에서 'q' 누르거나 X 버튼으로 종료")
    print("  Ctrl+C — 터미널에서 강제 종료")

    try:
        while pw.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        pw.stop()
        print("종료.")


if __name__ == "__main__":
    _run_standalone()
