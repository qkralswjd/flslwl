"""hardware/screen.py — mss 기반 화면 캡처.

기존 capture/screen_capture.py 대비 개선점:
  - Settings 객체로 직접 초기화 가능 (from_settings 클래스메서드)
  - grab_gray() 편의 메서드 추가 (그레이스케일 직접 반환)
  - frame_size 프로퍼티 (w, h) 추가
  - 컨텍스트 매니저(with 문) 지원
  - 모니터 목록 반환 타입을 list[dict]로 명시
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from config.settings import Settings


class ScreenCapturer:
    """mss 기반 단일 화면 캡처기.

    사용 예::

        cap = ScreenCapturer(monitor_index=2)
        frame = cap.grab()               # BGR ndarray
        gray  = cap.grab_gray()          # GRAY ndarray

        # with 문으로 자동 close
        with ScreenCapturer.from_settings(settings) as cap:
            frame = cap.grab()
    """

    def __init__(
        self,
        monitor_index: int = 1,
        region: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Args:
            monitor_index: mss.monitors 인덱스 (0=전체, 1..N=물리 모니터).
            region: 모니터 내 서브 영역 dict {x, y, width, height}.
                    None 이면 모니터 전체를 캡처한다.
        """
        self._sct = mss.mss()
        self.monitor_index = monitor_index
        self.region = region
        self._monitor = self._resolve_monitor()

    # ── 팩토리 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ScreenCapturer":
        """Settings 객체로부터 ScreenCapturer를 생성한다.

        settings.capture.region_w == 0 이면 region 없이 모니터 전체를 캡처한다.
        """
        c = settings.capture
        region: Optional[Dict[str, int]] = None
        if c.region_w > 0 and c.region_h > 0:
            region = {
                "x": c.region_x,
                "y": c.region_y,
                "width": c.region_w,
                "height": c.region_h,
            }
        return cls(monitor_index=c.monitor_index, region=region)

    # ── 모니터 해석 ────────────────────────────────────────────────────────

    def _resolve_monitor(self) -> Dict[str, int]:
        monitors = self._sct.monitors
        if not (0 <= self.monitor_index < len(monitors)):
            raise ValueError(
                f"monitor_index {self.monitor_index} 범위 초과 "
                f"(유효: 0..{len(monitors) - 1})"
            )
        base = monitors[self.monitor_index]
        if self.region:
            return {
                "left":   base["left"] + self.region["x"],
                "top":    base["top"]  + self.region["y"],
                "width":  self.region["width"],
                "height": self.region["height"],
            }
        return dict(base)

    # ── 설정 변경 ──────────────────────────────────────────────────────────

    def set_monitor(self, monitor_index: int) -> None:
        """모니터 인덱스를 변경하고 내부 영역을 재계산한다."""
        self.monitor_index = monitor_index
        self._monitor = self._resolve_monitor()

    def set_region(self, region: Optional[Dict[str, int]]) -> None:
        """캡처 서브 영역을 변경한다. None 이면 전체 모니터 사용."""
        self.region = region
        self._monitor = self._resolve_monitor()

    # ── 프로퍼티 ───────────────────────────────────────────────────────────

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height) 픽셀 크기를 반환한다."""
        return self._monitor["width"], self._monitor["height"]

    def list_monitors(self) -> List[Dict[str, int]]:
        """mss가 인식하는 모니터 목록을 반환한다 (0=all, 1..N=physical)."""
        return list(self._sct.monitors)

    # ── 캡처 ──────────────────────────────────────────────────────────────

    def grab(self) -> np.ndarray:
        """현재 설정 영역을 캡처해 BGR ndarray로 반환한다."""
        shot = self._sct.grab(self._monitor)
        frame = np.asarray(shot)          # BGRA uint8
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def grab_gray(self) -> np.ndarray:
        """캡처 후 그레이스케일로 변환해 반환한다."""
        return cv2.cvtColor(self.grab(), cv2.COLOR_BGR2GRAY)

    # ── 자원 해제 ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """mss 컨텍스트를 닫는다."""
        self._sct.close()

    def __enter__(self) -> "ScreenCapturer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
