"""
perception/hp.py — HP 바 비율 읽기 (HSV 빨간색 열 비율).

기존 flslwl/automation/hp_reader.py 에서 개선:
  - from_settings() 팩토리 추가
  - region 을 (x, y, w, h) tuple 과 dict 둘 다 수용
  - read_interval_s 캐시 유지
  - 타입 힌트 완비
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# ─────────────────────────── HSV 상수 ────────────────────────────
# 빨간색은 Hue가 0°~10° 와 170°~180° 두 범위에 걸쳐 있다.
_HP_HSV_RANGES: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((0,   80, 40), (10,  255, 255)),   # 빨간 영역 1 (Hue 0–10)
    ((170, 80, 40), (180, 255, 255)),   # 빨간 영역 2 (Hue 170–180)
]

Region = Union[dict, Tuple[int, int, int, int]]


def _region_to_xywh(region: Region) -> Tuple[int, int, int, int]:
    """dict {'x','y','w','h'} 또는 (x,y,w,h) tuple 을 통일 변환."""
    if isinstance(region, dict):
        return region["x"], region["y"], region["w"], region["h"]
    return tuple(region[:4])  # type: ignore[return-value]


class HpReader:
    """
    화면 캡처 프레임에서 HP 바의 빨간 픽셀 비율을 읽어 HP% 를 반환한다.

    Parameters
    ----------
    region : dict | tuple
        HP 바가 위치한 영역. {'x':…,'y':…,'w':…,'h':…} 또는 (x,y,w,h).
    threshold_pct : float
        is_low() 판단 기준 퍼센트 (기본 50.0).
    read_interval_s : float
        동일 프레임 내 중복 읽기를 막는 캐시 TTL(초). 기본 0.5.
    """

    def __init__(
        self,
        region: Region,
        threshold_pct: float = 50.0,
        read_interval_s: float = 0.5,
    ) -> None:
        self._rx, self._ry, self._rw, self._rh = _region_to_xywh(region)
        self.threshold_pct = threshold_pct
        self.read_interval_s = read_interval_s

        self._cached_pct: Optional[float] = None
        self._last_read_at: float = 0.0

    # ──────────────────────────── 팩토리 ────────────────────────────
    @classmethod
    def from_settings(cls, settings) -> "HpReader":
        """
        Settings 객체로부터 HpReader 를 생성한다.

        settings.hp_bar 에 다음 필드가 있어야 한다:
            region_x, region_y, region_w, region_h
            threshold_pct  (Optional, 기본 50.0)
            read_interval_s (Optional, 기본 0.5)
        """
        hp = settings.hp_bar
        region = (hp.region_x, hp.region_y, hp.region_w, hp.region_h)
        return cls(
            region=region,
            threshold_pct=getattr(hp, "threshold_pct", 50.0),
            read_interval_s=getattr(hp, "read_interval_s", 0.5),
        )

    # ──────────────────────────── 공개 API ──────────────────────────
    def read(self, frame: np.ndarray) -> float:
        """
        BGR 프레임에서 HP 바 영역을 잘라 빨간 픽셀 열 비율을 계산한다.

        Returns
        -------
        float
            0.0 (HP 없음) ~ 100.0 (HP 최대). 읽기 실패 시 캐시값 또는 0.0.
        """
        now = time.monotonic()
        if (
            self._cached_pct is not None
            and now - self._last_read_at < self.read_interval_s
        ):
            return self._cached_pct

        pct = self._compute(frame)
        self._cached_pct = pct
        self._last_read_at = now
        return pct

    def get_cached(self) -> Optional[float]:
        """마지막으로 계산된 HP% 를 반환. read() 호출 전이면 None."""
        return self._cached_pct

    def is_low(self, threshold_pct: Optional[float] = None) -> bool:
        """
        캐시된 HP% 가 기준치 이하인지 반환.

        Parameters
        ----------
        threshold_pct : float | None
            None 이면 생성 시 설정된 threshold_pct 사용.
        """
        if self._cached_pct is None:
            return False
        thr = threshold_pct if threshold_pct is not None else self.threshold_pct
        return self._cached_pct <= thr

    def invalidate(self) -> None:
        """캐시를 강제 무효화. 다음 read() 호출 시 재계산된다."""
        self._cached_pct = None
        self._last_read_at = 0.0

    # ──────────────────────────── 내부 ──────────────────────────────
    def _compute(self, frame: np.ndarray) -> float:
        """
        HP 바 ROI를 잘라 빨간 픽셀이 있는 열(column)의 비율을 반환한다.

        알고리즘:
          1. BGR → HSV 변환
          2. 두 빨간 HSV 범위로 마스크 생성
          3. 열(column)별로 하나라도 빨간 픽셀이 있으면 그 열을 "활성"으로 간주
          4. 활성 열 수 / 전체 열 수 × 100 = HP%
        """
        h, w = frame.shape[:2]
        x1 = max(0, self._rx)
        y1 = max(0, self._ry)
        x2 = min(w, self._rx + self._rw)
        y2 = min(h, self._ry + self._rh)
        if x2 <= x1 or y2 <= y1:
            return self._cached_pct or 0.0

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _HP_HSV_RANGES:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # 열(column)마다 빨간 픽셀 존재 여부 (axis=0 → 세로 방향 OR)
        col_has_red = np.any(mask > 0, axis=0)
        total_cols = col_has_red.shape[0]
        if total_cols == 0:
            return 0.0

        red_cols = int(np.sum(col_has_red))
        return round(red_cols / total_cols * 100.0, 1)
