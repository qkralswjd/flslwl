"""
perception/loot.py — 아이템 드롭 테두리 탐지 (HSV 흰색 + dilation).

기존 flslwl/automation/loot_detector.py 에서 개선:
  - from_settings() 팩토리 추가
  - region 파라미터 통일 (dict / tuple 모두 수용)
  - scan_interval_s 캐시 추가
  - invalidate() API 유지
  - 타입 힌트 완비
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# ─────────────────────────── HSV 상수 ────────────────────────────
# 아이템 드롭 박스의 흰색 테두리: 채도 낮고 밝기 높음
_WHITE_LOWER = np.array([0,   0, 200], dtype=np.uint8)
_WHITE_UPPER = np.array([180, 50, 255], dtype=np.uint8)

# dilation 커널 — 테두리 픽셀을 굵게 만들어 연결성 확보
_DIL_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
_DIL_ITERS = 3

# 박스 크기 필터 (노이즈 제거)
_MIN_AREA = 200
_MAX_AREA = 60_000

Region = Union[dict, Tuple[int, int, int, int]]


def _region_to_xywh(region: Region) -> Tuple[int, int, int, int]:
    if isinstance(region, dict):
        return region["x"], region["y"], region["w"], region["h"]
    return tuple(region[:4])  # type: ignore[return-value]


@dataclass
class LootItem:
    """
    탐지된 아이템 드롭 하나.

    Attributes
    ----------
    x, y : int      — ROI 내 바운딩 박스 좌상단 (픽셀)
    w, h : int      — 박스 크기
    cx, cy : int    — 중심 좌표
    label : str     — 항상 "loot" (OCR 제거됨)
    score : float   — 흰 픽셀 비율 기반 신뢰도 (0.0~1.0)
    """
    x: int
    y: int
    w: int
    h: int
    cx: int
    cy: int
    label: str
    score: float

    # 하위 호환: (x, y, label, score) 튜플처럼 unpacking 지원
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.label
        yield self.score


class LootDetector:
    """
    화면 프레임에서 아이템 드롭 테두리(흰색 사각형)를 탐지한다.

    Parameters
    ----------
    region : dict | tuple | None
        탐지 영역. None 이면 전체 프레임 사용.
    scan_interval_s : float
        캐시 TTL(초). 기본 0.5.
    min_area : int
        최소 바운딩 박스 면적(픽셀²). 기본 200.
    max_area : int
        최대 바운딩 박스 면적(픽셀²). 기본 60000.
    """

    def __init__(
        self,
        region: Optional[Region] = None,
        scan_interval_s: float = 0.5,
        min_area: int = _MIN_AREA,
        max_area: int = _MAX_AREA,
    ) -> None:
        if region is not None:
            self._rx, self._ry, self._rw, self._rh = _region_to_xywh(region)
        else:
            self._rx = self._ry = self._rw = self._rh = 0
        self._has_region = region is not None
        self.scan_interval_s = scan_interval_s
        self.min_area = min_area
        self.max_area = max_area

        self._cached: Optional[List[LootItem]] = None
        self._last_scan_at: float = 0.0

    # ──────────────────────────── 팩토리 ────────────────────────────
    @classmethod
    def from_settings(cls, settings) -> "LootDetector":
        """
        Settings.loot 로부터 LootDetector 를 생성한다.

        settings.loot 필드:
            region_x, region_y, region_w, region_h (0 이면 전체 프레임)
            scan_interval_s  (Optional, 기본 0.5)
            min_area         (Optional, 기본 200)
            max_area         (Optional, 기본 60000)
        """
        loot = settings.loot
        rx = getattr(loot, "region_x", 0)
        ry = getattr(loot, "region_y", 0)
        rw = getattr(loot, "region_w", 0)
        rh = getattr(loot, "region_h", 0)
        region = (rx, ry, rw, rh) if (rw > 0 and rh > 0) else None
        return cls(
            region=region,
            scan_interval_s=getattr(loot, "scan_interval_s", 0.5),
            min_area=getattr(loot, "min_area", _MIN_AREA),
            max_area=getattr(loot, "max_area", _MAX_AREA),
        )

    # ──────────────────────────── 공개 API ──────────────────────────
    def find(self, frame: np.ndarray) -> List[LootItem]:
        """
        BGR 프레임에서 아이템 드롭 박스 목록을 반환한다.

        scan_interval_s 이내 재호출 시 캐시를 반환한다.
        반환 좌표는 region 기준 ROI 내 좌표다.
        """
        now = time.monotonic()
        if (
            self._cached is not None
            and now - self._last_scan_at < self.scan_interval_s
        ):
            return self._cached

        items = self._scan(frame)
        self._cached = items
        self._last_scan_at = now
        return items

    def find_nearest(
        self,
        frame: np.ndarray,
        ref_x: int,
        ref_y: int,
    ) -> Optional[LootItem]:
        """
        탐지된 아이템 중 (ref_x, ref_y) 에 가장 가까운 것을 반환한다.
        탐지 결과 없으면 None.
        """
        items = self.find(frame)
        if not items:
            return None
        return min(
            items,
            key=lambda it: (it.cx - ref_x) ** 2 + (it.cy - ref_y) ** 2,
        )

    def invalidate(self) -> None:
        """캐시를 강제 무효화. 다음 find() 호출 시 재탐지한다."""
        self._cached = None
        self._last_scan_at = 0.0

    # ──────────────────────────── 내부 ──────────────────────────────
    def _scan(self, frame: np.ndarray) -> List[LootItem]:
        """실제 탐지 로직 — HSV 마스크 → dilation → 컨투어 → 박스 필터."""
        h, w = frame.shape[:2]

        # ROI 크롭
        if self._has_region and self._rw > 0 and self._rh > 0:
            x1 = max(0, self._rx)
            y1 = max(0, self._ry)
            x2 = min(w, self._rx + self._rw)
            y2 = min(h, self._ry + self._rh)
            roi = frame[y1:y2, x1:x2]
        else:
            x1, y1 = 0, 0
            roi = frame

        if roi.size == 0:
            return []

        # HSV 마스크
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _WHITE_LOWER, _WHITE_UPPER)

        # Dilation — 테두리 픽셀 연결
        mask = cv2.dilate(mask, _DIL_KERNEL, iterations=_DIL_ITERS)

        # 컨투어 → 바운딩 박스
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        items: List[LootItem] = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < self.min_area or area > self.max_area:
                continue

            # 신뢰도: ROI 내 흰 픽셀 비율
            roi_mask = mask[by : by + bh, bx : bx + bw]
            white_ratio = float(np.count_nonzero(roi_mask)) / max(area, 1)
            score = min(white_ratio * 4.0, 1.0)  # 25% 이상이면 1.0

            items.append(
                LootItem(
                    x=bx,
                    y=by,
                    w=bw,
                    h=bh,
                    cx=bx + bw // 2,
                    cy=by + bh // 2,
                    label="loot",
                    score=round(score, 3),
                )
            )

        # 중심 기준 면적 내림차순 정렬
        items.sort(key=lambda it: it.w * it.h, reverse=True)
        return items
