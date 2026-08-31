"""HP 바 인식 모듈.

화면의 지정 영역에서 HP 바의 HSV 파란색 픽셀 비율로 HP%를 추정합니다.

리니지 클래식 HP 바는 파란색(royal blue) 계열입니다.

사용법:
    reader = HpReader(region={"x": 0, "y": 0, "width": 200, "height": 10})
    hp_pct = reader.read(frame)   # 0.0 ~ 100.0
    if reader.is_low(50.0):       # HP 50% 미만 체크
        use_potion()
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("hp_reader")

# 리니지 클래식 HP 바 HSV 범위 (빨간색 계열)
# 실측값: center_HSV=[0, 252, 87] → H=0, S=252, V=87
# OpenCV HSV: H=0~180, S=0~255, V=0~255
_HP_HSV_RANGES = [
    # 빨간색 영역 1 (H=0~10, V 하한 40으로 낮춤 — 어두운 빨간도 포함)
    ((0, 80, 40), (10, 255, 255)),
    # 빨간색 영역 2 (H=170~180, 색상환 반대편)
    ((170, 80, 40), (180, 255, 255)),
]


def _calc_hp_pct(crop: np.ndarray) -> float:
    """HP 바 크롭 이미지에서 HP% 를 계산합니다.

    가로 방향 HP 바 길이 비율로 계산합니다.
    - 각 열(column)에 빨간 픽셀이 하나라도 있으면 "채워진 열"로 판단
    - 왼쪽부터 연속으로 채워진 열의 비율 = HP%
    - 전체 픽셀 비율 방식은 테두리/여백 포함 시 50% 오류 발생

    Args:
        crop: BGR 이미지 (HP 바 영역)

    Returns:
        HP% (0.0 ~ 100.0)
    """
    if crop.size == 0:
        return 100.0

    # BGR → HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 빨간색 픽셀 마스크 생성
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in _HP_HSV_RANGES:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.bitwise_or(mask, m)

    total_cols = mask.shape[1]
    if total_cols == 0:
        return 100.0

    # 각 열에 빨간 픽셀이 하나라도 있으면 True
    col_has_red = np.any(mask > 0, axis=0)  # shape: (width,)

    # HP 바 중간에 텍스트("HP : 109 / 109")가 있어 연속 열이 끊김
    # → 전체 빨간 열 수 비율로 계산
    red_cols = int(np.count_nonzero(col_has_red))
    hp_pct = round((red_cols / total_cols) * 100.0, 1)

    return hp_pct


class HpReader:
    """화면 지정 영역에서 HP 바 비율을 읽습니다.

    HSV 빨간색 픽셀 비율로 HP%를 추정합니다.
    HP 바가 꽉 찬 상태 = 100%, 완전히 빈 상태 = 0%.

    주의:
        HP 바 영역을 정확히 지정해야 합니다.
        다른 빨간색 UI 요소가 포함되면 오작동할 수 있습니다.
    """

    def __init__(
        self,
        region: dict,
        threshold_pct: float = 50.0,
        read_interval_s: float = 0.5,
    ):
        """
        Args:
            region: {"x","y","width","height"} — 절대 화면 좌표 기준
            threshold_pct: is_low() 판정 기준 (기본 50%)
            read_interval_s: 읽기 최소 간격 (초) — CPU 과부하 방지
        """
        self.region         = region
        self.threshold_pct  = threshold_pct
        self.read_interval  = read_interval_s

        self._last_read_time: float = 0.0
        self._cached_hp: float      = 100.0

    def read(self, frame: np.ndarray) -> float:
        """프레임에서 HP%를 읽어 반환합니다.

        read_interval 보다 짧은 간격으로 호출되면 캐시 값을 반환합니다.

        Args:
            frame: 캡처 프레임 (BGR numpy array)

        Returns:
            HP% (0.0 ~ 100.0)
        """
        now = time.time()
        if now - self._last_read_time < self.read_interval:
            return self._cached_hp

        self._last_read_time = now

        # region 크롭
        x = self.region.get("x", 0)
        y = self.region.get("y", 0)
        w = self.region.get("width", 200)
        h = self.region.get("height", 10)

        # 프레임 경계 체크
        fh, fw = frame.shape[:2]
        x2 = min(x + w, fw)
        y2 = min(y + h, fh)

        if x >= fw or y >= fh or x2 <= x or y2 <= y:
            logger.warning(
                f"[HpReader] region이 프레임 밖: "
                f"region=({x},{y},{w},{h}) frame=({fw},{fh})"
            )
            return self._cached_hp

        crop = frame[y:y2, x:x2]
        hp_pct = _calc_hp_pct(crop)

        # 급격한 변화만 로그 (5% 이상 변화 시)
        if abs(hp_pct - self._cached_hp) >= 5.0:
            logger.debug(
                f"[HpReader] HP 변화: {self._cached_hp:.1f}% → {hp_pct:.1f}%"
            )

        self._cached_hp = hp_pct
        return hp_pct

    def is_low(self, threshold_pct: Optional[float] = None) -> bool:
        """HP가 기준 이하인지 확인합니다.

        Args:
            threshold_pct: 기준값 (None이면 __init__에서 설정한 값 사용)

        Returns:
            True: HP < threshold_pct
        """
        threshold = threshold_pct if threshold_pct is not None else self.threshold_pct
        return self._cached_hp < threshold

    def get_cached(self) -> float:
        """마지막으로 읽은 HP% 값을 반환합니다."""
        return self._cached_hp
