"""레벨 OCR 인식 모듈.

화면의 지정 영역에서 "Lv.숫자" 형식의 레벨을 읽어옵니다.
easyocr을 사용하여 정확도를 높입니다.

사용법:
    reader = LevelReader(region={"x": 10, "y": 10, "width": 80, "height": 30})
    level = reader.read(frame)   # 현재 레벨 int, 실패 시 None
"""

import logging
import re
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("level_reader")

# easyocr은 최초 1회만 초기화 (무거움)
_ocr_reader = None


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("[LevelReader] easyocr 초기화 중...")
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("[LevelReader] easyocr 초기화 완료")
        except ImportError:
            logger.error("[LevelReader] easyocr 미설치. pip install easyocr")
            raise
    return _ocr_reader


def _preprocess(crop: np.ndarray) -> np.ndarray:
    """OCR 인식률을 높이기 위한 전처리."""
    # 2배 확대 (작은 텍스트 인식률 향상)
    h, w = crop.shape[:2]
    enlarged = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    # 그레이스케일
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # 이진화 (밝은 배경에 어두운 텍스트 or 반대)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _parse_level(text: str) -> Optional[int]:
    """OCR 텍스트에서 레벨 숫자를 추출합니다.

    인식 가능 형식:
        "Lv.5", "Lv 5", "LV.5", "lv5", "5"
    """
    text = text.strip().replace(" ", "")

    # Lv.숫자 형식
    m = re.search(r"[Ll][Vv]\.?(\d+)", text)
    if m:
        return int(m.group(1))

    # 숫자만 있는 경우 (region이 레벨 숫자만 보이는 곳)
    m = re.search(r"^\d+$", text)
    if m:
        val = int(m.group(0))
        if 1 <= val <= 99:
            return val

    return None


class LevelReader:
    """화면 지정 영역에서 현재 레벨을 OCR로 읽습니다."""

    def __init__(
        self,
        region: dict,
        monitor_offset: tuple[int, int] = (0, 0),
        read_interval_s: float = 2.0,
    ):
        """
        Args:
            region: {"x", "y", "width", "height"} — 캡처 프레임 내 좌표
            monitor_offset: (offset_x, offset_y) — 모니터 오프셋 (미사용 시 (0,0))
            read_interval_s: OCR 호출 최소 간격 (초) — 너무 자주 호출 방지
        """
        self.region         = region
        self.monitor_offset = monitor_offset
        self.read_interval  = read_interval_s

        self._last_read_time  = 0.0
        self._cached_level: Optional[int] = None
        self._ocr            = None   # lazy init

    def _ensure_ocr(self):
        if self._ocr is None:
            self._ocr = _get_ocr()

    def read(self, frame: np.ndarray) -> Optional[int]:
        """프레임에서 레벨을 읽어 반환합니다.

        read_interval 보다 짧은 간격으로 호출되면 캐시 값을 반환합니다.

        Returns:
            int: 현재 레벨 (1~99)
            None: 인식 실패
        """
        now = time.time()
        if now - self._last_read_time < self.read_interval:
            return self._cached_level

        self._last_read_time = now

        # region 크롭
        x  = self.region.get("x", 0)
        y  = self.region.get("y", 0)
        w  = self.region.get("width", 100)
        h  = self.region.get("height", 30)
        crop = frame[y:y + h, x:x + w]

        if crop.size == 0:
            logger.warning("[LevelReader] region이 프레임 밖입니다")
            return self._cached_level

        # 전처리
        processed = _preprocess(crop)

        # OCR
        try:
            self._ensure_ocr()
            results = self._ocr.readtext(processed, detail=1, paragraph=False)
        except Exception as e:
            logger.error(f"[LevelReader] OCR 오류: {e}")
            return self._cached_level

        # 결과 파싱
        for (_, text, confidence) in results:
            if confidence < 0.3:
                continue
            level = _parse_level(text)
            if level is not None:
                if level != self._cached_level:
                    logger.info(f"[LevelReader] 레벨 감지: {self._cached_level} → {level} (conf={confidence:.2f})")
                self._cached_level = level
                return level

        logger.debug(f"[LevelReader] 레벨 인식 실패. OCR 결과: {[(t, f'{c:.2f}') for _, t, c in results]}")
        return self._cached_level

    def get_cached(self) -> Optional[int]:
        """마지막으로 성공한 레벨 값을 반환합니다."""
        return self._cached_level
