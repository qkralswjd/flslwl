"""아데나(전리품) 텍스트 탐지 모듈.

몬스터가 죽은 후 바닥에 나타나는 "아데나" 텍스트를
easyocr로 탐지하여 클릭 좌표를 반환합니다.

사용법:
    detector = LootDetector(
        scan_region={"x": 0, "y": 0, "width": 1440, "height": 780},
        loot_keywords=["아데나", "Adena"],
    )
    loots = detector.find(frame)
    # loots = [(center_x, center_y, text, confidence), ...]
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("loot_detector")

_ocr_reader = None


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("[LootDetector] easyocr 초기화 중...")
            # 한국어 + 영어 동시 인식 (아데나 = 한글)
            _ocr_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
            logger.info("[LootDetector] easyocr 초기화 완료")
        except ImportError:
            logger.error("[LootDetector] easyocr 미설치. pip install easyocr")
            raise
    return _ocr_reader


def _preprocess_for_loot(crop: np.ndarray) -> np.ndarray:
    """아데나 텍스트 인식을 위한 전처리.
    
    리니지 클래식의 아이템 이름표는 보통 흰색/노란색 텍스트.
    """
    # 밝은 픽셀만 강조 (아이템 이름은 밝은 색)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 밝은 텍스트 추출
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 약간 팽창시켜 글자 연결
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(bright, kernel, iterations=1)

    return dilated


class LootDetector:
    """화면에서 아이템 이름(아데나 등) 텍스트를 탐지합니다."""

    def __init__(
        self,
        scan_region: dict,
        loot_keywords: list[str] = None,
        scan_interval_s: float  = 0.5,
        min_confidence: float   = 0.4,
        roi_offset: tuple       = (0, 0),
        capture_offset: tuple   = (0, 0),
    ):
        """
        Args:
            scan_region: 스캔할 ROI 영역 {"x","y","width","height"}
            loot_keywords: 탐지할 텍스트 목록 (기본: ["아데나","Adena","adena"])
            scan_interval_s: 스캔 최소 간격 (초)
            min_confidence: OCR 최소 신뢰도
            roi_offset: ROI 오프셋 (ox, oy)
            capture_offset: 캡처 영역 오프셋 (ox, oy)
        """
        self.scan_region    = scan_region
        self.loot_keywords  = loot_keywords or ["아데나", "Adena", "adena", "ADENA"]
        self.scan_interval  = scan_interval_s
        self.min_confidence = min_confidence
        self.roi_offset     = roi_offset
        self.capture_offset = capture_offset

        self._last_scan_time = 0.0
        self._cached_loots: list = []
        self._ocr = None

    def _ensure_ocr(self):
        if self._ocr is None:
            self._ocr = _get_ocr()

    def _is_loot_text(self, text: str) -> bool:
        """탐지된 텍스트가 아이템 이름인지 확인합니다."""
        text_lower = text.lower().strip()
        for kw in self.loot_keywords:
            if kw.lower() in text_lower:
                return True
        return False

    def find(self, frame: np.ndarray) -> list[tuple[int, int, str, float]]:
        """프레임에서 아이템 텍스트를 찾습니다.

        Returns:
            [(screen_x, screen_y, text, confidence), ...]
            screen 좌표는 절대 화면 좌표 (Pico 클릭에 바로 사용 가능)
        """
        now = time.time()
        if now - self._last_scan_time < self.scan_interval:
            return self._cached_loots

        self._last_scan_time = now

        # 스캔 영역 크롭
        rx = self.scan_region.get("x", 0)
        ry = self.scan_region.get("y", 0)
        rw = self.scan_region.get("width", frame.shape[1])
        rh = self.scan_region.get("height", frame.shape[0])
        crop = frame[ry:ry + rh, rx:rx + rw]

        if crop.size == 0:
            return self._cached_loots

        # 전처리
        processed = _preprocess_for_loot(crop)

        # OCR
        try:
            self._ensure_ocr()
            results = self._ocr.readtext(processed, detail=1, paragraph=False)
        except Exception as e:
            logger.error(f"[LootDetector] OCR 오류: {e}")
            return self._cached_loots

        loots = []
        for (bbox, text, confidence) in results:
            if confidence < self.min_confidence:
                continue
            if not self._is_loot_text(text):
                continue

            # bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            # 중심 좌표 계산 (crop 내 좌표)
            pts = np.array(bbox)
            cx_crop = int(pts[:, 0].mean())
            cy_crop = int(pts[:, 1].mean())

            # 절대 화면 좌표로 변환
            # crop 내 좌표 → scan_region 내 좌표 → 캡처 내 좌표 → 절대 좌표
            screen_x = (cx_crop + rx
                        + self.roi_offset[0]
                        + self.capture_offset[0])
            screen_y = (cy_crop + ry
                        + self.roi_offset[1]
                        + self.capture_offset[1])

            loots.append((screen_x, screen_y, text.strip(), confidence))
            logger.info(f"[LootDetector] 발견: '{text}' at ({screen_x},{screen_y}) conf={confidence:.2f}")

        self._cached_loots = loots
        return loots

    def find_nearest(
        self,
        frame: np.ndarray,
        ref_x: int = 0,
        ref_y: int = 0,
    ) -> Optional[tuple[int, int, str, float]]:
        """가장 가까운 아이템을 반환합니다.

        Args:
            ref_x, ref_y: 기준 좌표 (캐릭터 위치 등)
        """
        loots = self.find(frame)
        if not loots:
            return None

        import math
        return min(loots, key=lambda l: math.hypot(l[0] - ref_x, l[1] - ref_y))
