"""아데나(전리품) 텍스트 탐지 모듈.

몬스터가 죽은 후 바닥에 나타나는 "아데나" 텍스트를
easyocr로 탐지하여 클릭 좌표를 반환합니다.

탐지 전략:
    1. HSV 색상 필터로 아데나 이름표 후보 영역(노란/흰 텍스트) 추출
    2. 컨투어로 텍스트 블럭 후보 박스 생성
    3. 각 후보 박스를 업스케일 후 OCR → 키워드 매칭
    → 전체 화면 OCR 대신 후보 영역만 OCR하므로 속도/정확도 모두 향상

사용법:
    detector = LootDetector(
        scan_region={"x": 0, "y": 0, "width": 1920, "height": 850},
        loot_keywords=["아데나", "Adena"],
    )
    loots = detector.find(frame)
    # loots = [(center_x, center_y, text, confidence), ...]
"""

import logging
import time
from typing import List, Optional, Tuple

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
            _ocr_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
            logger.info("[LootDetector] easyocr 초기화 완료")
        except ImportError:
            logger.error("[LootDetector] easyocr 미설치. pip install easyocr")
            raise
    return _ocr_reader


# ── 리니지 클래식 아데나 이름표 색상 범위 ────────────────────────────────
# 실제 픽셀 분석 결과 (adena_crop.png 실측):
#   - 흰 테두리: HSV S<50, V>200 (S_max=50으로 약간 채도 있는 흰색도 포함)
#   - 텍스트:    gray/silver (S<60, V=120~170) — 마우스 호버 전 기본 색상
#   - 배경:      dark (V≈50~80)
# 탐지 전략: 흰 테두리만 마스크 → dilation으로 이름표 전체 박스 연결
_ADENA_WHITE_LOWER = (0,   0,   200)   # 흰 테두리 HSV 하한
_ADENA_WHITE_UPPER = (180, 50,  255)   # 흰 테두리 HSV 상한
_ADENA_DILATION_K  = 7                 # dilation 커널 크기 (7×7)
_ADENA_DILATION_IT = 3                 # dilation 반복 횟수
_ADENA_BOX_MIN_W   = 40                # 최소 박스 너비 (px)
_ADENA_BOX_MIN_H   = 15                # 최소 박스 높이 (px)
_ADENA_BOX_MAX_W   = 300               # 최대 박스 너비 (너무 크면 노이즈)
_ADENA_BOX_MAX_H   = 80                # 최대 박스 높이


def _extract_candidate_boxes(
    crop_bgr: np.ndarray,
    padding: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """흰 테두리 HSV 마스크 + dilation으로 아데나 이름표 후보 박스 (x,y,w,h) 추출.

    전략:
    - 흰 테두리(HSV S<50, V>200)만 마스크로 추출
    - 7×7 kernel dilation × 3 → 얇은 테두리 선들을 하나의 덩어리로 연결
    - bounding rect로 이름표 전체 영역 박스 생성
    - w>30 & h>10 조건으로 작은 노이즈 제거

    실측 결과 (adena_crop.png):
    - 흰 테두리: x=25~159, y=26~61 (696픽셀)
    - dilation 후: box(x=16, y=17, w=153, h=54) — 이름표 전체 포함
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # 흰 테두리 마스크
    mask = cv2.inRange(hsv, _ADENA_WHITE_LOWER, _ADENA_WHITE_UPPER)

    # dilation: 얇은 테두리 선들을 연결 → 이름표 전체 영역 확장
    kernel = np.ones((_ADENA_DILATION_K, _ADENA_DILATION_K), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=_ADENA_DILATION_IT)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_crop, w_crop = crop_bgr.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 크기 필터 (너무 작거나 너무 큰 박스 제거)
        if w < _ADENA_BOX_MIN_W or h < _ADENA_BOX_MIN_H:
            continue
        if w > _ADENA_BOX_MAX_W or h > _ADENA_BOX_MAX_H:
            continue
        # 패딩 추가 (글자가 잘리지 않도록)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_crop, x + w + padding)
        y2 = min(h_crop, y + h + padding)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes




def _preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """후보 패치를 OCR에 최적화된 형태로 전처리.

    실측 최적 파라미터 (adena_crop.png 기준):
    - 4배 업스케일 (INTER_CUBIC)
    - 그레이스케일 변환
    - 고정 임계값 135 역방향 이진화
      * 배경(V≈50~100)   → 255(흰) — 배경 제거
      * 글씨(V≈130~165)  →   0(검) — 회색/은색 텍스트 보존
      * 흰 테두리(V≥200) →   0(검) — 테두리도 검게
    - 3채널 변환 (easyocr/tesseract 입력 호환)

    검증 결과 (tesseract psm=6, kor+eng):
    - thresh=130 → '1마데나. [291]:' ✅
    - thresh=135 → '마데나 123 1).'  ✅  ← 채택 (소음 최소)
    - thresh=140 → '마데나 [29 ㅣ'   ✅
    """
    h, w = patch.shape[:2]

    # 4배 업스케일
    SCALE = 4
    patch = cv2.resize(patch, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # 고정 임계값 역방향 이진화
    # THRESH_BINARY_INV: 픽셀 > thresh → 0(검), ≤ thresh → 255(흰)
    _, thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)

    # 흑백 3채널로 변환 (easyocr/tesseract 입력 호환)
    thresh_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return thresh_3ch


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
        # "데나" 추가: OCR 오인식 시 부분 매칭 허용 ("마데나", "1마데나" 등)
        self.loot_keywords  = loot_keywords or ["아데나", "데나", "Adena", "adena", "ADENA"]
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

    def find(self, frame: np.ndarray) -> List[Tuple[int, int, str, float]]:
        """프레임에서 아데나 이름표를 HSV 박스 탐지로 찾습니다.

        전략 (OCR 제거 — HSV만으로 즉시 반환):
            1. scan_region 크롭
            2. HSV 흰 테두리 마스크 + dilation → 후보 박스 추출
            3. 크기 필터로 오탐 제거
            4. 박스 중심을 화면 절대좌표로 변환 → 즉시 반환

        Returns:
            [(screen_x, screen_y, "adena", 1.0), ...]
        """
        now = time.time()
        if now - self._last_scan_time < self.scan_interval:
            return self._cached_loots

        self._last_scan_time = now

        # ── 스캔 영역 크롭 ───────────────────────────────────────────
        rx = self.scan_region.get("x", 0)
        ry = self.scan_region.get("y", 0)
        rw = self.scan_region.get("width",  frame.shape[1])
        rh = self.scan_region.get("height", frame.shape[0])
        rw = min(rw, frame.shape[1] - rx)
        rh = min(rh, frame.shape[0] - ry)
        crop = frame[ry:ry + rh, rx:rx + rw]
        if crop.size == 0:
            return self._cached_loots

        # ── HSV 흰 테두리 박스 탐지 ─────────────────────────────────
        boxes = _extract_candidate_boxes(crop)

        if not boxes:
            logger.debug("[LootDetector] 후보 박스 없음")
            self._cached_loots = []
            return self._cached_loots

        # ── 박스 중심 → 화면 절대좌표 변환 ─────────────────────────
        loots = []
        for (bx, by, bw, bh) in boxes:
            screen_x = (bx + bw // 2 + rx
                        + self.roi_offset[0]
                        + self.capture_offset[0])
            screen_y = (by + bh // 2 + ry
                        + self.roi_offset[1]
                        + self.capture_offset[1])
            loots.append((screen_x, screen_y, "adena", 1.0))
            logger.info(
                f"[LootDetector] ✅ HSV탐지: at ({screen_x},{screen_y}) "
                f"box={bw}×{bh}"
            )

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


    def invalidate(self) -> None:
        """캐시를 무효화합니다. 다음 find() 호출 시 즉시 재스캔합니다.
        
        SCAN 상태 진입 시 호출하면 이전 캐시로 인한 오탐/미탐을 방지합니다.
        """
        self._last_scan_time = 0.0
        self._cached_loots   = []
        logger.debug("[LootDetector] 캐시 무효화")
