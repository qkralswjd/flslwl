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
# 실제 픽셀 분석 결과 (adena_sample.png 기준):
#   - 텍스트:  gray/silver  (HSV S≈0, V≈165)
#   - 테두리:  흰색          (HSV S<30, V>220)
# → 무채색 영역(흰+회색) 을 후보 박스로 사용
_ADENA_COLOR_RANGES = [
    # 흰색 테두리 (실측: S<30, V>220)
    {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 40, "v_min": 200},
    # 회색/은색 텍스트 (실측: S<50, V=140~210)
    {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 50, "v_min": 140},
]


def _extract_candidate_boxes(
    crop_bgr: np.ndarray,
    min_area: int = 30,
    max_area: int = 8000,
    padding: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """HSV 색상 필터 + 컨투어로 아데나 이름표 후보 박스 (x,y,w,h) 추출."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for rng in _ADENA_COLOR_RANGES:
        s_max = rng.get("s_max", 255)
        mask = cv2.inRange(
            hsv,
            (rng["h_min"], rng["s_min"], rng["v_min"]),
            (rng["h_max"], s_max,        255),
        )
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 노이즈 제거 + 글자 연결
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_crop, w_crop = crop_bgr.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 패딩 추가 (글자가 잘리지 않도록)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_crop, x + w + padding)
        y2 = min(h_crop, y + h + padding)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    # 인접 박스 병합 (같은 이름표 글자들을 하나로)
    boxes = _merge_nearby_boxes(boxes, gap=12)
    return boxes


def _merge_nearby_boxes(
    boxes: List[Tuple[int, int, int, int]],
    gap: int = 12,
) -> List[Tuple[int, int, int, int]]:
    """가까운 박스들을 하나로 병합."""
    if not boxes:
        return []

    merged = True
    result = list(boxes)
    while merged:
        merged = False
        new_result = []
        used = [False] * len(result)
        for i, (x1, y1, w1, h1) in enumerate(result):
            if used[i]:
                continue
            bx1, by1, bx2, by2 = x1, y1, x1 + w1, y1 + h1
            for j, (x2, y2, w2, h2) in enumerate(result):
                if i == j or used[j]:
                    continue
                cx1, cy1, cx2, cy2 = x2, y2, x2 + w2, y2 + h2
                # 겹치거나 gap 이내이면 병합
                if (bx1 - gap <= cx2 and bx2 + gap >= cx1 and
                        by1 - gap <= cy2 and by2 + gap >= cy1):
                    bx1 = min(bx1, cx1)
                    by1 = min(by1, cy1)
                    bx2 = max(bx2, cx2)
                    by2 = max(by2, cy2)
                    used[j] = True
                    merged = True
            new_result.append((bx1, by1, bx2 - bx1, by2 - by1))
            used[i] = True
        result = new_result
    return result


def _preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """후보 패치를 OCR에 최적화된 형태로 전처리.

    실측 최적 파라미터 (adena_sample.png 기준):
    - 3배 업스케일
    - 그레이스케일
    - 임계값 150 이진화 역방향 (흰/회색 글씨→검은 글씨)
    → tesseract psm6 kor+eng 에서 "마데나" 수준 인식
    """
    h, w = patch.shape[:2]

    # 3배 강제 업스케일 (실측: scale=3 최적)
    SCALE = 3
    patch = cv2.resize(patch, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # 고정 임계값 이진화: 회색 글씨(V≈165)를 검게, 흰 배경(V≥190)을 희게
    # THRESH_BINARY_INV: 픽셀 > thresh → 0(검), ≤ thresh → 255(흰)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

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
        """프레임에서 아이템 텍스트를 찾습니다.

        전략:
            1. scan_region 크롭
            2. HSV 색상 필터로 아데나 이름표 후보 박스 추출
            3. 후보 박스마다 업스케일+OTSU 전처리 후 OCR
            4. 키워드 매칭 → 절대 좌표 반환

        Returns:
            [(screen_x, screen_y, text, confidence), ...]
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
        # 프레임 경계 초과 방지
        rw = min(rw, frame.shape[1] - rx)
        rh = min(rh, frame.shape[0] - ry)
        crop = frame[ry:ry + rh, rx:rx + rw]
        if crop.size == 0:
            return self._cached_loots

        # ── HSV 색상 필터로 후보 박스 추출 ─────────────────────────
        boxes = _extract_candidate_boxes(crop)

        if not boxes:
            logger.debug("[LootDetector] 후보 박스 없음 (아데나 색상 미탐지)")
            self._cached_loots = []
            return self._cached_loots

        logger.debug(f"[LootDetector] 후보 박스 {len(boxes)}개 → OCR 시작")

        # ── 후보 박스별 OCR ─────────────────────────────────────────
        loots = []
        try:
            self._ensure_ocr()
            for (bx, by, bw, bh) in boxes:
                patch = crop[by:by + bh, bx:bx + bw]
                if patch.size == 0:
                    continue

                processed = _preprocess_patch(patch)

                results = self._ocr.readtext(
                    processed,
                    detail=1,
                    paragraph=False,
                    allowlist=None,
                )

                for (bbox, text, confidence) in results:
                    if confidence < self.min_confidence:
                        continue
                    if not self._is_loot_text(text):
                        continue

                    # 박스 중심을 절대 좌표로 변환
                    screen_x = (bx + bw // 2 + rx
                                + self.roi_offset[0]
                                + self.capture_offset[0])
                    screen_y = (by + bh // 2 + ry
                                + self.roi_offset[1]
                                + self.capture_offset[1])

                    loots.append((screen_x, screen_y, text.strip(), confidence))
                    logger.info(
                        f"[LootDetector] ✅ 발견: '{text}' "
                        f"at ({screen_x},{screen_y}) conf={confidence:.2f}"
                    )

        except Exception as e:
            logger.error(f"[LootDetector] OCR 오류: {e}")
            return self._cached_loots

        if not loots:
            logger.debug(f"[LootDetector] 후보 {len(boxes)}개 OCR → 아데나 키워드 없음")

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
