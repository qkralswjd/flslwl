"""텔레포트 처리 모듈.

텔레포트 키를 누르면 나오는 목적지 리스트 창에서
원하는 목적지 텍스트를 OCR로 찾아 클릭합니다.

사용법:
    handler = TeleportHandler(
        key="F1",
        destination_region={"x":400,"y":200,"width":300,"height":400},
        destination_text="허수아비",
    )
    success = handler.execute(frame_grabber, pico_worker)
"""

import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("teleport_handler")

_ocr_reader = None


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("[TeleportHandler] easyocr 초기화 중...")
            _ocr_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
            logger.info("[TeleportHandler] easyocr 초기화 완료")
        except ImportError:
            logger.error("[TeleportHandler] easyocr 미설치.")
            raise
    return _ocr_reader


class TeleportHandler:
    """텔레포트 키 입력 → 목적지 창 OCR → 목적지 클릭."""

    def __init__(
        self,
        key: str,
        destination_region: dict,
        destination_text: str,
        capture_offset: tuple[int, int] = (0, 0),
        wait_after_key_ms: int   = 800,
        wait_after_click_ms: int = 2000,
        max_retries: int         = 3,
        min_confidence: float    = 0.3,
    ):
        """
        Args:
            key: 텔레포트 키 이름 (예: "F1", "T")
            destination_region: 목적지 창이 뜨는 영역 {"x","y","width","height"}
                                (캡처 프레임 기준 좌표)
            destination_text: 클릭할 목적지 텍스트 (예: "허수아비")
            capture_offset: 캡처 영역의 절대 화면 오프셋 (x, y)
            wait_after_key_ms: 키 입력 후 창이 뜰 때까지 대기 (ms)
            wait_after_click_ms: 클릭 후 텔레포트 완료 대기 (ms)
            max_retries: 목적지 탐색 재시도 횟수
            min_confidence: OCR 최소 신뢰도
        """
        self.key                 = key.upper()
        self.destination_region  = destination_region
        self.destination_text    = destination_text
        self.capture_offset      = capture_offset
        self.wait_after_key_ms   = wait_after_key_ms
        self.wait_after_click_ms = wait_after_click_ms
        self.max_retries         = max_retries
        self.min_confidence      = min_confidence
        self._ocr                = None

    def _ensure_ocr(self):
        if self._ocr is None:
            self._ocr = _get_ocr()

    def _find_destination(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        """목적지 텍스트를 찾아 절대 화면 좌표를 반환합니다."""
        rx = self.destination_region.get("x", 0)
        ry = self.destination_region.get("y", 0)
        rw = self.destination_region.get("width", 300)
        rh = self.destination_region.get("height", 400)

        crop = frame[ry:ry + rh, rx:rx + rw]
        if crop.size == 0:
            return None

        # 전처리: 2배 확대 + 그레이스케일
        h, w = crop.shape[:2]
        enlarged = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

        try:
            self._ensure_ocr()
            results = self._ocr.readtext(gray, detail=1, paragraph=False)
        except Exception as e:
            logger.error(f"[TeleportHandler] OCR 오류: {e}")
            return None

        target = self.destination_text.lower().strip()
        best_match = None
        best_conf  = 0.0

        for (bbox, text, confidence) in results:
            if confidence < self.min_confidence:
                continue
            if target in text.lower().strip():
                if confidence > best_conf:
                    best_conf  = confidence
                    best_match = (bbox, text)

        if best_match is None:
            logger.debug(
                f"[TeleportHandler] '{self.destination_text}' 탐색 실패. "
                f"OCR 결과: {[(t, f'{c:.2f}') for _, t, c in results[:5]]}"
            )
            return None

        bbox = best_match[0]
        pts  = np.array(bbox)

        # 2배 확대했으므로 좌표를 원래 스케일로 되돌림
        cx_crop = int(pts[:, 0].mean() / 2)
        cy_crop = int(pts[:, 1].mean() / 2)

        # 절대 화면 좌표
        screen_x = cx_crop + rx + self.capture_offset[0]
        screen_y = cy_crop + ry + self.capture_offset[1]

        logger.info(
            f"[TeleportHandler] '{self.destination_text}' 발견 "
            f"at ({screen_x},{screen_y}) conf={best_conf:.2f}"
        )
        return screen_x, screen_y

    def execute(
        self,
        frame_grabber: Callable[[], np.ndarray],
        pico_worker,
    ) -> bool:
        """텔레포트를 실행합니다.

        Args:
            frame_grabber: 현재 화면 프레임을 반환하는 함수 (capturer.grab)
            pico_worker: PicoSerialWorker 인스턴스

        Returns:
            True: 성공, False: 실패
        """
        logger.info(f"[TeleportHandler] 텔레포트 키({self.key}) 입력")

        # 1. 텔레포트 키 입력
        pico_worker.key_tap_name(self.key, hold_ms=80)
        time.sleep(self.wait_after_key_ms / 1000.0)

        # 2. 목적지 창에서 텍스트 탐색 (재시도)
        for attempt in range(self.max_retries):
            frame = frame_grabber()
            coord = self._find_destination(frame)

            if coord:
                sx, sy = coord
                logger.info(f"[TeleportHandler] 클릭: ({sx},{sy})")
                pico_worker.click(sx, sy)
                time.sleep(self.wait_after_click_ms / 1000.0)
                return True

            logger.warning(
                f"[TeleportHandler] 목적지 탐색 실패 "
                f"({attempt + 1}/{self.max_retries}), 0.5초 후 재시도"
            )
            time.sleep(0.5)

        logger.error("[TeleportHandler] 텔레포트 실패 — 목적지를 찾지 못함")
        return False
