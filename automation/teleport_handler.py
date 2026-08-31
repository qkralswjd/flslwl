"""텔레포트 처리 모듈.

텔레포트 키를 누르면 나오는 목적지 리스트 창에서
원하는 목적지 텍스트를 OCR로 찾아 클릭합니다.

비블로킹 구조
-------------
execute()는 내부적으로 단계(phase) 기반으로 동작합니다.
매 tick마다 호출하면 현재 단계가 준비됐을 때만 다음 단계로 진행하고,
대기 중이면 즉시 None을 반환합니다.

반환값:
    True  : 텔레포트 성공 (WAIT_COMPLETE 완료)
    False : 최대 재시도 소진 (실패)
    None  : 아직 진행 중 (다음 tick에 다시 호출)

단계 흐름:
    IDLE
      -> SEND_KEY          (키 입력 즉시)
      -> WAIT_WINDOW       (wait_after_key_ms 비블로킹 대기)
      -> FIND_DESTINATION  (OCR 탐색, 실패 시 WAIT_RETRY -> FIND_DESTINATION)
      -> CLICK_DESTINATION (클릭 즉시)
      -> WAIT_COMPLETE     (wait_after_click_ms 비블로킹 대기)
      -> done -> True

사용법:
    handler = TeleportHandler(
        key="F1",
        destination_region={"x":400,"y":200,"width":300,"height":400},
        destination_text="허수아비",
    )
    # tick 루프 안에서:
    result = handler.tick(frame_grabber, pico_worker)
    # result: True(성공) / False(실패) / None(진행중)

    # 또는 동기 래퍼 (기존 호환):
    success = handler.execute(frame_grabber, pico_worker)
"""

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("teleport_handler")

_ocr_reader = None


class _TPhase(Enum):
    """TeleportHandler 내부 단계."""
    IDLE             = auto()
    SEND_KEY         = auto()   # 키 입력 완료, 창 대기 시작
    WAIT_WINDOW      = auto()   # wait_after_key_ms 비블로킹 대기
    FIND_DESTINATION = auto()   # OCR 탐색 시도
    WAIT_RETRY       = auto()   # OCR 실패 후 0.5초 대기
    CLICK_DESTINATION= auto()   # 클릭 완료, 텔레포트 대기 시작
    WAIT_COMPLETE    = auto()   # wait_after_click_ms 비블로킹 대기
    SUCCESS          = auto()
    FAILURE          = auto()


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

        # 비블로킹 단계 상태
        self._phase: _TPhase = _TPhase.IDLE
        self._phase_until: float = 0.0   # 이 시각까지 현재 phase 유지
        self._attempt: int = 0           # OCR 탐색 시도 횟수

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

    def reset(self) -> None:
        """단계를 IDLE로 초기화합니다. 새 텔레포트 시작 전에 호출하세요."""
        self._phase       = _TPhase.IDLE
        self._phase_until = 0.0
        self._attempt     = 0

    def tick(
        self,
        frame_grabber: Callable[[], np.ndarray],
        pico_worker,
    ) -> Optional[bool]:
        """비블로킹 텔레포트 단계 진행기.

        매 tick마다 호출합니다. 대기 중이면 즉시 None 반환.

        Returns:
            True  : 텔레포트 성공
            False : 최대 재시도 소진 (실패)
            None  : 진행 중 (다음 tick에 다시 호출)
        """
        now = time.monotonic()

        # ── IDLE: 새 텔레포트 시작 ──────────────────────────────────────
        if self._phase == _TPhase.IDLE:
            logger.info(f"[TeleportHandler] 키({self.key}) 입력")
            pico_worker.key_tap_name(self.key, hold_ms=80)
            self._attempt     = 0
            self._phase       = _TPhase.WAIT_WINDOW
            self._phase_until = now + self.wait_after_key_ms / 1000.0
            return None

        # ── WAIT_WINDOW: 창 뜰 때까지 비블로킹 대기 ────────────────────
        if self._phase == _TPhase.WAIT_WINDOW:
            if now < self._phase_until:
                return None
            self._phase = _TPhase.FIND_DESTINATION
            return None

        # ── FIND_DESTINATION: OCR 탐색 ─────────────────────────────────
        if self._phase == _TPhase.FIND_DESTINATION:
            frame = frame_grabber()
            coord = self._find_destination(frame)

            if coord:
                sx, sy = coord
                logger.info(f"[TeleportHandler] 클릭: ({sx},{sy})")
                pico_worker.click(sx, sy)
                self._phase       = _TPhase.WAIT_COMPLETE
                self._phase_until = now + self.wait_after_click_ms / 1000.0
                return None

            self._attempt += 1
            if self._attempt >= self.max_retries:
                logger.error("[TeleportHandler] 텔레포트 실패 -- 목적지 미발견")
                self._phase = _TPhase.FAILURE
                return None

            logger.warning(
                f"[TeleportHandler] 목적지 탐색 실패 "
                f"({self._attempt}/{self.max_retries}), 0.5초 후 재시도"
            )
            self._phase       = _TPhase.WAIT_RETRY
            self._phase_until = now + 0.5
            return None

        # ── WAIT_RETRY: 재탐색 전 0.5초 비블로킹 대기 ──────────────────
        if self._phase == _TPhase.WAIT_RETRY:
            if now < self._phase_until:
                return None
            self._phase = _TPhase.FIND_DESTINATION
            return None

        # ── WAIT_COMPLETE: 텔레포트 완료 비블로킹 대기 ──────────────────
        if self._phase == _TPhase.WAIT_COMPLETE:
            if now < self._phase_until:
                return None
            logger.info("[TeleportHandler] 텔레포트 완료")
            self._phase = _TPhase.SUCCESS
            return None

        # ── 터미널 상태 ─────────────────────────────────────────────────
        if self._phase == _TPhase.SUCCESS:
            return True
        if self._phase == _TPhase.FAILURE:
            return False

        return None  # 안전망

    def execute(
        self,
        frame_grabber: Callable[[], np.ndarray],
        pico_worker,
    ) -> bool:
        """동기 래퍼 (기존 호환용).

        tick()을 성공/실패까지 반복 호출합니다.
        이 메서드는 여전히 블로킹이지만, tick() 자체는
        비블로킹이므로 HuntingStateMachine의 _update_use_scroll_dummy에서는
        tick()을 직접 사용하는 방식으로 전환했습니다.

        기존 코드가 execute()를 직접 부르는 경우를 위한 호환 래퍼.
        """
        self.reset()
        while True:
            result = self.tick(frame_grabber, pico_worker)
            if result is True:
                return True
            if result is False:
                return False
            # None: 대기 중 -- 짧게 슬립 후 재시도 (동기 래퍼 한정)
            time.sleep(0.01)
