"""
perception/level.py — 레벨 OCR (easyocr, 비동기 스레드).

기존 flslwl/automation/level_reader.py 에서 개선:
  - min_confidence 기본값  0.1 → 0.4 (오인식 감소)
  - easyocr Reader 초기화를 별도 스레드에서 수행 (UI 블로킹 방지)
  - from_settings() 팩토리 추가
  - get_cached() / invalidate() API 통일
  - 타입 힌트 완비
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

Region = Union[dict, Tuple[int, int, int, int]]

# easyocr 는 설치 환경에 따라 없을 수 있으므로 런타임에 lazy import
_easyocr_lock = threading.Lock()
_easyocr_module = None  # 초기화 성공 시 easyocr 모듈 자체를 저장


def _load_easyocr():
    global _easyocr_module
    with _easyocr_lock:
        if _easyocr_module is None:
            try:
                import easyocr as _m  # noqa: F401
                _easyocr_module = _m
                log.debug("easyocr 모듈 로드 완료")
            except ImportError:
                log.warning("easyocr 가 설치되지 않아 LevelReader 는 None 만 반환합니다.")
    return _easyocr_module


def _region_to_xywh(region: Region) -> Tuple[int, int, int, int]:
    if isinstance(region, dict):
        return region["x"], region["y"], region["w"], region["h"]
    return tuple(region[:4])  # type: ignore[return-value]


# ─────────────────────────── 파싱 헬퍼 ──────────────────────────
_LEVEL_PATTERNS = [
    re.compile(r"[Ll][Ee][Vv][Ee]?[Ll]?\.?\s*(\d{1,2})"),  # LEV 42, Lv. 5
    re.compile(r"[Ll][Vv]\.?\s*(\d{1,2})"),                  # Lv.42, lv 5
    re.compile(r"\b(\d{1,2})\b"),                             # 숫자만
]


def _parse_level(text: str) -> Optional[int]:
    """OCR 결과 문자열에서 레벨 숫자(1~99)를 추출한다."""
    text = text.strip()
    for pat in _LEVEL_PATTERNS:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 99:
                return val
    return None


class LevelReader:
    """
    화면 ROI 에서 레벨 숫자를 OCR 로 읽어 반환한다.

    easyocr Reader 초기화(최초 1회, 수 초 소요)는 백그라운드 스레드에서
    수행하므로 생성 즉시 UI 를 블로킹하지 않는다.
    초기화가 완료되기 전에 read() 를 호출하면 None 을 반환한다.

    Parameters
    ----------
    region : dict | tuple
        레벨 텍스트가 표시되는 영역.
    monitor_offset : (int, int)
        전체 화면 좌표 → ROI 좌표 오프셋 (기존 호환용).
    read_interval_s : float
        OCR 재실행 최소 간격(초). 기본 2.0.
    min_confidence : float
        easyocr confidence 필터 하한. 기본 0.4 (기존 0.1 → 개선).
    gpu : bool
        easyocr GPU 사용 여부. 기본 False.
    """

    def __init__(
        self,
        region: Region,
        monitor_offset: Tuple[int, int] = (0, 0),
        read_interval_s: float = 2.0,
        min_confidence: float = 0.4,
        gpu: bool = False,
    ) -> None:
        self._rx, self._ry, self._rw, self._rh = _region_to_xywh(region)
        self._monitor_offset = monitor_offset
        self.read_interval_s = read_interval_s
        self.min_confidence = min_confidence
        self._gpu = gpu

        self._reader = None          # easyocr.Reader — 초기화 후 설정
        self._reader_ready = False
        self._init_error: Optional[Exception] = None

        self._cached_level: Optional[int] = None
        self._last_read_at: float = 0.0

        # 백그라운드 초기화
        t = threading.Thread(target=self._init_reader, daemon=True, name="LevelReaderInit")
        t.start()

    # ──────────────────────────── 팩토리 ────────────────────────────
    @classmethod
    def from_settings(cls, settings) -> "LevelReader":
        """
        Settings.level_ocr 와 Settings.capture 로부터 LevelReader 를 생성한다.

        settings.level_ocr 필드:
            region_x, region_y, region_w, region_h
            min_confidence  (Optional, 기본 0.4)
            read_interval_s (Optional, 기본 2.0)
            gpu             (Optional, 기본 False)
        settings.capture 필드:
            region_x, region_y — 전체 캡처 영역 오프셋
        """
        lo = settings.level_ocr
        cap = settings.capture
        region = (lo.region_x, lo.region_y, lo.region_w, lo.region_h)
        monitor_offset = (
            getattr(cap, "region_x", 0),
            getattr(cap, "region_y", 0),
        )
        return cls(
            region=region,
            monitor_offset=monitor_offset,
            read_interval_s=getattr(lo, "read_interval_s", 2.0),
            min_confidence=getattr(lo, "min_confidence", 0.4),
            gpu=getattr(lo, "gpu", False),
        )

    # ──────────────────────────── 공개 API ──────────────────────────
    @property
    def is_ready(self) -> bool:
        """easyocr Reader 초기화가 완료되었으면 True."""
        return self._reader_ready

    def read(self, frame: np.ndarray) -> Optional[int]:
        """
        BGR 프레임에서 레벨 텍스트를 OCR 하여 1~99 범위 정수를 반환한다.

        초기화 미완료, OCR 실패, 파싱 실패 시 None 반환.
        read_interval_s 이내 중복 호출은 캐시를 반환한다.
        """
        if not self._reader_ready:
            return self._cached_level

        now = time.monotonic()
        if (
            self._cached_level is not None
            and now - self._last_read_at < self.read_interval_s
        ):
            return self._cached_level

        level = self._ocr(frame)
        if level is not None:
            self._cached_level = level
            self._last_read_at = now
        return self._cached_level

    def get_cached(self) -> Optional[int]:
        """마지막으로 읽은 레벨. read() 미호출 시 None."""
        return self._cached_level

    def invalidate(self) -> None:
        """캐시를 강제 무효화."""
        self._cached_level = None
        self._last_read_at = 0.0

    # ──────────────────────────── 내부 ──────────────────────────────
    def _init_reader(self) -> None:
        """백그라운드 스레드: easyocr Reader 초기화."""
        try:
            mod = _load_easyocr()
            if mod is None:
                return
            self._reader = mod.Reader(["en"], gpu=self._gpu, verbose=False)
            self._reader_ready = True
            log.debug("LevelReader: easyocr Reader 초기화 완료")
        except Exception as e:
            self._init_error = e
            log.error("LevelReader: easyocr 초기화 실패 — %s", e)

    def _ocr(self, frame: np.ndarray) -> Optional[int]:
        """ROI 를 잘라 easyocr 로 읽고 레벨 숫자를 반환한다."""
        if self._reader is None:
            return None

        h, w = frame.shape[:2]
        ox, oy = self._monitor_offset
        x1 = max(0, self._rx - ox)
        y1 = max(0, self._ry - oy)
        x2 = min(w, x1 + self._rw)
        y2 = min(h, y1 + self._rh)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        try:
            results = self._reader.readtext(roi, detail=1)
        except Exception as e:
            log.debug("LevelReader OCR 오류: %s", e)
            return None

        best: Optional[int] = None
        best_conf = -1.0
        for _bbox, text, conf in results:
            if conf < self.min_confidence:
                continue
            parsed = _parse_level(text)
            if parsed is not None and conf > best_conf:
                best = parsed
                best_conf = conf

        return best
