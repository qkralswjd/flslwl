"""
tests/test_perception.py — perception/ 레이어 단위 테스트.

외부 의존:
  - cv2, numpy: 설치 필요
  - easyocr: 없으면 LevelReader 관련 테스트는 skip
  - perception.hp, perception.loot: cv2/numpy 만 있으면 OK
"""
from __future__ import annotations

import sys
import time
import types
import unittest
from dataclasses import dataclass, field
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

# ─────────────────────────── 헬퍼 ───────────────────────────────
def _solid_bgr(h: int, w: int, bgr: Tuple[int, int, int]) -> np.ndarray:
    """단색 BGR 프레임 생성."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def _make_red_bar_frame(bar_w: int = 100, bar_h: int = 20, fill: float = 0.6) -> np.ndarray:
    """HP 바 시뮬레이션: 가로 100픽셀, fill 비율만큼 빨간색 채움."""
    frame = _solid_bgr(bar_h, bar_w, (0, 0, 0))  # 검정 배경
    red_cols = int(bar_w * fill)
    # 순수 빨간(BGR: 0,0,255) → HSV 에서 Hue≈0, S=255, V=255 → 범위 내
    frame[:, :red_cols] = (0, 0, 255)
    return frame


# ─────────────────────────── Settings 더미 ───────────────────────
@dataclass
class _HpBarSettings:
    region_x: int = 0
    region_y: int = 0
    region_w: int = 100
    region_h: int = 20
    threshold_pct: float = 50.0
    read_interval_s: float = 0.5


@dataclass
class _LevelOcrSettings:
    region_x: int = 0
    region_y: int = 0
    region_w: int = 120
    region_h: int = 30
    min_confidence: float = 0.4
    read_interval_s: float = 2.0
    gpu: bool = False


@dataclass
class _CaptureSettings:
    region_x: int = 0
    region_y: int = 0


@dataclass
class _LootSettings:
    region_x: int = 0
    region_y: int = 0
    region_w: int = 0
    region_h: int = 0
    scan_interval_s: float = 0.5
    min_area: int = 200
    max_area: int = 60000


@dataclass
class _Settings:
    hp_bar: _HpBarSettings = field(default_factory=_HpBarSettings)
    level_ocr: _LevelOcrSettings = field(default_factory=_LevelOcrSettings)
    capture: _CaptureSettings = field(default_factory=_CaptureSettings)
    loot: _LootSettings = field(default_factory=_LootSettings)


# ════════════════════════════════════════════════════════════════
#  HpReader 테스트
# ════════════════════════════════════════════════════════════════
class TestHpReader(unittest.TestCase):
    def setUp(self):
        from perception.hp import HpReader
        self.HpReader = HpReader

    def _reader(self, **kw):
        region = kw.pop("region", (0, 0, 100, 20))
        return self.HpReader(region=region, **kw)

    # ── 기본 동작 ────────────────────────────────────────────────
    def test_full_red_bar_returns_100(self):
        frame = _make_red_bar_frame(fill=1.0)
        r = self._reader()
        pct = r.read(frame)
        self.assertGreaterEqual(pct, 95.0, "전체 빨간 → HP ≈ 100%")

    def test_half_red_bar_approx_50(self):
        frame = _make_red_bar_frame(fill=0.5)
        r = self._reader()
        pct = r.read(frame)
        self.assertGreater(pct, 40.0)
        self.assertLess(pct, 60.0)

    def test_black_bar_returns_0(self):
        frame = _solid_bgr(20, 100, (0, 0, 0))
        r = self._reader()
        pct = r.read(frame)
        self.assertEqual(pct, 0.0)

    def test_return_type_is_float(self):
        frame = _make_red_bar_frame(fill=0.7)
        r = self._reader()
        self.assertIsInstance(r.read(frame), float)

    def test_result_range_0_100(self):
        for fill in [0.0, 0.25, 0.5, 0.75, 1.0]:
            frame = _make_red_bar_frame(fill=fill)
            r = self._reader()
            pct = r.read(frame)
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 100.0)

    # ── 캐시 ─────────────────────────────────────────────────────
    def test_cache_returns_same_value_within_interval(self):
        frame = _make_red_bar_frame(fill=0.8)
        r = self._reader(read_interval_s=10.0)
        v1 = r.read(frame)
        # 프레임을 검정으로 바꿔도 캐시 기간 내라 같은 값 반환
        black = _solid_bgr(20, 100, (0, 0, 0))
        v2 = r.read(black)
        self.assertEqual(v1, v2)

    def test_cache_expires_after_interval(self):
        frame = _make_red_bar_frame(fill=0.8)
        r = self._reader(read_interval_s=0.01)
        v1 = r.read(frame)
        time.sleep(0.05)
        black = _solid_bgr(20, 100, (0, 0, 0))
        v2 = r.read(black)
        self.assertAlmostEqual(v2, 0.0, delta=1.0)

    def test_get_cached_returns_none_before_first_read(self):
        r = self._reader()
        self.assertIsNone(r.get_cached())

    def test_get_cached_after_read(self):
        frame = _make_red_bar_frame(fill=0.6)
        r = self._reader()
        r.read(frame)
        self.assertIsNotNone(r.get_cached())

    def test_invalidate_clears_cache(self):
        frame = _make_red_bar_frame(fill=0.6)
        r = self._reader()
        r.read(frame)
        r.invalidate()
        self.assertIsNone(r.get_cached())

    # ── is_low ───────────────────────────────────────────────────
    def test_is_low_true_when_below_threshold(self):
        frame = _make_red_bar_frame(fill=0.3)  # ~30%
        r = self._reader(threshold_pct=50.0)
        r.read(frame)
        self.assertTrue(r.is_low())

    def test_is_low_false_when_above_threshold(self):
        frame = _make_red_bar_frame(fill=0.8)  # ~80%
        r = self._reader(threshold_pct=50.0)
        r.read(frame)
        self.assertFalse(r.is_low())

    def test_is_low_false_before_read(self):
        r = self._reader()
        self.assertFalse(r.is_low())

    def test_is_low_custom_threshold(self):
        frame = _make_red_bar_frame(fill=0.6)
        r = self._reader()
        r.read(frame)
        self.assertFalse(r.is_low(threshold_pct=50.0))
        self.assertTrue(r.is_low(threshold_pct=70.0))

    # ── region 형식 ──────────────────────────────────────────────
    def test_region_as_dict(self):
        from perception.hp import HpReader
        r = HpReader(region={"x": 0, "y": 0, "w": 100, "h": 20})
        frame = _make_red_bar_frame(fill=1.0)
        self.assertGreaterEqual(r.read(frame), 90.0)

    def test_region_as_tuple(self):
        from perception.hp import HpReader
        r = HpReader(region=(0, 0, 100, 20))
        frame = _make_red_bar_frame(fill=1.0)
        self.assertGreaterEqual(r.read(frame), 90.0)

    # ── from_settings ────────────────────────────────────────────
    def test_from_settings(self):
        from perception.hp import HpReader
        s = _Settings()
        r = HpReader.from_settings(s)
        self.assertIsInstance(r, HpReader)
        frame = _make_red_bar_frame(fill=0.6)
        pct = r.read(frame)
        self.assertGreater(pct, 0.0)

    # ── 경계 조건 ─────────────────────────────────────────────────
    def test_region_outside_frame_returns_cached_or_zero(self):
        frame = _solid_bgr(20, 100, (0, 0, 255))
        r = self._reader(region=(200, 200, 50, 50))  # 프레임 밖
        pct = r.read(frame)
        self.assertIsInstance(pct, float)


# ════════════════════════════════════════════════════════════════
#  LevelReader 테스트
# ════════════════════════════════════════════════════════════════
class TestLevelReader(unittest.TestCase):
    """easyocr 없으면 대부분 skip. 있어도 OCR 결과는 환경 의존."""

    def _make_reader(self, **kw):
        from perception.level import LevelReader
        region = kw.pop("region", (0, 0, 120, 30))
        return LevelReader(region=region, **kw)

    def test_import_ok(self):
        import perception.level  # noqa: F401

    def test_initial_cached_is_none(self):
        r = self._make_reader()
        self.assertIsNone(r.get_cached())

    def test_is_ready_eventually_bool(self):
        r = self._make_reader()
        # ready 는 bool
        self.assertIsInstance(r.is_ready, bool)

    def test_read_before_ready_returns_none(self):
        from perception.level import LevelReader
        r = LevelReader.__new__(LevelReader)
        r._reader_ready = False
        r._cached_level = None
        r.read_interval_s = 2.0
        r._last_read_at = 0.0
        frame = _solid_bgr(30, 120, (0, 0, 0))
        result = r.read(frame)
        self.assertIsNone(result)

    def test_invalidate(self):
        r = self._make_reader()
        r._cached_level = 42
        r.invalidate()
        self.assertIsNone(r.get_cached())

    def test_from_settings(self):
        from perception.level import LevelReader
        s = _Settings()
        r = LevelReader.from_settings(s)
        self.assertIsInstance(r, LevelReader)
        self.assertEqual(r.min_confidence, 0.4)

    def test_parse_level_helper(self):
        from perception.level import _parse_level
        self.assertEqual(_parse_level("LEV 42"), 42)
        self.assertEqual(_parse_level("Lv.5"), 5)
        self.assertEqual(_parse_level("lv 99"), 99)
        self.assertEqual(_parse_level("42"), 42)
        self.assertIsNone(_parse_level("HELLO"))
        self.assertIsNone(_parse_level("100"))   # 범위 초과
        self.assertIsNone(_parse_level("0"))

    def test_read_returns_cached_within_interval(self):
        from perception.level import LevelReader
        r = LevelReader.__new__(LevelReader)
        r._reader_ready = True
        r._reader = None  # OCR 스킵
        r._cached_level = 30
        r._last_read_at = time.monotonic()
        r.read_interval_s = 10.0
        r.min_confidence = 0.4
        r._monitor_offset = (0, 0)
        r._rx = r._ry = 0
        r._rw, r._rh = 120, 30
        frame = _solid_bgr(30, 120, (0, 0, 0))
        result = r.read(frame)
        self.assertEqual(result, 30)


# ════════════════════════════════════════════════════════════════
#  LootDetector 테스트
# ════════════════════════════════════════════════════════════════
def _make_white_box_frame(
    fw: int = 200,
    fh: int = 200,
    bx: int = 50,
    by: int = 50,
    bw: int = 60,
    bh: int = 40,
) -> np.ndarray:
    """흰색 사각형이 있는 BGR 프레임 생성."""
    frame = _solid_bgr(fh, fw, (0, 50, 0))  # 어두운 초록 배경
    # 흰색 테두리 (BGR 255,255,255)
    frame[by : by + bh, bx : bx + bw] = (255, 255, 255)
    return frame


class TestLootDetector(unittest.TestCase):
    def setUp(self):
        from perception.loot import LootDetector
        self.LootDetector = LootDetector

    # ── 기본 탐지 ─────────────────────────────────────────────────
    def test_detects_white_box(self):
        detector = self.LootDetector()
        frame = _make_white_box_frame()
        items = detector.find(frame)
        self.assertGreater(len(items), 0, "흰 박스가 탐지되어야 한다")

    def test_no_detection_on_dark_frame(self):
        detector = self.LootDetector()
        frame = _solid_bgr(200, 200, (0, 50, 0))  # 흰 박스 없음
        detector.invalidate()
        items = detector.find(frame)
        self.assertEqual(len(items), 0)

    def test_loot_item_has_cx_cy(self):
        detector = self.LootDetector()
        frame = _make_white_box_frame(bx=50, by=50, bw=60, bh=40)
        items = detector.find(frame)
        if items:
            it = items[0]
            self.assertGreater(it.cx, 0)
            self.assertGreater(it.cy, 0)
            self.assertEqual(it.label, "loot")

    def test_label_is_loot(self):
        detector = self.LootDetector()
        frame = _make_white_box_frame()
        items = detector.find(frame)
        for it in items:
            self.assertEqual(it.label, "loot")

    def test_score_range(self):
        detector = self.LootDetector()
        frame = _make_white_box_frame()
        items = detector.find(frame)
        for it in items:
            self.assertGreaterEqual(it.score, 0.0)
            self.assertLessEqual(it.score, 1.0)

    # ── 캐시 ─────────────────────────────────────────────────────
    def test_cache_within_interval(self):
        detector = self.LootDetector(scan_interval_s=10.0)
        frame = _make_white_box_frame()
        r1 = detector.find(frame)
        # 프레임을 바꿔도 캐시 내 결과 유지
        black = _solid_bgr(200, 200, (0, 0, 0))
        r2 = detector.find(black)
        self.assertEqual(len(r1), len(r2))

    def test_invalidate_clears_cache(self):
        detector = self.LootDetector(scan_interval_s=10.0)
        frame = _make_white_box_frame()
        detector.find(frame)
        detector.invalidate()
        black = _solid_bgr(200, 200, (0, 0, 0))
        r2 = detector.find(black)
        self.assertEqual(len(r2), 0)

    # ── find_nearest ─────────────────────────────────────────────
    def test_find_nearest_returns_none_on_empty(self):
        detector = self.LootDetector()
        black = _solid_bgr(200, 200, (0, 0, 0))
        result = detector.find_nearest(black, 100, 100)
        self.assertIsNone(result)

    def test_find_nearest_returns_closest(self):
        detector = self.LootDetector()
        # 두 흰 박스를 만든다
        frame = _solid_bgr(300, 400, (0, 50, 0))
        frame[50:90, 50:120] = (255, 255, 255)   # 박스 A: cx≈85, cy≈70
        frame[200:240, 300:370] = (255, 255, 255) # 박스 B: cx≈335, cy≈220
        detector.invalidate()
        # (80, 70) 에 가까운 것 → 박스 A
        nearest = detector.find_nearest(frame, 80, 70)
        if nearest:
            self.assertLess(nearest.cx, 200)

    # ── region 형식 ──────────────────────────────────────────────
    def test_region_as_tuple(self):
        from perception.loot import LootDetector
        d = LootDetector(region=(0, 0, 200, 200))
        frame = _make_white_box_frame()
        items = d.find(frame)
        self.assertIsInstance(items, list)

    def test_region_as_dict(self):
        from perception.loot import LootDetector
        d = LootDetector(region={"x": 0, "y": 0, "w": 200, "h": 200})
        frame = _make_white_box_frame()
        items = d.find(frame)
        self.assertIsInstance(items, list)

    # ── from_settings ────────────────────────────────────────────
    def test_from_settings_no_region(self):
        from perception.loot import LootDetector
        s = _Settings()
        d = LootDetector.from_settings(s)
        self.assertIsInstance(d, LootDetector)

    def test_from_settings_with_region(self):
        from perception.loot import LootDetector
        s = _Settings()
        s.loot.region_w = 200
        s.loot.region_h = 200
        d = LootDetector.from_settings(s)
        self.assertTrue(d._has_region)

    # ── LootItem unpacking (하위 호환) ───────────────────────────
    def test_loot_item_iterable(self):
        from perception.loot import LootItem
        it = LootItem(x=10, y=20, w=50, h=30, cx=35, cy=35, label="loot", score=0.9)
        x, y, label, score = it
        self.assertEqual(x, 10)
        self.assertEqual(label, "loot")

    # ── 면적 필터 ─────────────────────────────────────────────────
    def test_small_box_filtered_out(self):
        from perception.loot import LootDetector
        d = LootDetector(min_area=5000)  # 아주 큰 최소 면적
        frame = _make_white_box_frame(bw=10, bh=10)  # 100px² → 필터링
        d.invalidate()
        items = d.find(frame)
        self.assertEqual(len(items), 0)


# ════════════════════════════════════════════════════════════════
#  __init__.py lazy import 확인
# ════════════════════════════════════════════════════════════════
class TestPerceptionPackage(unittest.TestCase):
    def test_import_package_without_error(self):
        import perception  # noqa: F401

    def test_all_submodule_names(self):
        import perception
        for name in ["hp", "level", "loot"]:
            self.assertIn(name, perception.__all__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
