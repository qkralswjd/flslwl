"""tests/test_hardware.py — hardware 레이어 유닛 테스트.

실제 장치(Pico, 모니터) 없이 실행 가능한 순수 로직 테스트.
  - NullPicoWorker 인터페이스 완전성
  - KEY_NAME_MAP 정합성
  - ScreenCapturer 초기화 (mss 모킹)
  - Settings 연동 팩토리
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ── mss 스텁 (실제 디스플레이 없이 import 가능하도록) ────────────────────────
def _make_mss_stub():
    mss_mod = types.ModuleType("mss")
    monitor_list = [
        {"left": 0,    "top": 0, "width": 3840, "height": 2160},  # 0: all
        {"left": 0,    "top": 0, "width": 1920, "height": 1080},  # 1: primary
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # 2: secondary
    ]
    fake_ctx = MagicMock()
    fake_ctx.monitors = monitor_list
    fake_ctx.grab.return_value = MagicMock()
    mss_mod.mss = MagicMock(return_value=fake_ctx)
    return mss_mod, fake_ctx, monitor_list


class TestNullPicoWorker(unittest.TestCase):
    """NullPicoWorker 인터페이스 완전성 검증."""

    def setUp(self):
        # 서브모듈 직접 import (hardware/__init__ 의 lazy 정책 때문)
        import hardware.pico as _pico
        self.w = _pico.NullPicoWorker()

    def test_is_connected_always_true(self):
        self.assertTrue(self.w.is_connected)

    def test_is_idle_always_true(self):
        self.assertTrue(self.w.is_idle)

    def test_start_stop_no_exception(self):
        self.w.start()
        self.w.stop()
        self.w.stop_target()

    def test_mouse_methods_no_exception(self):
        self.w.click(100, 200)
        self.w.click(100, 200, pulse_ms=30)
        self.w.click_current_pos()
        self.w.click_current_pos(pulse_ms=10)
        self.w.drag(0, 0, 100, 100)
        self.w.drag(0, 0, 100, 100, steps=4)

    def test_keyboard_methods_no_exception(self):
        self.w.key_tap(0x3A)
        self.w.key_tap(0x3A, hold_ms=100)
        self.w.key_down(0x3A)
        self.w.key_up(0x3A)
        self.w.key_tap_name("F1")
        self.w.key_tap_name("ENTER")
        self.w.key_tap_name("SPACE", hold_ms=80)

    def test_unknown_key_name_no_exception(self):
        """알 수 없는 키 이름은 조용히 무시해야 한다."""
        self.w.key_tap_name("NONEXISTENT_KEY_XYZ")

    def test_enqueue_no_exception(self):
        self.w.enqueue("STOP")


class TestKeyNameMap(unittest.TestCase):
    """KEY_NAME_MAP 정합성 검증."""

    def setUp(self):
        import hardware.pico as _pico
        self.m = _pico.KEY_NAME_MAP

    def test_function_keys_present(self):
        for n in range(1, 13):
            key = f"F{n}"
            self.assertIn(key, self.m, f"{key} 없음")

    def test_special_keys_present(self):
        for key in ("ENTER", "ESCAPE", "ESC", "SPACE", "BACKSPACE",
                    "TAB", "DELETE", "HOME", "END", "UP", "DOWN", "LEFT", "RIGHT"):
            self.assertIn(key, self.m, f"{key} 없음")

    def test_alphabet_keys_present(self):
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIn(ch, self.m, f"{ch} 없음")

    def test_digit_keys_present(self):
        for d in "0123456789":
            self.assertIn(d, self.m, f"{d} 없음")

    def test_values_are_ints(self):
        for k, v in self.m.items():
            self.assertIsInstance(v, int, f"{k}의 값이 int가 아님: {type(v)}")

    def test_esc_alias(self):
        """ESC 와 ESCAPE 는 같은 키코드여야 한다."""
        self.assertEqual(self.m["ESC"], self.m["ESCAPE"])

    def test_f1_keycode(self):
        self.assertEqual(self.m["F1"], 0x3A)

    def test_enter_keycode(self):
        self.assertEqual(self.m["ENTER"], 0x28)


class TestListSerialPorts(unittest.TestCase):
    """list_serial_ports() 반환 타입 검증."""

    def test_returns_list(self):
        import hardware.pico as _pico
        result = _pico.list_serial_ports()
        self.assertIsInstance(result, list)

    def test_elements_are_strings(self):
        import hardware.pico as _pico
        for port in _pico.list_serial_ports():
            self.assertIsInstance(port, str)


class TestScreenCapturer(unittest.TestCase):
    """ScreenCapturer 초기화 및 메서드 검증 (mss 모킹)."""

    def setUp(self):
        self.mss_mod, self.fake_ctx, self.monitors = _make_mss_stub()
        # cv2도 스텁 처리
        cv2_mod = types.ModuleType("cv2")
        import numpy as np
        fake_bgr = np.zeros((1080, 1920, 3), dtype="uint8")
        cv2_mod.cvtColor    = MagicMock(return_value=fake_bgr)
        cv2_mod.COLOR_BGRA2BGR = 4
        cv2_mod.COLOR_BGR2GRAY = 6
        self._patches = [
            patch.dict(sys.modules, {"mss": self.mss_mod, "cv2": cv2_mod}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _make(self, monitor_index=1, region=None):
        # 모듈 캐시 우회를 위해 매번 reimport
        import importlib
        if "hardware.screen" in sys.modules:
            del sys.modules["hardware.screen"]
        mod = importlib.import_module("hardware.screen")
        return mod.ScreenCapturer(monitor_index=monitor_index, region=region)

    def test_init_full_monitor(self):
        cap = self._make(monitor_index=1)
        self.assertEqual(cap.monitor_index, 1)
        self.assertIsNone(cap.region)

    def test_init_with_region(self):
        region = {"x": 10, "y": 20, "width": 800, "height": 600}
        cap = self._make(monitor_index=1, region=region)
        self.assertEqual(cap.region, region)

    def test_invalid_monitor_raises(self):
        with self.assertRaises(ValueError):
            self._make(monitor_index=99)

    def test_frame_size_full_monitor(self):
        cap = self._make(monitor_index=1)
        w, h = cap.frame_size
        self.assertEqual(w, 1920)
        self.assertEqual(h, 1080)

    def test_frame_size_with_region(self):
        region = {"x": 0, "y": 0, "width": 640, "height": 360}
        cap = self._make(monitor_index=1, region=region)
        w, h = cap.frame_size
        self.assertEqual(w, 640)
        self.assertEqual(h, 360)

    def test_list_monitors_returns_list(self):
        cap = self._make(monitor_index=1)
        result = cap.list_monitors()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_set_monitor(self):
        cap = self._make(monitor_index=1)
        cap.set_monitor(2)
        self.assertEqual(cap.monitor_index, 2)

    def test_set_region(self):
        cap = self._make(monitor_index=1)
        new_region = {"x": 5, "y": 5, "width": 200, "height": 100}
        cap.set_region(new_region)
        self.assertEqual(cap.region, new_region)

    def test_context_manager(self):
        import importlib
        if "hardware.screen" in sys.modules:
            del sys.modules["hardware.screen"]
        mod = importlib.import_module("hardware.screen")
        with mod.ScreenCapturer(monitor_index=1) as cap:
            self.assertIsNotNone(cap)


class TestNullPicoFromSettings(unittest.TestCase):
    """NullPicoWorker.from_settings() 팩토리 테스트."""

    def test_from_settings_returns_instance(self):
        import hardware.pico as _pico
        # Settings 없이 None 넘겨도 동작해야 함 (인자 미사용)
        worker = _pico.NullPicoWorker.from_settings(None)  # type: ignore[arg-type]
        self.assertIsInstance(worker, _pico.NullPicoWorker)
        self.assertTrue(worker.is_connected)
        self.assertTrue(worker.is_idle)


if __name__ == "__main__":
    unittest.main(verbosity=2)
