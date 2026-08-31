"""LevelReader._parse_level() — LEV: 패턴 포함 파싱 테스트."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from automation.level_reader import _parse_level


class TestParseLevelLEV:
    """LEV: 계열 패턴 테스트 (리니지 클래식 UI 실제 형식)."""

    def test_lev_colon_int(self):
        assert _parse_level("LEV:14") == 14

    def test_lev_colon_lowercase(self):
        assert _parse_level("lev:5") == 5

    def test_lev_mixed_case(self):
        assert _parse_level("Lev:7") == 7

    def test_lev_dot(self):
        assert _parse_level("LEV.9") == 9

    def test_lev_space(self):
        assert _parse_level("LEV 3") == 3

    def test_lev_no_sep(self):
        assert _parse_level("LEV14") == 14

    def test_lev_with_spaces_around(self):
        assert _parse_level("  LEV:14  ") == 14

    def test_lev_two_digits(self):
        assert _parse_level("LEV:10") == 10


class TestParseLevelLv:
    """기존 Lv. 계열 패턴이 여전히 작동하는지 확인."""

    def test_lv_dot(self):
        assert _parse_level("Lv.5") == 5

    def test_lv_space(self):
        assert _parse_level("Lv 5") == 5

    def test_lv_upper(self):
        assert _parse_level("LV.5") == 5

    def test_lv_lower(self):
        assert _parse_level("lv5") == 5

    def test_lv_none_on_garbage(self):
        assert _parse_level("HUNTING") is None

    def test_lv_none_on_empty(self):
        assert _parse_level("") is None

    def test_number_only(self):
        assert _parse_level("7") == 7

    def test_number_out_of_range(self):
        # 100은 1~99 범위 밖 — Lv./LEV. 패턴 없으면 None
        assert _parse_level("100") is None

    def test_lev_priority_over_number(self):
        # "LEV:14" 는 LEV 패턴으로 14 반환 (숫자만 패턴과 충돌 없음)
        assert _parse_level("LEV:14") == 14
