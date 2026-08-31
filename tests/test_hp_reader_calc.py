"""HpReader._calc_hp_pct() 단위 테스트.

실측값 기반:
  center_BGR=[1, 1, 87]  center_HSV=[0, 252, 87]
  → 리니지 클래식 HP 바는 빨간색(H=0, S=252, V=87) 계열

검증 케이스:
- HP 100% → 전체 열이 빨간색
- HP 50%  → 왼쪽 절반만 빨간색
- HP 0%   → 빨간 픽셀 없음
- 텍스트 포함 케이스 (중간 일부 열에 빨간 픽셀 없음)
- 테두리 포함 케이스 (위아래 여백)
"""

import numpy as np
import pytest

from automation.hp_reader import _calc_hp_pct

# ─── 실측 HP 바 색상 ─────────────────────────────────────────────────────────
# 실제 게임 캡처: center_BGR=[1, 1, 87] → BGR 순서 (B=1, G=1, R=87)
# HSV: H=0, S=252, V=87 → 어두운 빨간색
_RED_BGR = (1, 1, 87)


# ─── 헬퍼 ───────────────────────────────────────────────────────────────────

def _make_bar(width: int, height: int, filled_ratio: float) -> np.ndarray:
    """왼쪽 filled_ratio 비율만큼 빨간색으로 채운 이미지 반환."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    filled_cols = int(round(width * filled_ratio))
    if filled_cols > 0:
        img[:, :filled_cols] = _RED_BGR
    return img


def _make_bar_with_border(width: int, inner_height: int, filled_ratio: float) -> np.ndarray:
    """위아래 1px 어두운 테두리 + 내부 HP 바 이미지."""
    total_height = inner_height + 2
    img = np.zeros((total_height, width, 3), dtype=np.uint8)
    filled_cols = int(round(width * filled_ratio))
    if filled_cols > 0:
        img[1:1 + inner_height, :filled_cols] = _RED_BGR
    return img


def _make_bar_with_text_gap(width: int, height: int, filled_ratio: float,
                             gap_start: int, gap_len: int) -> np.ndarray:
    """HP 바 중간에 텍스트 영역(빨간 픽셀 없는 열)이 있는 이미지.

    HP 바 텍스트('HP : 109 / 109')로 인해 연속 열이 끊기는 케이스.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    filled_cols = int(round(width * filled_ratio))
    if filled_cols > 0:
        img[:, :filled_cols] = _RED_BGR
    # 텍스트 영역: gap_start~gap_start+gap_len 열을 검정으로 덮음
    if gap_start < filled_cols:
        img[:, gap_start:gap_start + gap_len] = (0, 0, 0)
    return img


# ─── 기본 케이스 ────────────────────────────────────────────────────────────

def test_hp_100_full_bar():
    """HP 100%: 전체 열이 빨간색."""
    img = _make_bar(200, 20, 1.0)
    assert _calc_hp_pct(img) == 100.0


def test_hp_50_half_bar():
    """HP 50%: 왼쪽 절반만 빨간색."""
    img = _make_bar(200, 20, 0.5)
    assert _calc_hp_pct(img) == 50.0


def test_hp_0_empty_bar():
    """HP 0%: 빨간 픽셀 없음."""
    img = _make_bar(200, 20, 0.0)
    assert _calc_hp_pct(img) == 0.0


def test_hp_25_quarter_bar():
    """HP 25%: 왼쪽 1/4만 빨간색."""
    img = _make_bar(200, 20, 0.25)
    assert _calc_hp_pct(img) == 25.0


def test_hp_75_three_quarter_bar():
    """HP 75%: 왼쪽 3/4 빨간색."""
    img = _make_bar(200, 20, 0.75)
    assert _calc_hp_pct(img) == 75.0


# ─── 텍스트 갭 포함 케이스 ──────────────────────────────────────────────────

def test_hp_100_with_text_gap():
    """HP 100% + 중간 텍스트 갭: 전체 열 비율로 계산 → 갭 제외한 비율."""
    # 200열 중 10열이 텍스트로 비어있으면 190/200 = 95%
    img = _make_bar_with_text_gap(200, 20, 1.0, gap_start=90, gap_len=10)
    result = _calc_hp_pct(img)
    assert result == 95.0


def test_hp_50_with_text_gap():
    """HP 50% + 텍스트 갭이 HP 바 안에 있는 경우."""
    # 200열 중 왼쪽 100열이 HP 바, 그 중 10열이 텍스트 갭 → 90/200 = 45%
    img = _make_bar_with_text_gap(200, 20, 0.5, gap_start=45, gap_len=10)
    result = _calc_hp_pct(img)
    assert result == 45.0


# ─── 테두리 포함 케이스 ─────────────────────────────────────────────────────

def test_hp_100_with_border():
    """HP 100% + 테두리."""
    img = _make_bar_with_border(200, 18, 1.0)
    assert _calc_hp_pct(img) == 100.0


def test_hp_50_with_border():
    """HP 50% + 테두리."""
    img = _make_bar_with_border(200, 18, 0.5)
    assert _calc_hp_pct(img) == 50.0


def test_hp_0_with_border():
    """HP 0% + 테두리."""
    img = _make_bar_with_border(200, 18, 0.0)
    assert _calc_hp_pct(img) == 0.0


# ─── 실제 사이즈 케이스 (w=335, h=53) ────────────────────────────────────────

def test_real_size_100pct():
    """실제 region 크기(335×53) HP 100%."""
    img = _make_bar(335, 53, 1.0)
    assert _calc_hp_pct(img) == 100.0


def test_real_size_50pct():
    """실제 region 크기(335×53) HP 50% (반올림 오차 ±1% 허용)."""
    img = _make_bar(335, 53, 0.5)
    assert abs(_calc_hp_pct(img) - 50.0) <= 1.0


# ─── 경계값 ─────────────────────────────────────────────────────────────────

def test_empty_image():
    """빈 이미지 → 100.0 반환 (기본값)."""
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _calc_hp_pct(img) == 100.0


def test_single_pixel_red():
    """1×1 빨간 픽셀 → 100%."""
    img = np.array([[list(_RED_BGR)]], dtype=np.uint8)
    assert _calc_hp_pct(img) == 100.0


def test_single_pixel_black():
    """1×1 검은 픽셀 → 0%."""
    img = np.array([[[0, 0, 0]]], dtype=np.uint8)
    assert _calc_hp_pct(img) == 0.0
