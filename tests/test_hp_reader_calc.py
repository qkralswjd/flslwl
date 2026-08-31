"""HpReader._calc_hp_pct() 단위 테스트.

가로 방향 연속 채우기 로직 검증:
- HP 100% → 전체 열이 파란색
- HP 50%  → 왼쪽 절반만 파란색
- HP 0%   → 파란 픽셀 없음
- 테두리 포함 케이스 (위아래 1px 테두리 = 비파란 행)

리니지 클래식 HP 바는 파란색(royal blue, BGR≈(220,80,30)) 계열.
"""

import numpy as np
import pytest

from automation.hp_reader import _calc_hp_pct

# ─── 헬퍼 ───────────────────────────────────────────────────────────────────


# 리니지 클래식 HP 바 파란색 (BGR: B=220, G=80, R=30 → HSV H≈110)
_BLUE_BGR = (220, 80, 30)


def _make_bar(width: int, height: int, filled_ratio: float) -> np.ndarray:
    """왼쪽 filled_ratio 비율만큼 파란색(BGR≈220,80,30)으로 채운 이미지 반환."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    filled_cols = int(round(width * filled_ratio))
    if filled_cols > 0:
        img[:, :filled_cols] = _BLUE_BGR
    return img


def _make_bar_with_border(width: int, inner_height: int, filled_ratio: float) -> np.ndarray:
    """위아래 1px 어두운 테두리 + 내부 HP 바 이미지."""
    total_height = inner_height + 2  # 위아래 1px
    img = np.zeros((total_height, width, 3), dtype=np.uint8)
    filled_cols = int(round(width * filled_ratio))
    if filled_cols > 0:
        # 내부 행(1 ~ inner_height)만 파란색
        img[1:1 + inner_height, :filled_cols] = _BLUE_BGR
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


# ─── 테두리 포함 케이스 ─────────────────────────────────────────────────────


def test_hp_100_with_border():
    """HP 100% + 테두리: 전체 채워진 경우 100% 반환."""
    img = _make_bar_with_border(200, 18, 1.0)
    assert _calc_hp_pct(img) == 100.0


def test_hp_50_with_border():
    """HP 50% + 테두리: 절반 채워진 경우 50% 반환 (테두리가 비율에 영향 없음)."""
    img = _make_bar_with_border(200, 18, 0.5)
    assert _calc_hp_pct(img) == 50.0


def test_hp_0_with_border():
    """HP 0% + 테두리: 빨간 픽셀 없음 → 0%."""
    img = _make_bar_with_border(200, 18, 0.0)
    assert _calc_hp_pct(img) == 0.0


# ─── 이전 버전의 오류 케이스 재현 ────────────────────────────────────────────


def test_old_pixel_ratio_bug_100pct():
    """이전 로직(전체 픽셀 비율) 재현: h=53 중 실제 HP 바가 26px 높이면 50% 오류.

    새 로직(가로 방향 열 비율)은 100%를 올바르게 반환해야 함.
    """
    # h=53, 실제 HP 바는 내부 26px (위아래 13~14px 여백/테두리)
    width = 335
    total_h = 53
    bar_h = 26  # 실제 HP 바 높이

    img = np.zeros((total_h, width, 3), dtype=np.uint8)
    # HP 100%: 전체 열을 파란색으로 채우되 높이는 bar_h만
    top = (total_h - bar_h) // 2
    img[top:top + bar_h, :] = _BLUE_BGR

    result = _calc_hp_pct(img)
    # 새 로직: 가로 방향으로 모든 열이 채워짐 → 100%
    assert result == 100.0, f"예상 100.0, 실제 {result}"


def test_old_pixel_ratio_bug_50pct():
    """이전 로직 재현: HP 100%인데 50%로 나오던 케이스 방어.

    HP 바 높이가 전체 region 높이의 절반 → 전체 픽셀 비율 방식이면 50%,
    새 로직(열 비율)은 100%를 올바르게 반환.
    """
    width = 100
    total_h = 20
    bar_h = 10  # 절반 높이

    img = np.zeros((total_h, width, 3), dtype=np.uint8)
    img[5:15, :] = _BLUE_BGR  # 가운데 10px 행에 HP 바

    result = _calc_hp_pct(img)
    assert result == 100.0, f"예상 100.0, 실제 {result}"


# ─── 경계값 ─────────────────────────────────────────────────────────────────


def test_empty_image():
    """빈 이미지 → 100.0 반환 (기본값)."""
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _calc_hp_pct(img) == 100.0


def test_single_pixel_blue():
    """1×1 파란 픽셀 → 100%."""
    img = np.array([[list(_BLUE_BGR)]], dtype=np.uint8)
    assert _calc_hp_pct(img) == 100.0


def test_single_pixel_black():
    """1×1 검은 픽셀 → 0%."""
    img = np.array([[[0, 0, 0]]], dtype=np.uint8)
    assert _calc_hp_pct(img) == 0.0
