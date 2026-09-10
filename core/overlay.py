"""core/overlay.py — cv2 오버레이 드로잉 헬퍼.

flslwl/overlay/overlay.py 를 bot2 아키텍처에 맞게 이식.

변경점:
  - Enemy 는 core.tracking.Enemy 사용 (flslwl tracker.Enemy 대체)
  - draw_hud() 에 hp / level / mode_state 파라미터 추가
  - draw_detection_zone() 딕셔너리 대신 Settings 객체도 허용
  - 모든 함수 Optional import — cv2 미설치 환경에서도 import 오류 없음
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.tracking import Enemy

# ── 색상 상수 ────────────────────────────────────────────────────────────────
_COLOR_TARGET   = (0, 255,   0)    # 초록 — 현재 타겟
_COLOR_ACTIVE   = (0,   0, 255)    # 빨강 — 일반 감지
_COLOR_PREDICT  = (0, 255, 255)    # 노랑 — 위치 예측 중
_COLOR_HUD      = (0, 255,   0)    # 녹색 — HUD 텍스트
_COLOR_ROI      = (255, 255,  0)   # 시안 — ROI 테두리
_COLOR_ZONE     = (255, 120,  0)   # 주황 — 탐지 존 테두리

MAX_ARROW_LENGTH = 80


# ════════════════════════════════════════════════════════════════════════════
# draw_enemies
# ════════════════════════════════════════════════════════════════════════════

def draw_enemies(
    frame: np.ndarray,
    enemies: List["Enemy"],
    roi_offset: tuple = (0, 0),
    current_target_id: Optional[int] = None,
    target_state_name: str = "",
) -> np.ndarray:
    """enemies 리스트를 frame 에 그린다.

    Parameters
    ----------
    frame               : BGR numpy 배열 (수정됨)
    enemies             : Enemy 객체 리스트
    roi_offset          : (ox, oy) — ROI 잘라낸 경우 오프셋
    current_target_id   : 현재 타겟 Enemy.id
    target_state_name   : SequentialTargetSM 상태 이름 문자열
    """
    ox, oy = roi_offset

    for enemy in enemies:
        x, y = enemy.x + ox, enemy.y + oy
        w, h = enemy.width, enemy.height
        is_target = (enemy.id == current_target_id)

        if is_target:
            color     = _COLOR_TARGET
            thickness = 3
        elif enemy.predicted:
            color     = _COLOR_PREDICT
            thickness = 2
        else:
            color     = _COLOR_ACTIVE
            thickness = 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # 현재 타겟 — 상태 배지
        if is_target and target_state_name:
            badge_text = f"▶ {target_state_name}"
            badge_pos  = (x, y - 28)
            (tw, th), _ = cv2.getTextSize(
                badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (badge_pos[0] - 2, badge_pos[1] - th - 2),
                (badge_pos[0] + tw + 2, badge_pos[1] + 2),
                (0, 80, 0), cv2.FILLED,
            )
            cv2.putText(
                frame, badge_text, badge_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_TARGET, 1, cv2.LINE_AA,
            )

        # 레이블 (Enemy ID / 좌표 / 속도)
        label_lines = [
            f"Enemy #{enemy.id}" + (" ◀ TARGET" if is_target else ""),
            f"X:{enemy.center_x + ox} Y:{enemy.center_y + oy}",
            f"VX:{enemy.velocity_x:+.0f} VY:{enemy.velocity_y:+.0f}",
        ]
        if enemy.confidence < 1.0:
            label_lines.append(f"Conf:{enemy.confidence * 100:.0f}%")

        text_y = y - (36 if is_target else 10)
        for line in reversed(label_lines):
            cv2.putText(
                frame, line, (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
            text_y -= 16

        # 속도 화살표
        cx, cy = enemy.center_x + ox, enemy.center_y + oy
        vx, vy = enemy.velocity_x * 3, enemy.velocity_y * 3
        length = math.hypot(vx, vy)
        if length > MAX_ARROW_LENGTH:
            scale = MAX_ARROW_LENGTH / length
            vx, vy = vx * scale, vy * scale
        if length > 1:
            cv2.arrowedLine(
                frame, (cx, cy),
                (int(cx + vx), int(cy + vy)),
                color, 2, tipLength=0.3,
            )

    return frame


# ════════════════════════════════════════════════════════════════════════════
# draw_hud
# ════════════════════════════════════════════════════════════════════════════

def draw_hud(
    frame: np.ndarray,
    capture_fps: float,
    detection_fps: float,
    processing_ms: float,
    enemy_count: int,
    pico_connected: bool = False,
    pico_port: Optional[str] = None,
    target_state_name: str = "",
    current_target_id: Optional[int] = None,
    hp_ratio: Optional[float] = None,
    level: Optional[int] = None,
    mode_state: str = "",
) -> np.ndarray:
    """화면 좌상단에 HUD 정보를 그린다.

    Parameters
    ----------
    hp_ratio    : 0.0–1.0 HP 비율 (None 이면 미표시)
    level       : 캐릭터 레벨 (None 이면 미표시)
    mode_state  : 현재 FSM 모드 상태 이름
    """
    lines = [
        f"Capture FPS : {capture_fps:.1f}",
        f"Detection FPS: {detection_fps:.1f}",
        f"Processing  : {processing_ms:.1f} ms",
        f"Enemies     : {enemy_count}",
    ]

    if hp_ratio is not None:
        hp_pct = hp_ratio * 100.0
        hp_color = (
            (0, 255,   0) if hp_pct >= 60 else
            (0, 165, 255) if hp_pct >= 30 else
            (0,   0, 255)
        )
        lines.append(f"HP          : {hp_pct:.0f}%")
    else:
        hp_color = _COLOR_HUD

    if level is not None:
        lines.append(f"Level       : {level}")

    if mode_state:
        lines.append(f"Mode State  : {mode_state}")

    y = 20
    for i, line in enumerate(lines):
        color = hp_color if ("HP" in line) else _COLOR_HUD
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA,
        )
        y += 21

    # 타겟 상태
    if current_target_id is not None:
        target_line  = f"Target      : #{current_target_id}  [{target_state_name}]"
        target_color = _COLOR_TARGET
    else:
        target_line  = f"Target      : None  [{target_state_name}]"
        target_color = (180, 180, 180)
    cv2.putText(
        frame, target_line, (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, target_color, 1, cv2.LINE_AA,
    )
    y += 21

    # Pico 상태
    if pico_port is not None:
        pico_text  = f"Pico        : {'Connected' if pico_connected else 'Disconnected'} ({pico_port})"
        pico_color = (0, 255, 0) if pico_connected else (0, 100, 255)
    else:
        pico_text  = "Pico        : Disabled"
        pico_color = (120, 120, 120)
    cv2.putText(
        frame, pico_text, (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, pico_color, 1, cv2.LINE_AA,
    )

    return frame


# ════════════════════════════════════════════════════════════════════════════
# draw_roi
# ════════════════════════════════════════════════════════════════════════════

def draw_roi(frame: np.ndarray, roi) -> np.ndarray:
    """ROI 사각형을 시안 테두리로 그린다.

    roi 는 dict {"x", "y", "width", "height"} 또는 None.
    """
    if roi is None:
        return frame
    x = roi.get("x", 0) if isinstance(roi, dict) else getattr(roi, "x", 0)
    y = roi.get("y", 0) if isinstance(roi, dict) else getattr(roi, "y", 0)
    w = roi.get("width", 0) if isinstance(roi, dict) else getattr(roi, "width", 0)
    h = roi.get("height", 0) if isinstance(roi, dict) else getattr(roi, "height", 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), _COLOR_ROI, 1)
    return frame


# ════════════════════════════════════════════════════════════════════════════
# draw_detection_zone
# ════════════════════════════════════════════════════════════════════════════

def draw_detection_zone(
    frame: np.ndarray,
    detection_zone,
    roi_offset: tuple = (0, 0),
) -> np.ndarray:
    """탐지 존 사각형을 주황 테두리로 그린다.

    detection_zone 는 dict {"enabled", "center_x", "center_y",
    "half_width", "half_height"} 또는 None.
    """
    if detection_zone is None:
        return frame

    enabled = (
        detection_zone.get("enabled", False)
        if isinstance(detection_zone, dict)
        else getattr(detection_zone, "enabled", False)
    )
    if not enabled:
        return frame

    def _get(obj, key, default):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    ox, oy = roi_offset
    cx  = _get(detection_zone, "center_x",   720) + ox
    cy  = _get(detection_zone, "center_y",   390) + oy
    hw  = _get(detection_zone, "half_width",  420)
    hh  = _get(detection_zone, "half_height", 280)

    x0, y0 = cx - hw, cy - hh
    x1, y1 = cx + hw, cy + hh

    cv2.rectangle(frame, (x0, y0), (x1, y1), _COLOR_ZONE, 1)
    cv2.line(frame, (cx - 8, cy), (cx + 8, cy), _COLOR_ZONE, 1)
    cv2.line(frame, (cx, cy - 8), (cx, cy + 8), _COLOR_ZONE, 1)
    cv2.putText(
        frame, "DETECT ZONE", (x0 + 4, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_ZONE, 1, cv2.LINE_AA,
    )
    return frame


# ════════════════════════════════════════════════════════════════════════════
# draw_mode_state_banner
# ════════════════════════════════════════════════════════════════════════════

def draw_mode_state_banner(
    frame: np.ndarray,
    state_name: str,
    color: tuple = (0, 165, 255),
) -> np.ndarray:
    """화면 하단 중앙에 현재 FSM 상태 배너를 그린다."""
    if not state_name:
        return frame
    h, w = frame.shape[:2]
    text = f"[ {state_name} ]"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    tx = (w - tw) // 2
    ty = h - 20
    # 반투명 배경 (검정 직사각형)
    cv2.rectangle(
        frame,
        (tx - 6, ty - th - 4),
        (tx + tw + 6, ty + 4),
        (0, 0, 0), cv2.FILLED,
    )
    cv2.putText(
        frame, text, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
    )
    return frame
