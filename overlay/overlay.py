"""Drawing helpers: tracker output → on-screen overlay.

변경점 (순차 타겟 통합):
    draw_enemies() 에 current_target_id, target_state 파라미터 추가.
    - 현재 타겟 : 초록 굵은 박스 + 상태 배지 (LOCKING / CLICKING / WAITING_DEAD)
    - 일반 감지 : 빨간 박스 (원본과 동일)
    - 예측(소실) : 노란 박스 (원본과 동일)
"""

import math
from typing import Optional

import cv2

MAX_ARROW_LENGTH = 80

# 상태별 색상
_COLOR_TARGET   = (0, 255, 0)    # 초록 — 현재 타겟
_COLOR_ACTIVE   = (0, 0, 255)    # 빨강 — 일반 감지
_COLOR_PREDICT  = (0, 255, 255)  # 노랑 — 위치 예측 중
_COLOR_HUD      = (0, 255, 0)    # 녹색 — HUD 텍스트


def draw_enemies(
    frame,
    enemies,
    roi_offset=(0, 0),
    current_target_id: Optional[int] = None,
    target_state_name: str = "",
):
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

        # 현재 타겟이면 상태 배지를 박스 위에 그림
        if is_target and target_state_name:
            badge_text = f"▶ {target_state_name}"
            badge_pos  = (x, y - 28)
            # 배경 직사각형 (가독성)
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,
                          (badge_pos[0] - 2, badge_pos[1] - th - 2),
                          (badge_pos[0] + tw + 2, badge_pos[1] + 2),
                          (0, 80, 0), cv2.FILLED)
            cv2.putText(frame, badge_text, badge_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_TARGET, 1, cv2.LINE_AA)

        # Enemy ID / 좌표 / 속도 레이블
        label_lines = [
            f"Enemy #{enemy.id}" + (" ◀ TARGET" if is_target else ""),
            f"X:{enemy.center_x + ox} Y:{enemy.center_y + oy}",
            f"VX:{enemy.velocity_x:+.0f} VY:{enemy.velocity_y:+.0f}",
        ]
        if enemy.confidence < 1.0:
            label_lines.append(f"Conf:{enemy.confidence * 100:.0f}%")

        text_y = y - (36 if is_target else 10)
        for line in reversed(label_lines):
            cv2.putText(frame, line, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            text_y -= 16

        # 속도 화살표
        cx, cy = enemy.center_x + ox, enemy.center_y + oy
        vx, vy = enemy.velocity_x * 3, enemy.velocity_y * 3
        length = math.hypot(vx, vy)
        if length > MAX_ARROW_LENGTH:
            scale = MAX_ARROW_LENGTH / length
            vx, vy = vx * scale, vy * scale
        if length > 1:
            cv2.arrowedLine(frame, (cx, cy),
                            (int(cx + vx), int(cy + vy)),
                            color, 2, tipLength=0.3)
    return frame


def draw_hud(
    frame,
    capture_fps: float,
    detection_fps: float,
    processing_ms: float,
    enemy_count: int,
    pico_connected: bool = False,
    pico_port: Optional[str] = None,
    target_state_name: str = "",
    current_target_id: Optional[int] = None,
):
    lines = [
        f"Capture FPS: {capture_fps:.1f}",
        f"Detection FPS: {detection_fps:.1f}",
        f"Processing: {processing_ms:.1f} ms",
        f"Enemies: {enemy_count}",
    ]

    # 타겟 상태 줄
    if current_target_id is not None:
        target_line  = f"Target: #{current_target_id}  [{target_state_name}]"
        target_color = _COLOR_TARGET
    else:
        target_line  = f"Target: None  [{target_state_name}]"
        target_color = (180, 180, 180)

    # Pico 상태 줄
    if pico_port is not None:
        pico_text  = f"Pico: {'Connected' if pico_connected else 'Disconnected'} ({pico_port})"
        pico_color = (0, 255, 0) if pico_connected else (0, 100, 255)
    else:
        pico_text  = "Pico: Disabled"
        pico_color = (120, 120, 120)

    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_HUD, 1, cv2.LINE_AA)
        y += 22

    cv2.putText(frame, target_line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, target_color, 1, cv2.LINE_AA)
    y += 22

    cv2.putText(frame, pico_text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, pico_color, 1, cv2.LINE_AA)
    return frame


def draw_roi(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
    return frame


def draw_detection_zone(frame, detection_zone, roi_offset=(0, 0)):
    """detection_zone 사각형을 화면에 표시합니다 (반투명 파란 테두리).

    Args:
        detection_zone: config["detection_zone"] 딕셔너리
        roi_offset: ROI offset (ox, oy)
    """
    if detection_zone is None or not detection_zone.get("enabled", False):
        return frame

    ox, oy = roi_offset
    cx  = detection_zone.get("center_x", 720) + ox
    cy  = detection_zone.get("center_y", 390) + oy
    hw  = detection_zone.get("half_width",  420)
    hh  = detection_zone.get("half_height", 280)

    x0, y0 = cx - hw, cy - hh
    x1, y1 = cx + hw, cy + hh

    # 파란 점선 테두리
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 120, 0), 1)

    # 중심 십자선
    cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (255, 120, 0), 1)
    cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (255, 120, 0), 1)

    # 라벨
    cv2.putText(frame, "DETECT ZONE", (x0 + 4, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1, cv2.LINE_AA)
    return frame
