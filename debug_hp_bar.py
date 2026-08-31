"""HP 바 디버그 스크립트.

실행하면:
1. 현재 화면에서 hp_bar.region 영역을 캡처
2. debug_hp_crop.png  — 원본 크롭 이미지
3. debug_hp_mask.png  — HSV 빨간색 마스크
4. debug_hp_info.txt  — 픽셀 분석 결과 (열별 빨간 픽셀 수)

Windows에서 실행:
    cd C:\\Users\\dongj\\flslwl\\flslwl
    python debug_hp_bar.py
"""

import json
import pathlib
import sys

import cv2
import mss
import numpy as np

# ── 설정 로드 ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
cfg_main = json.loads((ROOT / "config" / "config.json").read_text(encoding="utf-8"))
cfg_auto = json.loads((ROOT / "config" / "config_automation.json").read_text(encoding="utf-8"))

monitor_index = cfg_main.get("monitor_index", 2)
hp_region = cfg_auto["hp_bar"]["region"]   # x, y, width, height
offset_x  = cfg_auto.get("capture_offset", {}).get("x", 0)
offset_y  = cfg_auto.get("capture_offset", {}).get("y", 0)

print(f"monitor_index : {monitor_index}")
print(f"capture_offset: x={offset_x}, y={offset_y}")
print(f"hp_bar.region : {hp_region}")

# ── 화면 캡처 ──────────────────────────────────────────────────────────────
with mss.mss() as sct:
    monitors = sct.monitors
    print(f"\n=== 모니터 목록 ===")
    for i, m in enumerate(monitors):
        print(f"  [{i}] {m}")

    if monitor_index >= len(monitors):
        print(f"ERROR: monitor_index={monitor_index} 없음. 최대 인덱스={len(monitors)-1}")
        sys.exit(1)

    mon = monitors[monitor_index]
    print(f"\n사용 모니터: {mon}")

    shot = sct.grab(mon)
    frame = np.array(shot)[:, :, :3]   # BGRA → BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

print(f"frame size: {frame.shape[1]}x{frame.shape[0]} (w×h)")

# ── HP 바 크롭 ─────────────────────────────────────────────────────────────
x = hp_region["x"] - offset_x
y = hp_region["y"] - offset_y
w = hp_region["width"]
h = hp_region["height"]

print(f"\n크롭 좌표 (offset 적용 후): x={x}, y={y}, w={w}, h={h}")

fh, fw = frame.shape[:2]
x2 = min(x + w, fw)
y2 = min(y + h, fh)

if x < 0 or y < 0 or x >= fw or y >= fh:
    print(f"ERROR: 크롭 좌표가 프레임 밖입니다! frame=({fw},{fh})")
    sys.exit(1)

crop = frame[y:y2, x:x2]
print(f"crop shape: {crop.shape}")

# ── HSV 마스크 생성 ────────────────────────────────────────────────────────
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

HP_HSV_RANGES = [
    # 파란색 영역 (H=95~135, S=80+, V=60+)
    ((95, 80, 60), (135, 255, 255)),
]

mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
for (lower, upper) in HP_HSV_RANGES:
    m = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.bitwise_or(mask, m)

# ── 열별 빨간 픽셀 수 분석 ────────────────────────────────────────────────
col_red_count = np.sum(mask > 0, axis=0)   # 각 열의 빨간 픽셀 수
col_has_red   = col_red_count > 0

total_cols  = mask.shape[1]
total_rows  = mask.shape[0]
total_red   = int(np.count_nonzero(mask))

# 왼쪽부터 연속 채워진 열 수
filled_cols = 0
for has_red in col_has_red:
    if has_red:
        filled_cols += 1
    else:
        break

hp_pct_new  = round(filled_cols / total_cols * 100.0, 1)
hp_pct_old  = round(total_red / (total_cols * total_rows) * 100.0, 1)

print(f"\n=== 분석 결과 ===")
print(f"전체 열 수       : {total_cols}")
print(f"전체 행 수       : {total_rows}")
print(f"전체 빨간 픽셀   : {total_red} / {total_cols * total_rows}")
print(f"빨간 열 수(전체) : {int(np.count_nonzero(col_has_red))} / {total_cols}")
print(f"연속 채워진 열   : {filled_cols} / {total_cols}")
print(f"HP% (새 로직)    : {hp_pct_new}%  ← 연속 열 비율")
print(f"HP% (구 로직)    : {hp_pct_old}%  ← 전체 픽셀 비율")

# 처음 50열 빨간 픽셀 수 출력 (왼쪽 테두리 확인)
print(f"\n=== 처음 30열 빨간 픽셀 수 (왼쪽 테두리 확인) ===")
for i in range(min(30, total_cols)):
    bar = "█" * min(col_red_count[i], 20)
    print(f"  col[{i:3d}]: {col_red_count[i]:3d}px  {bar}")

# 마지막 10열 (오른쪽 끝 확인)
print(f"\n=== 마지막 10열 빨간 픽셀 수 (오른쪽 끝 확인) ===")
for i in range(max(0, total_cols - 10), total_cols):
    bar = "█" * min(col_red_count[i], 20)
    print(f"  col[{i:3d}]: {col_red_count[i]:3d}px  {bar}")

# ── 이미지 저장 ────────────────────────────────────────────────────────────
cv2.imwrite("debug_hp_crop.png", crop)
print(f"\n✅ debug_hp_crop.png 저장 완료")

# 마스크를 컬러로 저장 (빨간 = 감지된 픽셀)
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
mask_color[mask > 0] = (0, 0, 255)
cv2.imwrite("debug_hp_mask.png", mask_color)
print(f"✅ debug_hp_mask.png 저장 완료")

# 크롭 이미지에 감지된 픽셀 오버레이
overlay = crop.copy()
overlay[mask > 0] = (0, 0, 255)
cv2.imwrite("debug_hp_overlay.png", overlay)
print(f"✅ debug_hp_overlay.png 저장 완료")

print(f"\n→ debug_hp_crop.png 을 확인해서 HP 바가 올바르게 잡혔는지 확인하세요.")
print(f"→ debug_hp_mask.png 에서 빨간 픽셀이 어디에 있는지 확인하세요.")
