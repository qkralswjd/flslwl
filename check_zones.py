"""
DETECT ZONE (템플릿 매칭존) + 아데나 스캔존 동시 확인 스크립트
ESC 또는 Q 키로 종료
"""
import cv2
import numpy as np
import json
import mss

# config.json에서 detection_zone 읽기
with open("config/config.json", encoding="utf-8") as f:
    cfg = json.load(f)
dz = cfg.get("detection_zone", {})
cx = dz.get("center_x", 960);  cy = dz.get("center_y", 540)
hw = dz.get("half_width", 600); hh = dz.get("half_height", 400)

# 두 존 좌표 (동일해야 함)
DZ_X0, DZ_Y0 = cx - hw, cy - hh
DZ_X1, DZ_Y1 = cx + hw, cy + hh

ADENA_X0, ADENA_Y0 = DZ_X0, DZ_Y0   # 동일
ADENA_X1, ADENA_Y1 = DZ_X1, DZ_Y1   # 동일

print(f"[DETECT ZONE]  x={DZ_X0}~{DZ_X1}  y={DZ_Y0}~{DZ_Y1}")
print(f"[아데나 스캔존] x={ADENA_X0}~{ADENA_X1}  y={ADENA_Y0}~{ADENA_Y1}")
print("ESC / Q = 종료")

with mss.mss() as sct:
    mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    while True:
        shot  = sct.grab(mon)
        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

        # DETECT ZONE — 파란색 (main.py와 동일)
        cv2.rectangle(frame, (DZ_X0, DZ_Y0), (DZ_X1, DZ_Y1), (255, 120, 0), 2)
        cv2.putText(frame, "DETECT ZONE", (DZ_X0 + 4, DZ_Y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

        # 아데나 스캔존 — 초록색 점선 효과 (2px 안쪽)
        cv2.rectangle(frame, (ADENA_X0+3, ADENA_Y0+3),
                      (ADENA_X1-3, ADENA_Y1-3), (0, 255, 80), 1)
        cv2.putText(frame, "ADENA SCAN", (ADENA_X0 + 4, ADENA_Y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

        # 동일 여부 표시
        same = (DZ_X0 == ADENA_X0 and DZ_Y0 == ADENA_Y0
                and DZ_X1 == ADENA_X1 and DZ_Y1 == ADENA_Y1)
        status = "ZONE MATCH ✓" if same else "ZONE MISMATCH !"
        color  = (0, 255, 0) if same else (0, 0, 255)
        cv2.putText(frame, status, (cx - 80, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 화면 축소 표시 (1280 기준)
        sc = min(1.0, 1280 / frame.shape[1])
        disp = cv2.resize(frame, (int(frame.shape[1]*sc), int(frame.shape[0]*sc)))
        cv2.imshow("Zone Check [ESC=종료]", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

cv2.destroyAllWindows()
print("종료")
