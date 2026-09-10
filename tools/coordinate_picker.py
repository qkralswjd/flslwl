"""
tools/coordinate_picker.py — 화면 좌표 및 색상 픽커.

사용법:
  python -m tools.coordinate_picker
  python tools/coordinate_picker.py --monitor 2 --output coords.json

기능:
  - 화면 실시간 표시
  - 마우스 이동 → 현재 좌표·BGR·HSV 표시
  - 왼쪽 클릭 → 좌표 기록 (목록에 추가)
  - S → JSON 파일로 내보내기
  - C → 기록 초기화
  - Q / ESC → 종료

출력 JSON 구조:
  [{"idx": 1, "x": 960, "y": 540, "bgr": [B, G, R], "hsv": [H, S, V]}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("[coordinate_picker] pip install opencv-python numpy")
    sys.exit(1)


# ─────────────────────────── 전역 ───────────────────────────────

_cursor_pos  = (0, 0)
_picked      : list[dict] = []


def _on_mouse(event, x, y, flags, param):
    global _cursor_pos
    _cursor_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        if frame is not None:
            b, g, r = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])
            pixel_hsv = cv2.cvtColor(
                np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV
            )[0, 0].tolist()
            entry = {
                "idx": len(_picked) + 1,
                "x": x, "y": y,
                "bgr": [b, g, r],
                "hsv": pixel_hsv,
            }
            _picked.append(entry)
            print(f"[{entry['idx']:3d}] ({x:5d},{y:5d})  BGR={entry['bgr']}  HSV={entry['hsv']}")


def _save_json(path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_picked, f, ensure_ascii=False, indent=2)
    print(f"[coordinate_picker] {len(_picked)}개 좌표 → {path}")


# ─────────────────────────── 메인 ───────────────────────────────

def run(monitor_index: int = 1, output: str = ""):
    global _cursor_pos, _picked

    try:
        from hardware.screen import ScreenCapturer
        cap = ScreenCapturer(monitor_index=monitor_index)
        use_cap = True
    except ImportError:
        try:
            import mss as _mss
            cap = None
            sct = _mss.mss()
            monitor = sct.monitors[monitor_index]
            use_cap = False
        except ImportError:
            print("[coordinate_picker] mss 없음: pip install mss")
            return

    cv2.namedWindow("coordinate_picker", cv2.WINDOW_NORMAL)
    param = {"frame": None}
    cv2.setMouseCallback("coordinate_picker", _on_mouse, param)

    print("[coordinate_picker] 조작:")
    print("  마우스 이동    : 좌표·색상 표시")
    print("  왼쪽 클릭     : 좌표 기록")
    print("  S             : JSON 저장")
    print("  C             : 기록 초기화")
    print("  Q / ESC       : 종료")

    while True:
        if use_cap:
            frame = cap.grab()
        else:
            raw   = sct.grab(monitor)
            frame = np.array(raw)[:, :, :3]

        param["frame"] = frame
        vis = frame.copy()

        # 커서 정보 표시
        cx, cy = _cursor_pos
        h, w = frame.shape[:2]
        if 0 <= cy < h and 0 <= cx < w:
            b, g, r = int(frame[cy, cx, 0]), int(frame[cy, cx, 1]), int(frame[cy, cx, 2])
            pixel_hsv = cv2.cvtColor(
                np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV
            )[0, 0].tolist()
            info = f"({cx},{cy}) BGR=({b},{g},{r}) HSV={pixel_hsv}  기록:{len(_picked)}"
        else:
            info = f"({cx},{cy})  기록:{len(_picked)}"

        # 커서 십자선
        cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
        cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 0), 1)

        # 텍스트
        cv2.rectangle(vis, (0, 0), (len(info) * 7 + 10, 26), (30, 30, 30), -1)
        cv2.putText(vis, info, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 기록 목록 (마지막 5개)
        for i, entry in enumerate(_picked[-5:], 1):
            txt = f"{entry['idx']}: ({entry['x']},{entry['y']})"
            ypos = vis.shape[0] - (6 - i) * 20 - 5
            cv2.putText(vis, txt, (6, ypos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1)

        cv2.imshow("coordinate_picker", vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = output or f"coords_{ts}.json"
            _save_json(path)
        elif key == ord("c"):
            _picked.clear()
            print("[coordinate_picker] 기록 초기화")

    cv2.destroyAllWindows()
    if use_cap:
        try:
            cap.close()
        except Exception:
            pass

    if _picked and not output:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"coords_{ts}.json"
        _save_json(path)

    print(f"[coordinate_picker] 종료. 총 {len(_picked)}개 기록")


def main():
    parser = argparse.ArgumentParser(description="화면 좌표·색상 픽커")
    parser.add_argument("--monitor", "-m", type=int, default=1)
    parser.add_argument("--output",  "-o", default="",
                        help="저장 JSON 경로 (기본: coords_타임스탬프.json)")
    args = parser.parse_args()
    run(monitor_index=args.monitor, output=args.output)


if __name__ == "__main__":
    main()
