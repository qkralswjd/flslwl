"""
tools/capture_monster.py — 몬스터 탬플릿 이미지 수집 도구.

사용법:
  python -m tools.capture_monster
  python tools/capture_monster.py --out templates/ --monitor 2

기능:
  - 화면 캡처 실시간 표시 (cv2 창)
  - 마우스 드래그로 ROI 선택
  - Space / Enter → 선택 영역을 PNG 로 저장
  - R → ROI 초기화
  - Q / ESC → 종료

저장 형식: templates/monster_YYYYMMDD_HHMMSS.png
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

# ── cv2 확인 ────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
except ImportError:
    print("[capture_monster] cv2/numpy 가 필요합니다: pip install opencv-python numpy")
    sys.exit(1)


# ─────────────────────────── 전역 상태 ──────────────────────────

_dragging   = False
_drag_start = (0, 0)
_roi        : tuple | None = None  # (x1, y1, x2, y2)
_frame_copy : np.ndarray | None = None


def _on_mouse(event, x, y, flags, param):
    global _dragging, _drag_start, _roi, _frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        _dragging   = True
        _drag_start = (x, y)
        _roi        = None
    elif event == cv2.EVENT_MOUSEMOVE and _dragging:
        if _frame_copy is not None:
            vis = _frame_copy.copy()
            cv2.rectangle(vis, _drag_start, (x, y), (0, 255, 0), 2)
            cv2.imshow("capture_monster", vis)
    elif event == cv2.EVENT_LBUTTONUP:
        _dragging = False
        x1, y1 = _drag_start
        x2, y2 = x, y
        if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
            _roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            print(f"ROI 선택: {_roi}")


def _save_crop(frame: np.ndarray, roi: tuple, out_dir: Path) -> str:
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out_dir / f"monster_{ts}.png"
    cv2.imwrite(str(fname), crop)
    return str(fname)


# ─────────────────────────── 메인 ───────────────────────────────

def run(monitor_index: int = 1, out_dir: str = "templates"):
    global _frame_copy, _roi

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ScreenCapturer 사용 (없으면 mss 직접)
    try:
        from hardware.screen import ScreenCapturer
        cap = ScreenCapturer(monitor_index=monitor_index)
    except ImportError:
        try:
            import mss
            cap = None
            sct = mss.mss()
            monitor = sct.monitors[monitor_index]
        except ImportError:
            print("[capture_monster] mss 없음. pip install mss")
            return
    else:
        sct = None

    cv2.namedWindow("capture_monster", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("capture_monster", _on_mouse)

    print("[capture_monster] 조작:")
    print("  마우스 드래그 : ROI 선택")
    print("  Space / Enter : 선택 영역 저장")
    print("  R             : ROI 초기화")
    print("  Q / ESC       : 종료")

    count = 0
    while True:
        # 프레임 취득
        if cap is not None:
            frame = cap.grab()
        else:
            raw   = sct.grab(monitor)
            frame = np.array(raw)[:, :, :3]  # BGRA → BGR

        _frame_copy = frame.copy()

        # ROI 표시
        vis = frame.copy()
        if _roi:
            x1, y1, x2, y2 = _roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, "Space/Enter: save  R: reset",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(vis, "드래그로 ROI 선택",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(vis, f"저장: {count}개  출력: {out_dir}",
                    (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("capture_monster", vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            break
        elif key in (ord(" "), 13):  # Space or Enter
            if _roi:
                saved = _save_crop(frame, _roi, out_path)
                count += 1
                print(f"[capture_monster] 저장됨({count}): {saved}")
                _roi = None
            else:
                print("[capture_monster] ROI 를 먼저 선택하세요")
        elif key == ord("r"):
            _roi = None
            print("[capture_monster] ROI 초기화")

    cv2.destroyAllWindows()
    if cap:
        try:
            cap.close()
        except Exception:
            pass
    print(f"[capture_monster] 종료. 총 {count}개 저장 → {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="몬스터 탬플릿 이미지 수집")
    parser.add_argument("--monitor", "-m", type=int, default=1,
                        help="모니터 인덱스 (기본 1)")
    parser.add_argument("--out", "-o", default="templates",
                        help="저장 디렉토리 (기본 templates/)")
    args = parser.parse_args()
    run(monitor_index=args.monitor, out_dir=args.out)


if __name__ == "__main__":
    main()
