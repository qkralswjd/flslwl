"""몬스터 템플릿 캡처 도구

사용법:
    py -3.11 capture_monster.py

조작키 (백그라운드 실행 중):
    T         : 현재 화면 프리즈 → 풀스크린으로 표시 → 드래그 → 엔터 저장
    R         : 현재 화면 프리즈 → 풀스크린으로 표시 → 드래그 → 엔터 리젝트 저장
    ESC / C   : 취소
    Q         : 종료

저장 위치:
    config/templates/        : 몬스터 템플릿 (T → 엔터)
    config/templates_reject/ : 오탐지 리젝트 템플릿 (R → 엔터)
"""

import ctypes
import ctypes.wintypes
import json
import os
import sys

import cv2
import mss
import numpy as np

# ── 경로 설정 ────────────────────────────────────────────────────────
HERE          = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH   = os.path.join(HERE, "config", "config.json")
TEMPLATES_DIR = os.path.join(HERE, "config", "templates")
REJECTS_DIR   = os.path.join(HERE, "config", "templates_reject")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(REJECTS_DIR,   exist_ok=True)


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _imwrite(path: str, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return ok


def _next_index(directory: str, prefix: str, ext: str = ".png") -> int:
    existing = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
    nums = []
    for name in existing:
        try:
            nums.append(int(name[len(prefix):-len(ext)]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


# ── Win32 키/마우스 ───────────────────────────────────────────────────
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
GetCursorPos     = ctypes.windll.user32.GetCursorPos

VK_T       = ord('T')
VK_R       = ord('R')
VK_Q       = ord('Q')
VK_C       = ord('C')
VK_ESC     = 0x1B
VK_ENTER   = 0x0D
VK_LBUTTON = 0x01


def key_just_pressed(vk: int, prev: dict) -> bool:
    now = bool(GetAsyncKeyState(vk) & 0x8000)
    was = prev.get(vk, False)
    prev[vk] = now
    return now and not was


def get_cursor():
    pt = ctypes.wintypes.POINT()
    GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def lmb_down():
    return bool(GetAsyncKeyState(VK_LBUTTON) & 0x8000)


# ── 풀스크린 프리즈 선택 ─────────────────────────────────────────────
def freeze_and_select_fullscreen(frozen_frame, win_name, mon_left, mon_top):
    """프리즈된 화면을 1:1 풀사이즈로 띄우고 드래그로 영역 선택.
    엔터=확정, ESC/C=취소.
    반환: (x1,y1,x2,y2) 프레임 기준 픽셀좌표 or None
    """
    h, w = frozen_frame.shape[:2]

    # 풀스크린 창 띄우기
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    drag_start   = None
    drag_end     = None
    drag_active  = False
    sel_rect     = None
    prev_lmb     = False
    prev_keys    = {}

    while True:
        disp = frozen_frame.copy()

        # 현재 커서 위치 (모니터 기준 절대좌표 → 프레임 내 좌표)
        cx, cy = get_cursor()
        rx = cx - mon_left
        ry = cy - mon_top
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))

        cur_lmb = lmb_down()

        if cur_lmb and not prev_lmb:
            drag_start  = (rx, ry)
            drag_end    = (rx, ry)
            drag_active = True
            sel_rect    = None

        elif cur_lmb and drag_active:
            drag_end = (rx, ry)

        elif not cur_lmb and prev_lmb and drag_active:
            drag_end    = (rx, ry)
            drag_active = False
            x1 = min(drag_start[0], drag_end[0])
            y1 = min(drag_start[1], drag_end[1])
            x2 = max(drag_start[0], drag_end[0])
            y2 = max(drag_start[1], drag_end[1])
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                sel_rect = (x1, y1, x2, y2)
            else:
                sel_rect = None

        prev_lmb = cur_lmb

        # 드래그 중 사각형
        if drag_active and drag_start:
            cv2.rectangle(disp, drag_start, (rx, ry), (0, 255, 255), 1)

        # 확정 선택 사각형
        if sel_rect:
            x1, y1, x2, y2 = sel_rect
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 크기 표시
            label = f"{x2-x1}x{y2-y1}px  ENTER=저장  ESC=취소"
            cv2.putText(disp, label, (x1, max(y1-8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 안내 메시지
        guide = "드래그로 몬스터 선택 → ENTER 저장 / ESC 취소"
        cv2.putText(disp, guide, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, disp)
        cv2.waitKey(1)

        if key_just_pressed(VK_ENTER, prev_keys):
            if sel_rect:
                # 창 닫고 일반 모드로 복귀
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                return sel_rect
            # 선택 없으면 무시
        if key_just_pressed(VK_ESC, prev_keys) or key_just_pressed(VK_C, prev_keys):
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            return None


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    cfg         = _load_config()
    monitor_idx = cfg.get("monitor_index", 2)

    sct      = mss.mss()
    monitors = sct.monitors
    if monitor_idx >= len(monitors):
        print(f"[ERROR] monitor_index={monitor_idx} 범위 초과. 최대 {len(monitors)-1}")
        sys.exit(1)

    mon      = monitors[monitor_idx]
    mon_left = mon["left"]
    mon_top  = mon["top"]
    mon_w    = mon["width"]
    mon_h    = mon["height"]

    print("=" * 56)
    print("  몬스터 템플릿 캡처 도구 (풀스크린 프리즈 모드)")
    print("=" * 56)
    print(f"  모니터 #{monitor_idx}  {mon_w}x{mon_h}")
    print()
    print("  T       : 화면 프리즈 → 풀스크린 → 드래그 → 엔터 → 템플릿 저장")
    print("  R       : 화면 프리즈 → 풀스크린 → 드래그 → 엔터 → 리젝트 저장")
    print("  ESC / C : 취소")
    print("  Q       : 종료")
    print()

    WIN = "capture_monster"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 400, 80)

    prev_keys  = {}
    status_msg = "T=템플릿저장  R=리젝트저장  Q=종료"

    while True:
        # 실시간 캡처 (미리보기용 작은 창)
        shot  = sct.grab(mon)
        frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)

        n_tmpl = len([f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".png")])
        n_rej  = len([f for f in os.listdir(REJECTS_DIR)   if f.endswith(".png")])

        info = np.zeros((80, 400, 3), dtype=np.uint8)
        cv2.putText(info, f"templates={n_tmpl}  rejects={n_rej}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(info, status_msg,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow(WIN, info)
        cv2.waitKey(1)

        # T : 프리즈 → 풀스크린 → 템플릿 저장
        if key_just_pressed(VK_T, prev_keys):
            frozen = frame.copy()
            rect = freeze_and_select_fullscreen(frozen, WIN, mon_left, mon_top)
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, 400, 80)
            if rect:
                x1, y1, x2, y2 = rect
                crop = frozen[y1:y2, x1:x2]
                idx  = _next_index(TEMPLATES_DIR, "mob_")
                path = os.path.join(TEMPLATES_DIR, f"mob_{idx:03d}.png")
                _imwrite(path, crop)
                status_msg = f"✓ mob_{idx:03d}.png ({crop.shape[1]}x{crop.shape[0]}px)"
                print(f"[템플릿] {path}  size={crop.shape[1]}x{crop.shape[0]}")
            else:
                status_msg = "취소됨 — T=템플릿  R=리젝트  Q=종료"

        # R : 프리즈 → 풀스크린 → 리젝트 저장
        elif key_just_pressed(VK_R, prev_keys):
            frozen = frame.copy()
            rect = freeze_and_select_fullscreen(frozen, WIN, mon_left, mon_top)
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, 400, 80)
            if rect:
                x1, y1, x2, y2 = rect
                crop = frozen[y1:y2, x1:x2]
                idx  = _next_index(REJECTS_DIR, "rej_")
                path = os.path.join(REJECTS_DIR, f"rej_{idx:03d}.png")
                _imwrite(path, crop)
                status_msg = f"✓ rej_{idx:03d}.png ({crop.shape[1]}x{crop.shape[0]}px)"
                print(f"[리젝트] {path}  size={crop.shape[1]}x{crop.shape[0]}")
            else:
                status_msg = "취소됨 — T=템플릿  R=리젝트  Q=종료"

        # Q : 종료
        elif key_just_pressed(VK_Q, prev_keys):
            break

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    sct.close()
    print("[종료]")
    n_tmpl = len([f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".png")])
    print(f"  저장된 템플릿: {n_tmpl}개  ({TEMPLATES_DIR})")


if __name__ == "__main__":
    main()
