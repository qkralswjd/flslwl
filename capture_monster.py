"""몬스터 템플릿 캡처 도구

사용법:
    py -3.11 capture_monster.py

조작키:
    T         : 화면 프리즈 → 드래그로 영역 선택 → 엔터로 저장
    R         : 화면 프리즈 → 드래그로 영역 선택 → 엔터로 리젝트 저장
    C / ESC   : 프리즈 취소 (실시간 복귀)
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
HERE            = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH     = os.path.join(HERE, "config", "config.json")
TEMPLATES_DIR   = os.path.join(HERE, "config", "templates")
REJECTS_DIR     = os.path.join(HERE, "config", "templates_reject")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(REJECTS_DIR,   exist_ok=True)


# ── config 로드 ──────────────────────────────────────────────────────
def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 파일 저장 ────────────────────────────────────────────────────────
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


# ── 윈도우 키 감지 ────────────────────────────────────────────────────
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
GetCursorPos     = ctypes.windll.user32.GetCursorPos

VK_T      = ord('T')
VK_R      = ord('R')
VK_Q      = ord('Q')
VK_C      = ord('C')
VK_ESC    = 0x1B
VK_ENTER  = 0x0D
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


# ── 프리즈 모드: 정지 화면에서 드래그 선택 ──────────────────────────
def freeze_and_select(frozen_frame, win_name, mon_left, mon_top, scale):
    """프리즈된 화면에서 드래그로 영역 선택. 엔터=확정, ESC/C=취소.
    반환: (x1,y1,x2,y2) 모니터 기준 절대좌표 or None
    """
    drag_start  = None
    drag_end    = None
    drag_active = False
    sel_rect    = None
    prev_lmb    = False
    prev_keys   = {}

    status = "드래그로 몬스터 선택 → 엔터 저장 / ESC 취소"

    while True:
        disp = frozen_frame.copy()

        cx, cy = get_cursor()
        rx = int((cx - mon_left) * scale)
        ry = int((cy - mon_top)  * scale)

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
            w, h = x2 - x1, y2 - y1
            if w > 5 and h > 5:
                # 화면 표시 좌표 → 실제 모니터 좌표로 역변환
                sx1 = int(x1 / scale)
                sy1 = int(y1 / scale)
                sx2 = int(x2 / scale)
                sy2 = int(y2 / scale)
                sel_rect = (sx1, sy1, sx2, sy2)
                status = f"선택: ({sx1},{sy1})-({sx2},{sy2})  엔터=저장  ESC=취소"
            else:
                sel_rect = None
                status = "너무 작음. 다시 드래그하세요."

        prev_lmb = cur_lmb

        # 드래그 중 사각형
        if drag_active and drag_start:
            cv2.rectangle(disp, drag_start, (rx, ry), (0, 255, 255), 1)

        # 확정 선택 사각형 (표시 좌표)
        if sel_rect:
            dx1 = int(sel_rect[0] * scale)
            dy1 = int(sel_rect[1] * scale)
            dx2 = int(sel_rect[2] * scale)
            dy2 = int(sel_rect[3] * scale)
            cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)

            # 미리보기
            crop_disp = disp[dy1:dy2, dx1:dx2]
            if crop_disp.size > 0:
                ph = pw = 150
                preview = cv2.resize(crop_disp, (pw, ph))
                disp[10:10+ph, disp.shape[1]-pw-10:disp.shape[1]-10] = preview
                cv2.rectangle(disp,
                              (disp.shape[1]-pw-10, 10),
                              (disp.shape[1]-10, 10+ph),
                              (0, 0, 255), 1)

        # 상태 메시지
        cv2.putText(disp, "[FREEZE] " + status, (10, disp.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, disp)
        cv2.waitKey(1)

        # 엔터 → 저장
        if key_just_pressed(VK_ENTER, prev_keys):
            if sel_rect:
                return sel_rect
            status = "먼저 드래그로 영역을 선택하세요."

        # ESC / C → 취소
        if key_just_pressed(VK_ESC, prev_keys) or key_just_pressed(VK_C, prev_keys):
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
    print("  몬스터 템플릿 캡처 도구 (프리즈 모드)")
    print("=" * 56)
    print(f"  모니터 #{monitor_idx}  {mon_w}x{mon_h}")
    print()
    print("  T       : 화면 프리즈 → 드래그 → 엔터 → 템플릿 저장")
    print("  R       : 화면 프리즈 → 드래그 → 엔터 → 리젝트 저장")
    print("  ESC / C : 프리즈 취소")
    print("  Q       : 종료")
    print()

    WIN = "capture_monster"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)

    # 표시용 스케일 (창 크기 / 실제 해상도)
    disp_h = min(mon_h, 810)
    scale  = disp_h / mon_h

    prev_keys  = {}
    status_msg = "T키: 프리즈 후 템플릿 저장  /  R키: 리젝트 저장  /  Q: 종료"

    while True:
        # ── 실시간 캡처 ──────────────────────────────────────────
        shot  = sct.grab(mon)
        frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)

        # 표시용 리사이즈
        disp_w = int(mon_w * scale)
        disp   = cv2.resize(frame, (disp_w, disp_h))

        # 템플릿/리젝트 개수
        n_tmpl = len([f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".png")])
        n_rej  = len([f for f in os.listdir(REJECTS_DIR)   if f.endswith(".png")])
        cv2.putText(disp, f"templates={n_tmpl}  rejects={n_rej}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(disp, status_msg, (10, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(WIN, disp)
        cv2.waitKey(1)

        # ── T : 프리즈 → 템플릿 저장 ─────────────────────────────
        if key_just_pressed(VK_T, prev_keys):
            frozen      = frame.copy()
            frozen_disp = cv2.resize(frozen, (disp_w, disp_h))
            rect = freeze_and_select(frozen_disp, WIN, 0, 0, scale)
            if rect:
                x1, y1, x2, y2 = rect
                crop = frozen[y1:y2, x1:x2]
                idx  = _next_index(TEMPLATES_DIR, "mob_")
                path = os.path.join(TEMPLATES_DIR, f"mob_{idx:03d}.png")
                _imwrite(path, crop)
                status_msg = f"✓ 템플릿 저장: mob_{idx:03d}.png ({crop.shape[1]}x{crop.shape[0]}px)"
                print(f"[템플릿] {path}  size={crop.shape[1]}x{crop.shape[0]}")
            else:
                status_msg = "취소됨"

        # ── R : 프리즈 → 리젝트 저장 ─────────────────────────────
        elif key_just_pressed(VK_R, prev_keys):
            frozen      = frame.copy()
            frozen_disp = cv2.resize(frozen, (disp_w, disp_h))
            rect = freeze_and_select(frozen_disp, WIN, 0, 0, scale)
            if rect:
                x1, y1, x2, y2 = rect
                crop = frozen[y1:y2, x1:x2]
                idx  = _next_index(REJECTS_DIR, "rej_")
                path = os.path.join(REJECTS_DIR, f"rej_{idx:03d}.png")
                _imwrite(path, crop)
                status_msg = f"✓ 리젝트 저장: rej_{idx:03d}.png ({crop.shape[1]}x{crop.shape[0]}px)"
                print(f"[리젝트] {path}  size={crop.shape[1]}x{crop.shape[0]}")
            else:
                status_msg = "취소됨"

        # ── Q : 종료 ──────────────────────────────────────────────
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
