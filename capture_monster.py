"""몬스터 템플릿 캡처 도구

사용법:
    py -3.11 capture_monster.py

조작키:
    F1        : 현재 화면 스크린샷 저장 (screenshots/cap_NNN.png)
    마우스 드래그 : 몬스터 영역 선택 (빨간 사각형)
    T         : 선택 영역을 템플릿으로 저장 (config/templates/mob_NNN.png)
    R         : 선택 영역을 리젝트 템플릿으로 저장 (config/templates_reject/)
    C         : 선택 영역 취소
    Q / ESC   : 종료

저장 위치:
    screenshots/        : 원본 스크린샷
    config/templates/   : 몬스터 템플릿 (T키)
    config/templates_reject/ : 오탐지 리젝트 템플릿 (R키)
"""

import ctypes
import ctypes.wintypes
import json
import os
import sys
import time

import cv2
import mss
import numpy as np

# ── 경로 설정 ────────────────────────────────────────────────────────
HERE            = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH     = os.path.join(HERE, "config", "config.json")
SCREENSHOTS_DIR = os.path.join(HERE, "screenshots")
TEMPLATES_DIR   = os.path.join(HERE, "config", "templates")
REJECTS_DIR     = os.path.join(HERE, "config", "templates_reject")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR,   exist_ok=True)
os.makedirs(REJECTS_DIR,     exist_ok=True)


# ── config 로드 ──────────────────────────────────────────────────────
def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 파일 저장 (유니코드 경로 대응) ──────────────────────────────────
def _imwrite(path: str, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return ok


def _next_index(directory: str, prefix: str, ext: str = ".png") -> int:
    existing = [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(ext)
    ]
    nums = []
    for name in existing:
        try:
            nums.append(int(name[len(prefix):-len(ext)]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


# ── 윈도우 키 감지 ────────────────────────────────────────────────────
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState

VK_F1  = 0x70
VK_Q   = ord('Q')
VK_T   = ord('T')
VK_R   = ord('R')
VK_C   = ord('C')
VK_ESC = 0x1B


def key_pressed(vk: int) -> bool:
    return bool(GetAsyncKeyState(vk) & 0x8000)


def key_just_pressed(vk: int, prev: dict) -> bool:
    now = bool(GetAsyncKeyState(vk) & 0x8000)
    was = prev.get(vk, False)
    prev[vk] = now
    return now and not was


# ── 마우스 위치 ──────────────────────────────────────────────────────
GetCursorPos = ctypes.windll.user32.GetCursorPos

def get_cursor() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


GetKeyState     = ctypes.windll.user32.GetKeyState
VK_LBUTTON      = 0x01


def lmb_down() -> bool:
    return bool(GetAsyncKeyState(VK_LBUTTON) & 0x8000)


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    cfg          = _load_config()
    monitor_idx  = cfg.get("monitor_index", 2)

    sct = mss.mss()
    monitors = sct.monitors
    if monitor_idx >= len(monitors):
        print(f"[ERROR] monitor_index={monitor_idx} 범위 초과. 최대 {len(monitors)-1}")
        sys.exit(1)

    mon = monitors[monitor_idx]
    mon_left = mon["left"]
    mon_top  = mon["top"]
    mon_w    = mon["width"]
    mon_h    = mon["height"]

    print("=" * 56)
    print("  몬스터 템플릿 캡처 도구")
    print("=" * 56)
    print(f"  모니터 #{monitor_idx}  {mon_w}x{mon_h}  offset=({mon_left},{mon_top})")
    print()
    print("  F1  : 스크린샷 저장")
    print("  드래그 → T : 몬스터 템플릿 저장")
    print("  드래그 → R : 리젝트 템플릿 저장")
    print("  C   : 선택 취소")
    print("  Q / ESC : 종료")
    print()

    WIN = "capture_monster"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)

    # 드래그 상태
    drag_start  = None   # 드래그 시작 (화면 절대좌표)
    drag_end    = None
    drag_active = False
    sel_rect    = None   # (x1,y1,x2,y2) 화면 상대좌표 (모니터 기준)
    status_msg  = "대기 중 — F1로 스크린샷, 드래그로 영역 선택"

    prev_keys   = {}
    prev_lmb    = False
    last_frame  = None   # 가장 최근 grab 이미지

    while True:
        # ── 화면 캡처 ─────────────────────────────────────────────
        shot  = sct.grab(mon)
        frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
        last_frame = frame.copy()

        # ── 마우스 드래그 감지 ────────────────────────────────────
        cur_lmb = lmb_down()
        cx, cy  = get_cursor()
        # 모니터 내 상대좌표
        rx, ry  = cx - mon_left, cy - mon_top

        if cur_lmb and not prev_lmb:
            # 버튼 눌림 시작
            drag_start  = (rx, ry)
            drag_end    = (rx, ry)
            drag_active = True
            sel_rect    = None

        elif cur_lmb and drag_active:
            # 드래그 중
            drag_end = (rx, ry)

        elif not cur_lmb and prev_lmb and drag_active:
            # 버튼 뗌 → 선택 확정
            drag_end    = (rx, ry)
            drag_active = False
            x1 = min(drag_start[0], drag_end[0])
            y1 = min(drag_start[1], drag_end[1])
            x2 = max(drag_start[0], drag_end[0])
            y2 = max(drag_start[1], drag_end[1])
            w  = x2 - x1
            h  = y2 - y1
            if w > 5 and h > 5:
                sel_rect   = (x1, y1, x2, y2)
                status_msg = f"선택됨 ({x1},{y1})-({x2},{y2})  T:템플릿저장  R:리젝트저장  C:취소"
            else:
                sel_rect   = None
                status_msg = "선택 너무 작음. 다시 드래그하세요."

        prev_lmb = cur_lmb

        # ── 오버레이 그리기 ───────────────────────────────────────
        disp = frame.copy()

        # 드래그 중 실시간 사각형
        if drag_active and drag_start:
            cv2.rectangle(disp,
                          (drag_start[0], drag_start[1]),
                          (rx, ry),
                          (0, 255, 255), 1)

        # 확정된 선택 영역
        if sel_rect:
            x1, y1, x2, y2 = sel_rect
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 미리보기 (오른쪽 상단)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                ph, pw = 120, 120
                preview = cv2.resize(crop, (pw, ph))
                disp[10:10+ph, disp.shape[1]-pw-10:disp.shape[1]-10] = preview
                cv2.rectangle(disp,
                              (disp.shape[1]-pw-10, 10),
                              (disp.shape[1]-10,    10+ph),
                              (0, 0, 255), 1)

        # 상태 메시지
        cv2.putText(disp, status_msg, (10, disp.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

        # 템플릿 개수 표시
        n_tmpl = len([f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".png")])
        n_rej  = len([f for f in os.listdir(REJECTS_DIR)   if f.endswith(".png")])
        cv2.putText(disp, f"templates={n_tmpl}  rejects={n_rej}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 리사이즈해서 표시 (창이 너무 크면 조절)
        h_disp = min(disp.shape[0], 810)
        scale  = h_disp / disp.shape[0]
        disp_s = cv2.resize(disp, (int(disp.shape[1]*scale), h_disp))
        cv2.imshow(WIN, disp_s)

        # ── 키 처리 ───────────────────────────────────────────────
        cv2.waitKey(1)

        # F1 : 스크린샷 저장
        if key_just_pressed(VK_F1, prev_keys):
            idx  = _next_index(SCREENSHOTS_DIR, "cap_")
            path = os.path.join(SCREENSHOTS_DIR, f"cap_{idx:03d}.png")
            _imwrite(path, last_frame)
            status_msg = f"✓ 스크린샷 저장: {os.path.basename(path)}"
            print(f"[캡처] {path}")

        # T : 템플릿 저장
        if key_just_pressed(VK_T, prev_keys):
            if sel_rect and last_frame is not None:
                x1, y1, x2, y2 = sel_rect
                crop = last_frame[y1:y2, x1:x2]
                idx  = _next_index(TEMPLATES_DIR, "mob_")
                path = os.path.join(TEMPLATES_DIR, f"mob_{idx:03d}.png")
                _imwrite(path, crop)
                status_msg = f"✓ 템플릿 저장: {os.path.basename(path)}  ({crop.shape[1]}x{crop.shape[0]}px)"
                print(f"[템플릿] {path}  size={crop.shape[1]}x{crop.shape[0]}")
                sel_rect = None
            else:
                status_msg = "먼저 몬스터 영역을 드래그로 선택하세요."

        # R : 리젝트 저장
        if key_just_pressed(VK_R, prev_keys):
            if sel_rect and last_frame is not None:
                x1, y1, x2, y2 = sel_rect
                crop = last_frame[y1:y2, x1:x2]
                idx  = _next_index(REJECTS_DIR, "rej_")
                path = os.path.join(REJECTS_DIR, f"rej_{idx:03d}.png")
                _imwrite(path, crop)
                status_msg = f"✓ 리젝트 저장: {os.path.basename(path)}  ({crop.shape[1]}x{crop.shape[0]}px)"
                print(f"[리젝트] {path}  size={crop.shape[1]}x{crop.shape[0]}")
                sel_rect = None
            else:
                status_msg = "먼저 영역을 드래그로 선택하세요."

        # C : 선택 취소
        if key_just_pressed(VK_C, prev_keys):
            sel_rect   = None
            status_msg = "선택 취소됨"

        # Q / ESC : 종료
        if key_just_pressed(VK_Q, prev_keys) or key_just_pressed(VK_ESC, prev_keys):
            break

    cv2.destroyAllWindows()
    sct.close()
    print("[종료]")
    n_tmpl = len([f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".png")])
    print(f"  저장된 템플릿: {n_tmpl}개  ({TEMPLATES_DIR})")


if __name__ == "__main__":
    main()
