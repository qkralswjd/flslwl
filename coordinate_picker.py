"""좌표 세팅 도구 (Coordinate Picker)

사용법:
    드래그         : 영역(region) 선택
    X 키           : 현재 마우스 위치를 단일 좌표로 즉시 캡처
    Enter          : 선택 확정 → 이름 입력 → 저장
    ESC            : 취소 / 다시 선택
    Q              : 종료

저장 위치:
    config/saved_coords.json       ← 전체 좌표 목록
    config/config_automation.json  ← [config 적용] 버튼으로 자동 반영

실행:
    py -3.11 coordinate_picker.py
"""

import ctypes
import ctypes.wintypes
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

import mss
import numpy as np
import cv2

# ── 경로 설정 ──────────────────────────────────────────────────────────
HERE                   = os.path.dirname(os.path.abspath(__file__))
AUTOMATION_CONFIG_PATH = os.path.join(HERE, "config", "config_automation.json")
COORDS_SAVE_PATH       = os.path.join(HERE, "config", "saved_coords.json")

# 게임이 실행 중인 모니터 인덱스 (mss 기준: 1=왼쪽, 2=오른쪽/주모니터)
MONITOR_INDEX = 2


# ── 저장/로드 ─────────────────────────────────────────────────────────

def load_saved_coords() -> dict:
    if not os.path.exists(COORDS_SAVE_PATH):
        return {}
    with open(COORDS_SAVE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_coords(coords: dict):
    os.makedirs(os.path.dirname(COORDS_SAVE_PATH), exist_ok=True)
    with open(COORDS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(coords, f, ensure_ascii=False, indent=4)


def load_automation_config() -> dict:
    if not os.path.exists(AUTOMATION_CONFIG_PATH):
        return {}
    with open(AUTOMATION_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_automation_config(cfg: dict):
    os.makedirs(os.path.dirname(AUTOMATION_CONFIG_PATH), exist_ok=True)
    with open(AUTOMATION_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


# ── 마우스 절대 좌표 읽기 ─────────────────────────────────────────────

def get_cursor_pos() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


# ── config 키 매핑 ────────────────────────────────────────────────────
# (표시이름) → (config 최상위 키, 하위 키)
CONFIG_KEY_MAP = {
    "HP 바 영역":           ("hp_bar",       "region"),
    "레벨 OCR 영역":        ("level_ocr",    "region"),
    "두루마리 목적지 창":   ("scroll_dummy", "destination_region"),
    "허수아비 공격 좌표":   ("dummy",        "attack_coord"),
    "사냥터 웨이포인트 추가": ("hunt_waypoints", "points"),   # 특수 처리
    "아데나 스캔 영역":     ("loot",         "scan_region"),
    "캡처 오프셋":          ("capture_offset", None),
    "기타 (저장만)":        (None,           None),
}


# ══════════════════════════════════════════════════════════════════════
#  오버레이 피커
# ══════════════════════════════════════════════════════════════════════

class OverlayPicker:
    """전체화면 반투명 오버레이로 좌표/영역을 선택합니다.

    - 드래그  : 영역(region) 반환
    - X 키    : 현재 마우스 단일 좌표 반환
    - Enter   : 선택 확정
    - ESC     : 취소 (None 반환)
    """

    WIN_TITLE = "좌표 피커 — 드래그:영역  |  X:단일좌표  |  Enter:확정  |  ESC:취소"

    def __init__(self):
        with mss.mss() as sct:
            mon = sct.monitors[MONITOR_INDEX]
            self._mon_left   = mon["left"]
            self._mon_top    = mon["top"]
            self._mon_width  = mon["width"]
            self._mon_height = mon["height"]

    def _grab_screen(self) -> np.ndarray:
        with mss.mss() as sct:
            mon = sct.monitors[MONITOR_INDEX]
            shot = sct.grab(mon)
            frame = np.asarray(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def pick(self) -> dict | None:
        """오버레이를 띄우고 선택 결과를 반환합니다.

        Returns:
            {"type": "point",  "x": int, "y": int}
            {"type": "region", "x": int, "y": int, "width": int, "height": int}
            None: 취소
        """
        screenshot = self._grab_screen()
        h, w = screenshot.shape[:2]

        # 어두운 오버레이
        dark = (screenshot * 0.40).astype(np.uint8)

        ox = self._mon_left
        oy = self._mon_top

        # ── OpenCV 창을 Primary 모니터(mss monitors[1]) 위치에 강제 배치 ──
        # 듀얼모니터 환경에서 창이 엉뚱한 모니터에 뜨는 것을 방지
        cv2.namedWindow(self.WIN_TITLE, cv2.WINDOW_NORMAL)
        # 1) 먼저 창을 Primary 모니터의 좌상단으로 이동
        cv2.moveWindow(self.WIN_TITLE, self._mon_left, self._mon_top)
        # 2) 그 다음 전체화면 전환 (이렇게 해야 올바른 모니터에서 전체화면)
        cv2.setWindowProperty(self.WIN_TITLE, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        state = {
            "hover":       (0, 0),     # (로컬 x, y)
            "drag_start":  None,       # (로컬 x, y)
            "drag_cur":    None,       # (로컬 x, y) — 드래그 중 현재
            "dragging":    False,
            "captured":    None,       # 확정된 결과 dict
        }

        def mouse_cb(event, x, y, flags, param):
            ax = x + ox
            ay = y + oy
            state["hover"] = (x, y)

            if event == cv2.EVENT_LBUTTONDOWN:
                state["drag_start"] = (x, y)
                state["drag_cur"]   = (x, y)
                state["dragging"]   = True
                state["captured"]   = None   # 재선택 시 초기화

            elif event == cv2.EVENT_MOUSEMOVE:
                if state["dragging"]:
                    state["drag_cur"] = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                if state["dragging"]:
                    state["drag_cur"] = (x, y)
                    state["dragging"] = False
                    sx, sy = state["drag_start"]
                    ex, ey = state["drag_cur"]
                    dw = abs(ex - sx)
                    dh = abs(ey - sy)
                    if dw > 5 or dh > 5:
                        # 영역 선택 — 절대 좌표로 저장
                        state["captured"] = {
                            "type":   "region",
                            "x":      min(sx, ex) + ox,
                            "y":      min(sy, ey) + oy,
                            "width":  dw,
                            "height": dh,
                        }
                    else:
                        # 클릭 = 단일 좌표
                        state["captured"] = {
                            "type": "point",
                            "x":    ax,
                            "y":    ay,
                        }

        cv2.setMouseCallback(self.WIN_TITLE, mouse_cb)

        result = None

        while True:
            display = dark.copy()

            lx, ly = state["hover"]
            ax_cur = lx + ox
            ay_cur = ly + oy

            # ── 가이드 텍스트 ──────────────────────────────────────────
            lines = [
                "드래그 : 영역(region) 선택",
                "X 키   : 현재 마우스 위치 단일 좌표",
                "Enter  : 선택 확정 후 이름 저장",
                "ESC    : 취소",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(display, txt, (20, 32 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                            (255, 255, 80), 2, cv2.LINE_AA)

            # ── 십자선 ────────────────────────────────────────────────
            cv2.line(display, (lx, 0),  (lx, h),  (0, 255, 0), 1)
            cv2.line(display, (0,  ly), (w,  ly), (0, 255, 0), 1)

            # ── 현재 좌표 표시 (하단) ─────────────────────────────────
            coord_txt = f"X: {ax_cur}   Y: {ay_cur}"
            cv2.putText(display, coord_txt, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2, cv2.LINE_AA)

            # ── 드래그 중 영역 미리보기 ───────────────────────────────
            if state["dragging"] and state["drag_start"] and state["drag_cur"]:
                sx, sy = state["drag_start"]
                ex, ey = state["drag_cur"]
                x0, y0 = min(sx, ex), min(sy, ey)
                x1, y1 = max(sx, ex), max(sy, ey)
                # 선택 영역은 밝게
                display[y0:y1, x0:x1] = screenshot[y0:y1, x0:x1]
                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 200, 255), 2)
                size_txt = f"{x1-x0} x {y1-y0}  |  ({x0+ox},{y0+oy})"
                cv2.putText(display, size_txt, (x0 + 4, max(y0 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 200, 255), 2, cv2.LINE_AA)

            # ── 선택 완료 표시 ────────────────────────────────────────
            cap = state["captured"]
            if cap:
                if cap["type"] == "point":
                    px = cap["x"] - ox
                    py = cap["y"] - oy
                    cv2.circle(display, (px, py), 12, (0, 255, 100), -1)
                    cv2.circle(display, (px, py), 12, (255, 255, 255), 2)
                    cv2.putText(display,
                                f"  ({cap['x']}, {cap['y']})",
                                (px + 16, py + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 100), 2, cv2.LINE_AA)
                else:
                    rx0 = cap["x"] - ox
                    ry0 = cap["y"] - oy
                    rx1 = rx0 + cap["width"]
                    ry1 = ry0 + cap["height"]
                    display[ry0:ry1, rx0:rx1] = screenshot[ry0:ry1, rx0:rx1]
                    cv2.rectangle(display, (rx0, ry0), (rx1, ry1),
                                  (0, 255, 100), 2)
                    info = (f"  ({cap['x']},{cap['y']})  "
                            f"{cap['width']}x{cap['height']}")
                    cv2.putText(display, info,
                                (rx0 + 4, max(ry0 - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 100), 2, cv2.LINE_AA)

                cv2.putText(display,
                            "Enter: 확정 저장   |   다시 드래그/클릭: 재선택",
                            (20, h - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 200), 2, cv2.LINE_AA)

            cv2.imshow(self.WIN_TITLE, display)
            key = cv2.waitKey(16) & 0xFF

            # X 키 → 현재 마우스 좌표를 단일 좌표로 즉시 캡처
            if key in (ord('x'), ord('X')):
                mx, my = get_cursor_pos()
                state["captured"] = {"type": "point", "x": mx, "y": my}
                state["dragging"] = False

            # Enter → 확정
            elif key == 13:
                if state["captured"]:
                    result = state["captured"]
                    break
                # 선택 없으면 무시

            # ESC → 취소
            elif key == 27:
                result = None
                break

            # Q → 종료
            elif key in (ord('q'), ord('Q')):
                result = None
                break

        cv2.destroyAllWindows()
        return result


# ══════════════════════════════════════════════════════════════════════
#  좌표 관리 메인 UI
# ══════════════════════════════════════════════════════════════════════

class CoordManagerUI:
    """저장된 좌표 목록 관리 + 새 좌표 추가 UI."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("📍 좌표 세팅 도구  (모니터 1 고정)")
        self.root.resizable(True, True)

        self.coords   = load_saved_coords()
        self.auto_cfg = load_automation_config()
        self._picker  = OverlayPicker()

        # ── Tkinter 창을 Primary 모니터(mss monitors[1]) 위치에 강제 배치 ──
        # 듀얼모니터에서 창이 모니터2에 뜨는 것을 방지
        with mss.mss() as sct:
            mon = sct.monitors[MONITOR_INDEX]
            mon_left = mon["left"]
            mon_top  = mon["top"]
        # 창 크기 설정 후 Primary 모니터 좌상단 기준으로 배치
        self.root.geometry(f"820x560+{mon_left + 40}+{mon_top + 40}")

        self._build_ui()
        self._refresh_list()

    # ── UI 구성 ────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── 안내 ──────────────────────────────────────────────────────
        info = (
            "📌 사용법\n"
            "  [➕ 새 좌표 추가]  버튼 클릭 → 화면에서 드래그(영역) 또는 X키(단일좌표) → Enter 확정 → 이름 입력\n"
            "  [⚙️ config 적용]   목록에서 선택 + 아래 적용 위치 콤보 설정 후 클릭 → config_automation.json 저장\n"
            "  [🔄 전체 일괄 적용] config_key가 지정된 모든 항목 한 번에 반영"
        )
        ttk.Label(self.root, text=info, justify="left",
                  foreground="#444", font=("", 9)).grid(
            row=0, column=0, columnspan=4, sticky="w", **pad)

        ttk.Separator(self.root, orient="horizontal").grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=2)

        # ── 좌표 목록 ─────────────────────────────────────────────────
        lf = ttk.LabelFrame(self.root, text="저장된 좌표 목록", padding=6)
        lf.grid(row=2, column=0, columnspan=4, sticky="nsew", **pad)
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)

        cols = ("이름", "타입", "값", "config 적용 위치")
        self.tree = ttk.Treeview(lf, columns=cols, show="headings", height=14,
                                 selectmode="browse")
        widths = [170, 65, 340, 200]
        for col, wd in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=wd, anchor="w")

        vsb = ttk.Scrollbar(lf, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(lf, orient="horizontal",  command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        # ── 버튼 행 ───────────────────────────────────────────────────
        bf = ttk.Frame(self.root, padding=4)
        bf.grid(row=3, column=0, columnspan=4, sticky="ew", padx=8)

        btn_defs = [
            ("➕ 새 좌표 추가",    self.add_coord),
            ("✏️ 이름 변경",       self.rename_coord),
            ("🗑️ 삭제",           self.delete_coord),
            ("⚙️ config 적용",    self.apply_to_config),
            ("🔄 전체 일괄 적용",  self.apply_all_to_config),
        ]
        for i, (label, cmd) in enumerate(btn_defs):
            ttk.Button(bf, text=label, command=cmd, width=16).grid(
                row=0, column=i, padx=3)

        # ── config 적용 위치 콤보 ─────────────────────────────────────
        cf = ttk.LabelFrame(self.root,
                            text="config_automation.json 적용 위치", padding=6)
        cf.grid(row=4, column=0, columnspan=4, sticky="ew", **pad)

        self._cfg_key_var = tk.StringVar(value=list(CONFIG_KEY_MAP.keys())[0])
        ttk.Label(cf, text="적용 위치:").grid(row=0, column=0, sticky="w")
        self._cfg_combo = ttk.Combobox(
            cf, textvariable=self._cfg_key_var,
            values=list(CONFIG_KEY_MAP.keys()),
            state="readonly", width=28,
        )
        self._cfg_combo.grid(row=0, column=1, padx=6)
        ttk.Label(cf,
                  text="← 목록에서 항목 선택 후 [⚙️ config 적용] 클릭",
                  foreground="gray").grid(row=0, column=2, padx=6, sticky="w")

        # ── 상태바 ────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="준비")
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="blue", font=("", 9)).grid(
            row=5, column=0, columnspan=4, sticky="w", padx=8, pady=4)

    # ── 목록 갱신 ──────────────────────────────────────────────────────

    def _refresh_list(self):
        sel = self.tree.selection()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for name, data in self.coords.items():
            typ = data.get("type", "?")
            val = _fmt(data)
            cfg = data.get("config_key", "-")
            self.tree.insert("", "end", iid=name,
                             values=(name, typ, val, cfg))
        # 선택 복원
        if sel and sel[0] in self.coords:
            self.tree.selection_set(sel[0])

    # ── 새 좌표 추가 ───────────────────────────────────────────────────

    def add_coord(self):
        """오버레이 피커를 열고 결과를 이름 입력 후 저장."""
        self.root.withdraw()
        self.status_var.set("화면에서 드래그(영역) 또는 X키(단일좌표) 선택 후 Enter...")

        def _do():
            result = self._picker.pick()
            self.root.deiconify()

            if result is None:
                self.status_var.set("취소됨")
                return

            # Tkinter 다이얼로그는 메인 스레드에서
            self.root.after(0, lambda: self._ask_name_and_save(result))

        threading.Thread(target=_do, daemon=True).start()

    def _ask_name_and_save(self, result: dict):
        """이름 입력 대화상자 → 저장."""
        preview = _fmt(result)
        name = simpledialog.askstring(
            "좌표 이름 입력",
            f"저장할 이름을 입력하세요:\n\n{preview}",
            parent=self.root,
        )
        if not name or not name.strip():
            self.status_var.set("이름 미입력 — 취소됨")
            return

        name = name.strip()
        self.coords[name] = result
        save_coords(self.coords)
        self._refresh_list()
        # 방금 추가한 항목 선택
        if name in self.coords:
            self.tree.selection_set(name)
            self.tree.see(name)
        self.status_var.set(f"✅ '{name}' 저장 완료: {preview}")

    # ── 이름 변경 ──────────────────────────────────────────────────────

    def rename_coord(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("선택 없음", "이름을 바꿀 항목을 선택하세요.")
            return
        old = sel[0]
        new = simpledialog.askstring(
            "이름 변경", f"'{old}' → 새 이름:", parent=self.root)
        if not new or not new.strip():
            return
        new = new.strip()
        self.coords[new] = self.coords.pop(old)
        save_coords(self.coords)
        self._refresh_list()
        self.status_var.set(f"✅ '{old}' → '{new}' 이름 변경됨")

    # ── 삭제 ───────────────────────────────────────────────────────────

    def delete_coord(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("선택 없음", "삭제할 항목을 선택하세요.")
            return
        name = sel[0]
        if messagebox.askyesno("삭제 확인", f"'{name}' 을(를) 삭제할까요?",
                               parent=self.root):
            del self.coords[name]
            save_coords(self.coords)
            self._refresh_list()
            self.status_var.set(f"🗑️ '{name}' 삭제됨")

    # ── config 적용 ────────────────────────────────────────────────────

    def apply_to_config(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("선택 없음", "적용할 항목을 선택하세요.")
            return
        name    = sel[0]
        data    = self.coords[name]
        cfg_key = self._cfg_key_var.get()
        self._write_to_config(name, data, cfg_key)

    def apply_all_to_config(self):
        """config_key가 지정된 모든 좌표 일괄 적용."""
        count = 0
        for name, data in self.coords.items():
            cfg_key = data.get("config_key")
            if not cfg_key or cfg_key == "-":
                continue
            if cfg_key in CONFIG_KEY_MAP:
                self._write_to_config(name, data, cfg_key)
                count += 1
        self.status_var.set(f"✅ {count}개 항목 일괄 적용 완료")

    def _write_to_config(self, name: str, data: dict, cfg_key: str):
        """data를 cfg_key 위치의 config_automation.json에 씁니다."""
        ac = self.auto_cfg
        parent_key, child_key = CONFIG_KEY_MAP.get(cfg_key, (None, None))

        if parent_key is None:
            # 기타 → config 반영 없이 이름만 저장
            pass

        elif cfg_key == "사냥터 웨이포인트 추가":
            if data["type"] != "point":
                messagebox.showerror("타입 오류",
                                     "웨이포인트는 단일 좌표(point)여야 합니다.")
                return
            hwp = ac.setdefault("hunt_waypoints", {})
            pts = hwp.setdefault("points", [])
            # 같은 이름이 있으면 덮어쓰기, 없으면 추가
            for i, pt in enumerate(pts):
                if pt.get("label") == name:
                    pts[i] = {"x": data["x"], "y": data["y"],
                               "label": name, "wait_ms": 2000}
                    break
            else:
                pts.append({"x": data["x"], "y": data["y"],
                             "label": name, "wait_ms": 2000})

        elif cfg_key == "캡처 오프셋":
            if data["type"] != "point":
                messagebox.showerror("타입 오류",
                                     "캡처 오프셋은 단일 좌표(point)여야 합니다.")
                return
            ac["capture_offset"] = {"x": data["x"], "y": data["y"]}

        elif child_key is None:
            ac[parent_key] = _to_cfg_val(data)

        else:
            node = ac.setdefault(parent_key, {})
            node[child_key] = _to_cfg_val(data)

        # config_key 메타 기록 (목록 표시용)
        self.coords[name]["config_key"] = cfg_key
        save_coords(self.coords)
        save_automation_config(ac)
        self.auto_cfg = ac
        self._refresh_list()
        self.status_var.set(
            f"✅ '{name}' → [{cfg_key}] 적용 완료  (config_automation.json 저장됨)")

    # ── 실행 ───────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ── 헬퍼 ─────────────────────────────────────────────────────────────

def _fmt(data: dict) -> str:
    if data.get("type") == "point":
        return f"X={data['x']},  Y={data['y']}"
    return (f"X={data.get('x',0)},  Y={data.get('y',0)},  "
            f"W={data.get('width',0)},  H={data.get('height',0)}")


def _to_cfg_val(data: dict) -> dict:
    if data.get("type") == "point":
        return {"x": data["x"], "y": data["y"]}
    return {"x": data["x"], "y": data["y"],
            "width": data["width"], "height": data["height"]}


# ── 진입점 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    CoordManagerUI().run()
