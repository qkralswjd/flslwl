"""좌표 세팅 도구 (Coordinate Picker)

기능:
    - 전체화면 반투명 오버레이
    - X 키 : 현재 마우스 위치 좌표 캡처
    - 드래그 : 영역(region) 선택
    - 이름 지정 후 저장
    - 저장된 좌표는 config/config_automation.json에 자동 반영

조작법:
    X         : 현재 마우스 위치를 좌표로 저장
    드래그    : 영역(region) 선택
    ESC       : 취소
    Enter     : 선택 확정 후 이름 입력
    D         : 저장된 좌표 목록 보기/삭제
    Q         : 종료

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
HERE = os.path.dirname(os.path.abspath(__file__))
AUTOMATION_CONFIG_PATH = os.path.join(HERE, "config", "config_automation.json")
COORDS_SAVE_PATH       = os.path.join(HERE, "config", "saved_coords.json")


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


# ── 좌표 타입 정의 ────────────────────────────────────────────────────

COORD_TYPES = {
    "point":  "단일 좌표 (클릭 위치)",
    "region": "영역 (드래그로 선택)",
}

# config_automation.json 키 매핑
CONFIG_KEY_MAP = {
    "레벨 OCR 영역":        ("level_ocr", "region"),
    "텔레포트 창 영역":     ("teleport", "destination_region"),
    "허수아비 공격 좌표":   ("dummy", "attack_coord"),
    "아데나 스캔 영역":     ("loot", "scan_region"),
    "웨이포인트 추가":      ("waypoints", None),   # 특수 처리
    "캡처 오프셋":          ("capture_offset", None),
    "기타 (저장만)":        (None, None),
}


# ══════════════════════════════════════════════════════════════════════
#  메인 코디네이트 피커 창
# ══════════════════════════════════════════════════════════════════════

class CoordinatePicker:
    """전체화면 반투명 오버레이로 좌표/영역을 선택합니다."""

    def __init__(self, monitor_index: int = 0):
        self.monitor_index = monitor_index
        self._result       = None   # {"type": "point"|"region", "data": {...}}
        self._running      = False

    def capture_screenshot(self) -> np.ndarray:
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor_index]
            shot = sct.grab(mon)
            frame = np.asarray(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def pick(self) -> dict | None:
        """오버레이를 띄우고 선택 결과를 반환합니다.

        Returns:
            {"type": "point", "x": int, "y": int}
            {"type": "region", "x": int, "y": int, "width": int, "height": int}
            None: 취소
        """
        # 스크린샷 캡처
        screenshot = self.capture_screenshot()
        h, w = screenshot.shape[:2]

        # 어두운 오버레이 합성
        overlay = screenshot.copy()
        dark    = (overlay * 0.45).astype(np.uint8)

        # OpenCV 창
        WIN = "CoordPicker — X:단일좌표 | 드래그:영역 | ESC:취소 | Enter:확정"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        state = {
            "drag_start": None,
            "drag_end":   None,
            "dragging":   False,
            "hover":      (0, 0),
            "mode":       None,   # "point" | "region"
            "captured":   None,
        }

        def mouse_cb(event, x, y, flags, param):
            with mss.mss() as sct:
                mon = sct.monitors[self.monitor_index]
                ox, oy = mon["left"], mon["top"]

            abs_x = x + ox
            abs_y = y + oy
            state["hover"] = (abs_x, abs_y)

            if event == cv2.EVENT_LBUTTONDOWN:
                state["drag_start"] = (x, y)
                state["dragging"]   = True

            elif event == cv2.EVENT_MOUSEMOVE:
                if state["dragging"]:
                    state["drag_end"] = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                if state["dragging"]:
                    state["drag_end"] = (x, y)
                    state["dragging"] = False
                    sx, sy = state["drag_start"]
                    ex, ey = state["drag_end"]
                    if abs(ex - sx) > 5 or abs(ey - sy) > 5:
                        rx  = min(sx, ex) + ox
                        ry  = min(sy, ey) + oy
                        rw  = abs(ex - sx)
                        rh  = abs(ey - sy)
                        state["captured"] = {
                            "type":   "region",
                            "x":      rx, "y": ry,
                            "width":  rw, "height": rh,
                        }
                    else:
                        state["captured"] = {
                            "type": "point",
                            "x":    abs_x, "y": abs_y,
                        }

        cv2.setMouseCallback(WIN, mouse_cb)

        result = None
        while True:
            display = dark.copy()

            # 가이드 텍스트
            guides = [
                "X 키 : 현재 마우스 위치 저장",
                "드래그 : 영역 선택",
                "Enter : 선택 확정",
                "ESC : 취소",
            ]
            for i, g in enumerate(guides):
                cv2.putText(display, g, (20, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)

            # 마우스 좌표 표시
            hx, hy = state["hover"]
            coord_txt = f"X: {hx}   Y: {hy}"
            cv2.putText(display, coord_txt, (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 십자선
            with mss.mss() as sct:
                mon = sct.monitors[self.monitor_index]
                ox, oy = mon["left"], mon["top"]
            lx = hx - ox
            ly = hy - oy
            cv2.line(display, (lx, 0),    (lx, h),    (0, 255, 0), 1)
            cv2.line(display, (0,  ly),   (w,  ly),   (0, 255, 0), 1)

            # 드래그 중 영역 표시
            if state["dragging"] and state["drag_start"] and state["drag_end"]:
                sx, sy = state["drag_start"]
                ex, ey = state["drag_end"]
                x0, y0 = min(sx, ex), min(sy, ey)
                x1, y1 = max(sx, ex), max(sy, ey)
                # 밝게 원본 복원
                display[y0:y1, x0:x1] = screenshot[y0:y1, x0:x1]
                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 200, 255), 2)
                size_txt = f"{abs(x1-x0)} x {abs(y1-y0)}"
                cv2.putText(display, size_txt, (x0+4, y0-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # 선택 완료 표시
            if state["captured"]:
                cap = state["captured"]
                if cap["type"] == "point":
                    px = cap["x"] - ox
                    py = cap["y"] - oy
                    cv2.circle(display, (px, py), 10, (0, 255, 0), -1)
                    cv2.putText(display, f"({cap['x']}, {cap['y']})",
                                (px + 14, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    rx0 = cap["x"] - ox
                    ry0 = cap["y"] - oy
                    rx1 = rx0 + cap["width"]
                    ry1 = ry0 + cap["height"]
                    display[ry0:ry1, rx0:rx1] = screenshot[ry0:ry1, rx0:rx1]
                    cv2.rectangle(display, (rx0, ry0), (rx1, ry1), (0, 255, 0), 2)
                confirm_txt = "Enter: 확정  |  다시 드래그하면 재선택"
                cv2.putText(display, confirm_txt, (20, h - 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)

            cv2.imshow(WIN, display)
            key = cv2.waitKey(16) & 0xFF

            # X 키 → 현재 마우스 좌표 즉시 저장
            if key == ord('x') or key == ord('X'):
                mx, my = get_cursor_pos()
                state["captured"] = {"type": "point", "x": mx, "y": my}

            # Enter → 확정
            elif key == 13:
                if state["captured"]:
                    result = state["captured"]
                    break

            # ESC → 취소
            elif key == 27:
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
        self.root.title("📍 좌표 세팅 도구")
        self.root.resizable(False, False)

        self.coords         = load_saved_coords()
        self.auto_cfg       = load_automation_config()
        self._picker        = CoordinatePicker(monitor_index=0)

        self._build_ui()
        self._refresh_list()

    # ── UI 구성 ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── 상단: 안내 ────────────────────────────────────────────────
        info = (
            "📌 사용법:\n"
            "  [새 좌표 추가] → 화면에서 X키(단일) 또는 드래그(영역) 선택\n"
            "  [config 적용]  → 선택한 항목을 config_automation.json에 저장\n"
            "  [삭제]         → 선택한 좌표 삭제"
        )
        ttk.Label(self.root, text=info, justify="left",
                  foreground="#333").grid(row=0, column=0, columnspan=3,
                                         padx=10, pady=8, sticky="w")

        # ── 좌표 목록 ─────────────────────────────────────────────────
        list_frame = ttk.LabelFrame(self.root, text="저장된 좌표 목록", padding=6)
        list_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=4, sticky="nsew")

        cols = ("이름", "타입", "값", "config 키")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=12)
        for col in cols:
            self.tree.heading(col, text=col)
        self.tree.column("이름",     width=160)
        self.tree.column("타입",     width=70)
        self.tree.column("값",       width=280)
        self.tree.column("config 키", width=200)

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        # ── 버튼 ──────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root, padding=4)
        btn_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=6)

        ttk.Button(btn_frame, text="➕ 새 좌표 추가",
                   command=self.add_coord).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="✏️ 이름 변경",
                   command=self.rename_coord).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text="🗑️ 삭제",
                   command=self.delete_coord).grid(row=0, column=2, padx=4)
        ttk.Button(btn_frame, text="⚙️ config 적용",
                   command=self.apply_to_config).grid(row=0, column=3, padx=4)
        ttk.Button(btn_frame, text="🔄 전체 config 반영",
                   command=self.apply_all_to_config).grid(row=0, column=4, padx=4)

        # ── config 키 선택 ────────────────────────────────────────────
        key_frame = ttk.LabelFrame(self.root, text="config_automation.json 적용 위치", padding=6)
        key_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=4, sticky="ew")

        self._config_key_var = tk.StringVar(value=list(CONFIG_KEY_MAP.keys())[0])
        ttk.Label(key_frame, text="적용 위치:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            key_frame, textvariable=self._config_key_var,
            values=list(CONFIG_KEY_MAP.keys()),
            state="readonly", width=25
        ).grid(row=0, column=1, padx=6)
        ttk.Label(key_frame,
                  text="→ [config 적용] 버튼으로 선택한 좌표를 해당 위치에 저장",
                  foreground="gray").grid(row=0, column=2, padx=6, sticky="w")

        # ── 상태바 ────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="준비")
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="blue").grid(row=4, column=0, columnspan=3,
                                          sticky="w", padx=10, pady=4)

    # ── 목록 갱신 ──────────────────────────────────────────────────────

    def _refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for name, data in self.coords.items():
            typ = data.get("type", "?")
            if typ == "point":
                val = f"X={data['x']}, Y={data['y']}"
            else:
                val = (f"X={data['x']}, Y={data['y']}, "
                       f"W={data['width']}, H={data['height']}")
            cfg_key = data.get("config_key", "-")
            self.tree.insert("", "end", iid=name,
                             values=(name, typ, val, cfg_key))

    # ── 좌표 추가 ──────────────────────────────────────────────────────

    def add_coord(self):
        self.root.withdraw()   # 메인 창 숨김
        self.status_var.set("화면에서 X키(단일) 또는 드래그(영역)로 선택하세요...")

        def _pick():
            result = self._picker.pick()
            self.root.deiconify()  # 메인 창 복원

            if result is None:
                self.status_var.set("취소됨")
                return

            # 이름 입력
            name = simpledialog.askstring(
                "좌표 이름",
                f"이 좌표의 이름을 입력하세요:\n"
                f"({result['type']}: {_fmt(result)})",
                parent=self.root,
            )
            if not name or not name.strip():
                self.status_var.set("이름 미입력 — 취소됨")
                return

            name = name.strip()
            self.coords[name] = result
            save_coords(self.coords)
            self._refresh_list()
            self.status_var.set(f"✅ '{name}' 저장 완료: {_fmt(result)}")

        threading.Thread(target=_pick, daemon=True).start()

    # ── 이름 변경 ──────────────────────────────────────────────────────

    def rename_coord(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("선택 없음", "이름을 바꿀 좌표를 선택하세요.")
            return
        old_name = sel[0]
        new_name = simpledialog.askstring(
            "이름 변경", f"'{old_name}' → 새 이름:", parent=self.root
        )
        if not new_name or not new_name.strip():
            return
        new_name = new_name.strip()
        self.coords[new_name] = self.coords.pop(old_name)
        save_coords(self.coords)
        self._refresh_list()
        self.status_var.set(f"✅ '{old_name}' → '{new_name}' 변경됨")

    # ── 삭제 ───────────────────────────────────────────────────────────

    def delete_coord(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("선택 없음", "삭제할 좌표를 선택하세요.")
            return
        name = sel[0]
        if messagebox.askyesno("삭제 확인", f"'{name}' 을(를) 삭제할까요?"):
            del self.coords[name]
            save_coords(self.coords)
            self._refresh_list()
            self.status_var.set(f"🗑️ '{name}' 삭제됨")

    # ── config 적용 (선택된 항목 → config_automation.json) ─────────────

    def apply_to_config(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("선택 없음", "적용할 좌표를 선택하세요.")
            return
        name     = sel[0]
        data     = self.coords[name]
        cfg_key  = self._config_key_var.get()
        keys     = CONFIG_KEY_MAP.get(cfg_key, (None, None))

        self._write_to_config(data, cfg_key, keys, name)

    def _write_to_config(self, data, cfg_key_name, keys, coord_name):
        parent_key, child_key = keys
        ac = self.auto_cfg

        if parent_key is None:
            # 기타 → 저장만
            pass

        elif cfg_key_name == "웨이포인트 추가":
            # 웨이포인트 리스트에 추가
            if data["type"] != "point":
                messagebox.showerror("타입 오류", "웨이포인트는 단일 좌표(point)여야 합니다.")
                return
            wps = ac.setdefault("waypoints", [])
            wps.append({
                "x":       data["x"],
                "y":       data["y"],
                "label":   coord_name,
                "wait_ms": 1500,
            })

        elif cfg_key_name == "캡처 오프셋":
            if data["type"] != "point":
                messagebox.showerror("타입 오류", "캡처 오프셋은 단일 좌표(point)여야 합니다.")
                return
            ac["capture_offset"] = {"x": data["x"], "y": data["y"]}

        elif child_key is None:
            # parent만 있고 child 없는 경우
            ac[parent_key] = _to_config_value(data)

        else:
            # 중첩 dict
            node = ac.setdefault(parent_key, {})
            if data["type"] == "point":
                node[child_key] = {"x": data["x"], "y": data["y"]}
            else:
                node[child_key] = {
                    "x": data["x"], "y": data["y"],
                    "width": data["width"], "height": data["height"]
                }

        # config_key 메타 저장 (표시용)
        self.coords[coord_name]["config_key"] = cfg_key_name
        save_coords(self.coords)
        save_automation_config(ac)
        self.auto_cfg = ac
        self._refresh_list()
        self.status_var.set(
            f"✅ '{coord_name}' → [{cfg_key_name}] 적용 완료 (config_automation.json 저장)"
        )

    def apply_all_to_config(self):
        """config_key가 지정된 모든 좌표를 일괄 적용."""
        count = 0
        for name, data in self.coords.items():
            cfg_key = data.get("config_key")
            if not cfg_key or cfg_key == "-":
                continue
            keys = CONFIG_KEY_MAP.get(cfg_key)
            if keys:
                self._write_to_config(data, cfg_key, keys, name)
                count += 1
        self.status_var.set(f"✅ {count}개 좌표 전체 적용 완료")

    # ── 실행 ───────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ── 헬퍼 ─────────────────────────────────────────────────────────────

def _fmt(data: dict) -> str:
    if data["type"] == "point":
        return f"X={data['x']}, Y={data['y']}"
    return f"X={data['x']}, Y={data['y']}, W={data['width']}, H={data['height']}"


def _to_config_value(data: dict) -> dict:
    if data["type"] == "point":
        return {"x": data["x"], "y": data["y"]}
    return {"x": data["x"], "y": data["y"],
            "width": data["width"], "height": data["height"]}


# ── 진입점 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    CoordManagerUI().run()
