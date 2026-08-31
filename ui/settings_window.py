"""메인 컨트롤 UI

구성:
    ┌──────────────────────────────────────┐
    │  🎮 모드 선택   [⚔ 1~10레벨]  [🏰 던전]  │
    ├──────────────────────────────────────┤
    │  🔌 COM 포트:  [COM4 ▼] [⟳]           │
    ├──────────────────────────────────────┤
    │         [ ▶  시 작 ]                  │
    │         [ ■  중 지 ]                  │
    ├──────────────────────────────────────┤
    │  📊 상태                              │
    │  상태머신: IDLE                       │
    │  Lv. --  HP: --%  Kill: 0  Time: 0m  │
    │  Capture: 0.0fps  Detection: 0.0fps  │
    ├──────────────────────────────────────┤
    │  [ 📍 좌표 세팅 도구 열기 ]            │
    └──────────────────────────────────────┘
"""

import json
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main as tracker_main
from config.config_loader import DEFAULT_CONFIG_PATH, load_config, save_config

AUTOMATION_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config_automation.json")
)


def _load_automation_config() -> dict:
    if not os.path.exists(AUTOMATION_CONFIG_PATH):
        return {}
    with open(AUTOMATION_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_automation_config(cfg: dict):
    os.makedirs(os.path.dirname(AUTOMATION_CONFIG_PATH), exist_ok=True)
    with open(AUTOMATION_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


def _get_serial_ports() -> list[str]:
    try:
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return ports if ports else ["(없음)"]
    except ImportError:
        return ["(pyserial 없음)"]


# ── 색상 팔레트 ────────────────────────────────────────────────────────
C_BG        = "#1e1e2e"   # 배경
C_PANEL     = "#2a2a3e"   # 패널
C_BORDER    = "#44475a"   # 테두리
C_TEXT      = "#cdd6f4"   # 기본 텍스트
C_DIM       = "#6c7086"   # 흐린 텍스트
C_LEVELING  = "#89b4fa"   # 레벨링 모드 색
C_DUNGEON   = "#f38ba8"   # 던전 모드 색
C_START     = "#a6e3a1"   # 시작 버튼
C_STOP      = "#f38ba8"   # 중지 버튼
C_ACCENT    = "#cba6f7"   # 강조색


class SettingsWindow:

    def __init__(self):
        self.config            = load_config()
        self.automation_config = _load_automation_config()
        self.stop_event        = threading.Event()
        self.worker_thread: threading.Thread | None = None

        # 현재 모드: "leveling" | "dungeon"
        self._mode = tk.StringVar(value="leveling")

        self.root = tk.Tk()
        self.root.title("🎮 Auto Hunting Controller")
        self.root.resizable(False, False)
        self.root.configure(bg=C_BG)

        self._build_ui()

    # ══════════════════════════════════════════════════════════════════
    #  UI 구성
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = self.root
        pad  = dict(padx=14, pady=6)

        # ── 타이틀 ────────────────────────────────────────────────────
        tk.Label(
            root, text="⚡  Auto Hunting Controller",
            bg=C_BG, fg=C_ACCENT,
            font=("Segoe UI", 13, "bold"),
        ).pack(fill="x", padx=14, pady=(12, 4))

        self._sep(root)

        # ── 모드 선택 ─────────────────────────────────────────────────
        mode_frame = tk.Frame(root, bg=C_BG)
        mode_frame.pack(fill="x", **pad)

        tk.Label(mode_frame, text="모드", bg=C_BG, fg=C_DIM,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))

        self._btn_lv = tk.Button(
            mode_frame, text="⚔  1~10레벨 모드",
            font=("Segoe UI", 10, "bold"),
            bg=C_LEVELING, fg="#1e1e2e",
            activebackground=C_LEVELING, activeforeground="#1e1e2e",
            relief="flat", padx=14, pady=6, cursor="hand2",
            command=lambda: self._switch_mode("leveling"),
        )
        self._btn_lv.pack(side="left", padx=4)

        self._btn_dn = tk.Button(
            mode_frame, text="🏰  던전 사냥 모드",
            font=("Segoe UI", 10, "bold"),
            bg=C_PANEL, fg=C_DIM,
            activebackground=C_DUNGEON, activeforeground="#1e1e2e",
            relief="flat", padx=14, pady=6, cursor="hand2",
            command=lambda: self._switch_mode("dungeon"),
        )
        self._btn_dn.pack(side="left", padx=4)

        self._sep(root)

        # ── COM 포트 ──────────────────────────────────────────────────
        port_frame = tk.Frame(root, bg=C_BG)
        port_frame.pack(fill="x", **pad)

        tk.Label(port_frame, text="🔌  COM 포트", bg=C_BG, fg=C_TEXT,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))

        self._port_var = tk.StringVar(
            value=self.config.get("pico", {}).get("serial_port", "COM3")
        )
        self._port_combo = ttk.Combobox(
            port_frame, textvariable=self._port_var,
            values=_get_serial_ports(), state="readonly", width=10,
        )
        self._port_combo.pack(side="left", padx=4)

        tk.Button(
            port_frame, text="⟳", font=("Segoe UI", 9),
            bg=C_PANEL, fg=C_TEXT, relief="flat", padx=6,
            cursor="hand2", command=self._refresh_ports,
        ).pack(side="left", padx=2)

        self._sep(root)

        # ── 시작 / 중지 버튼 ─────────────────────────────────────────
        btn_frame = tk.Frame(root, bg=C_BG)
        btn_frame.pack(fill="x", padx=14, pady=8)

        self._btn_start = tk.Button(
            btn_frame, text="▶   시  작",
            font=("Segoe UI", 14, "bold"),
            bg=C_START, fg="#1e1e2e",
            activebackground="#79c99e", activeforeground="#1e1e2e",
            relief="flat", pady=10, cursor="hand2",
            command=self.start,
        )
        self._btn_start.pack(fill="x", pady=(0, 6))

        self._btn_stop = tk.Button(
            btn_frame, text="■   중  지",
            font=("Segoe UI", 14, "bold"),
            bg=C_STOP, fg="#1e1e2e",
            activebackground="#d07080", activeforeground="#1e1e2e",
            relief="flat", pady=10, cursor="hand2",
            command=self.stop,
        )
        self._btn_stop.pack(fill="x")

        self._sep(root)

        # ── 상태 패널 ─────────────────────────────────────────────────
        stat_outer = tk.Frame(root, bg=C_PANEL, bd=0)
        stat_outer.pack(fill="x", padx=14, pady=6)

        tk.Label(stat_outer, text="📊  상태", bg=C_PANEL, fg=C_DIM,
                 font=("Segoe UI", 8)).grid(row=0, column=0, columnspan=2,
                                             sticky="w", padx=10, pady=(8, 2))

        # 상태머신 상태
        self._lbl_sm = tk.Label(
            stat_outer, text="IDLE",
            bg=C_PANEL, fg=C_ACCENT,
            font=("Segoe UI", 18, "bold"),
        )
        self._lbl_sm.grid(row=1, column=0, columnspan=2,
                           sticky="w", padx=10, pady=(0, 4))

        # Lv / HP / Kill / Time
        self._lbl_info = tk.Label(
            stat_outer,
            text="Lv.--   HP: --%   Kill: 0   Time: 0 min",
            bg=C_PANEL, fg=C_TEXT,
            font=("Segoe UI", 10),
        )
        self._lbl_info.grid(row=2, column=0, columnspan=2,
                             sticky="w", padx=10, pady=(0, 4))

        # FPS
        self._lbl_fps = tk.Label(
            stat_outer,
            text="Capture: 0.0 fps   Detection: 0.0 fps   Pico: --",
            bg=C_PANEL, fg=C_DIM,
            font=("Segoe UI", 8),
        )
        self._lbl_fps.grid(row=3, column=0, columnspan=2,
                            sticky="w", padx=10, pady=(0, 8))

        self._sep(root)

        # ── 좌표 세팅 도구 ────────────────────────────────────────────
        tk.Button(
            root, text="📍  좌표 세팅 도구 열기",
            font=("Segoe UI", 10),
            bg=C_PANEL, fg=C_TEXT,
            activebackground=C_BORDER, activeforeground=C_TEXT,
            relief="flat", pady=8, cursor="hand2",
            command=self.open_coord_picker,
        ).pack(fill="x", padx=14, pady=(0, 12))

        # 초기 모드 적용
        self._switch_mode("leveling")

    def _sep(self, parent):
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill="x", padx=10, pady=2)

    # ══════════════════════════════════════════════════════════════════
    #  모드 전환
    # ══════════════════════════════════════════════════════════════════

    def _switch_mode(self, mode: str):
        self._mode.set(mode)
        if mode == "leveling":
            self._btn_lv.config(bg=C_LEVELING, fg="#1e1e2e")
            self._btn_dn.config(bg=C_PANEL,    fg=C_DIM)
        else:
            self._btn_dn.config(bg=C_DUNGEON,  fg="#1e1e2e")
            self._btn_lv.config(bg=C_PANEL,    fg=C_DIM)

    # ══════════════════════════════════════════════════════════════════
    #  포트 새로고침
    # ══════════════════════════════════════════════════════════════════

    def _refresh_ports(self):
        ports = _get_serial_ports()
        self._port_combo["values"] = ports
        if self._port_var.get() not in ports and ports:
            self._port_var.set(ports[0])

    # ══════════════════════════════════════════════════════════════════
    #  시작 / 중지
    # ══════════════════════════════════════════════════════════════════

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        # COM 포트를 config에 반영
        self.config.setdefault("pico", {})["serial_port"] = self._port_var.get()
        save_config(self.config, DEFAULT_CONFIG_PATH)

        self.stop_event.clear()
        mode = self._mode.get()

        auto_cfg = self.automation_config if mode == "leveling" else None

        def worker():
            tracker_main.run(
                self.config,
                stop_event       = self.stop_event,
                status_callback  = self._on_status,
                automation_config= auto_cfg,
                mode             = mode,
            )

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

        # 버튼 상태
        self._btn_start.config(state="disabled", bg=C_BORDER)
        self._btn_stop.config(state="normal",   bg=C_STOP)

        self._update_sm_label("시작 중...")

    def stop(self):
        self.stop_event.set()
        self._btn_start.config(state="normal",   bg=C_START)
        self._btn_stop.config(state="disabled",  bg=C_BORDER)
        self._update_sm_label("중지됨")

    # ══════════════════════════════════════════════════════════════════
    #  상태 콜백
    # ══════════════════════════════════════════════════════════════════

    def _on_status(
        self,
        enemy_count: int,
        capture_fps: float,
        detection_fps: float,
        pico_connected: bool = False,
        target_state_name: str = "",
        current_target_id=None,
        # HuntingStateMachine 추가 정보 (선택)
        sm_state: str = "",
        sm_level: int = 0,
        sm_hp_pct=None,
        sm_kills: int = 0,
        sm_elapsed_min: float = 0.0,
    ):
        # 상태머신 상태 표시
        state_txt = sm_state or target_state_name or "IDLE"
        self.root.after(0, lambda: self._update_sm_label(state_txt))

        # 정보 라인
        hp_txt   = f"{sm_hp_pct:.0f}%" if sm_hp_pct is not None else "--%"
        info_txt = (
            f"Lv.{sm_level if sm_level else '--'}   "
            f"HP: {hp_txt}   "
            f"Kill: {sm_kills}   "
            f"Time: {sm_elapsed_min:.0f} min"
        )
        pico_txt = "연결됨" if pico_connected else "미연결"
        fps_txt  = (
            f"Capture: {capture_fps:.1f} fps   "
            f"Detection: {detection_fps:.1f} fps   "
            f"Pico: {pico_txt}"
        )

        self.root.after(0, lambda: self._lbl_info.config(text=info_txt))
        self.root.after(0, lambda: self._lbl_fps.config(text=fps_txt))

    def _update_sm_label(self, state: str):
        # 상태별 색상
        colors = {
            "IDLE":              C_DIM,
            "시작 중...":        C_TEXT,
            "중지됨":            C_DIM,
            "USE_SCROLL_DUMMY":  "#f9e2af",
            "MOVE_TO_DUMMY":     "#fab387",
            "ATTACKING_DUMMY":   "#89dceb",
            "USE_SPEED_POTION":  C_ACCENT,
            "MOVE_TO_HUNT_ZONE": "#fab387",
            "HUNTING_10":        C_START,
            "LOOTING":           "#94e2d5",
            "DONE_PHASE1":       "#f9e2af",
        }
        color = colors.get(state, C_TEXT)
        self._lbl_sm.config(text=state, fg=color)

    # ══════════════════════════════════════════════════════════════════
    #  좌표 세팅 도구
    # ══════════════════════════════════════════════════════════════════

    def open_coord_picker(self):
        picker_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "coordinate_picker.py")
        )
        try:
            subprocess.Popen(["py", "-3.11", picker_path])
        except FileNotFoundError:
            subprocess.Popen([sys.executable, picker_path])

    # ══════════════════════════════════════════════════════════════════
    #  실행 / 종료
    # ══════════════════════════════════════════════════════════════════

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    SettingsWindow().run()
