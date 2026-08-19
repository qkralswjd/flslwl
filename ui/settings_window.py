"""Tkinter 설정 UI: config 편집, Start/Stop, 실시간 상태 표시.

변경점 (순차 타겟 상태머신 통합):
    - Pico 섹션: click_target / click_cooldown_ms → 제거
    - 신규 SM 파라미터 추가:
        Lock Confirm Frames   — 연속 몇 프레임 확인 후 클릭할지
        Wait Dead Timeout(ms) — 타겟이 안 사라지면 강제 전환 시간
        Cooldown(ms)          — 다음 타겟으로 넘어가기 전 대기
        Target Priority       — nearest_center / nearest_origin / oldest / newest
    - 상태 표시줄에 현재 Target 상태(TargetState) 추가
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main as tracker_main
from config.config_loader import DEFAULT_CONFIG_PATH, load_config, save_config


def _get_serial_ports():
    """현재 연결된 COM 포트 목록을 반환합니다."""
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()] or ["(없음)"]
    except ImportError:
        return ["(pyserial 없음)"]


class SettingsWindow:
    # ── 일반 설정 필드 ────────────────────────────────────────────────
    FIELDS = [
        ("Monitor Index",                   ["monitor_index"]),
        ("Capture FPS",                     ["capture_fps"]),
        ("Detection FPS",                   ["detection_fps"]),
        ("Motion History",                  ["motion", "history"]),
        ("Motion Var Threshold",            ["motion", "var_threshold"]),
        ("Classifier Confidence Threshold", ["classifier", "confidence_threshold"]),
        ("Min Area",                        ["min_area"]),
        ("Max Area",                        ["max_area"]),
        ("Max Missing Frames",              ["max_missing_frames"]),
        ("Max Match Distance",              ["max_match_distance"]),
    ]

    # ── Pico 수치 설정 필드 ───────────────────────────────────────────
    PICO_NUM_FIELDS = [
        ("Click Pulse (ms)",        ["pico", "click_pulse_ms"]),
        ("Offset X (px)",           ["pico", "click_offset_x"]),
        ("Offset Y (px)",           ["pico", "click_offset_y"]),
    ]

    # ── 순차 상태머신(SM) 수치 설정 필드 ─────────────────────────────
    SM_NUM_FIELDS = [
        ("Lock Confirm Frames",     ["pico", "lock_confirm_frames"]),
        ("Wait Dead Timeout (ms)",  ["pico", "wait_dead_timeout_ms"]),
        ("Cooldown (ms)",           ["pico", "next_target_cooldown_ms"]),
    ]

    # ── 드래그 수치 설정 필드 ────────────────────────────────────────
    DRAG_NUM_FIELDS = [
        ("Drag DX (px)",   ["pico", "drag_dx"]),
        ("Drag DY (px)",   ["pico", "drag_dy"]),
        ("Drag Steps",     ["pico", "drag_steps"]),
    ]

    # ── SceneMotion 수치 설정 필드 ───────────────────────────────────
    SCENE_NUM_FIELDS = [
        ("Scene Threshold", ["scene_motion", "scene_threshold"]),
        ("Settle Frames",   ["scene_motion", "settle_frames"]),
    ]

    TARGET_PRIORITIES = ["nearest_center", "nearest_origin", "oldest", "newest"]

    def __init__(self):
        self.config       = load_config()
        self.stop_event   = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title("Game Enemy Tracker + Pico  [순차 타겟 SM]")
        self.root.resizable(False, False)

        self.vars        = {}   # 일반 설정
        self.pico_vars   = {}   # pico 수치 설정
        self.sm_vars     = {}   # SM 수치 설정
        self.drag_vars   = {}   # 드래그 수치 설정
        self.scene_vars  = {}   # SceneMotion 수치 설정
        self._build_ui()

    # ── 설정값 접근 헬퍼 ──────────────────────────────────────────────

    def _get(self, path):
        node = self.config
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        return node

    def _set(self, path, value):
        node = self.config
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    # ── UI 구성 ───────────────────────────────────────────────────────

    def _build_ui(self):
        left_frame   = ttk.LabelFrame(self.root, text="감지 설정",           padding=6)
        right_frame  = ttk.LabelFrame(self.root, text="Pico + 순차 타겟 설정", padding=6)
        extra_frame  = ttk.LabelFrame(self.root, text="드래그 / 이동감지 설정",  padding=6)
        bot_frame    = ttk.Frame(self.root, padding=4)

        left_frame.grid(row=0,  column=0, padx=6, pady=4, sticky="nsew")
        right_frame.grid(row=0, column=1, padx=6, pady=4, sticky="nsew")
        extra_frame.grid(row=0, column=2, padx=6, pady=4, sticky="nsew")
        bot_frame.grid(row=1,   column=0, columnspan=3, sticky="ew", padx=6, pady=4)

        # ── 일반 설정 ─────────────────────────────────────────────────
        for row, (label, path) in enumerate(self.FIELDS):
            ttk.Label(left_frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path)))
            ttk.Entry(left_frame, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)
            self.vars[label] = (var, path)

        # ── Pico 설정 ─────────────────────────────────────────────────
        row = 0

        # 활성화 체크박스
        self._pico_enabled = tk.BooleanVar(
            value=bool(self.config.get("pico", {}).get("enabled", False))
        )
        ttk.Checkbutton(
            right_frame, text="Pico 활성화", variable=self._pico_enabled
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1

        # 포트 선택
        ttk.Label(right_frame, text="Serial Port").grid(row=row, column=0, sticky="w", padx=4)
        self._port_var = tk.StringVar(
            value=self.config.get("pico", {}).get("serial_port", "COM3")
        )
        self._port_combo = ttk.Combobox(
            right_frame, textvariable=self._port_var, width=9,
            values=_get_serial_ports(), state="readonly"
        )
        self._port_combo.grid(row=row, column=1, padx=4, pady=2)
        ttk.Button(
            right_frame, text="⟳", width=3,
            command=self._refresh_ports
        ).grid(row=row, column=2, padx=2)
        row += 1

        # Pico 수치 설정 (pulse, offset)
        for label, path in self.PICO_NUM_FIELDS:
            ttk.Label(right_frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(right_frame, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)
            self.pico_vars[label] = (var, path)
            row += 1

        # ── 구분선 ────────────────────────────────────────────────────
        ttk.Separator(right_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        # ── 순차 타겟 상태머신(SM) 설정 ───────────────────────────────
        ttk.Label(
            right_frame, text="▶ 순차 타겟 상태머신",
            font=("", 9, "bold")
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))
        row += 1

        # SM 수치 필드
        for label, path in self.SM_NUM_FIELDS:
            ttk.Label(right_frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(right_frame, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)
            self.sm_vars[label] = (var, path)
            row += 1

        # 타겟 우선순위 드롭다운
        ttk.Label(right_frame, text="Target Priority").grid(row=row, column=0, sticky="w", padx=4)
        self._priority_var = tk.StringVar(
            value=self.config.get("pico", {}).get("target_priority", "nearest_center")
        )
        ttk.Combobox(
            right_frame, textvariable=self._priority_var,
            values=self.TARGET_PRIORITIES, state="readonly", width=14
        ).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        # 우선순위 설명
        hint = (
            "nearest_center : ROI 중심에서 가장 가까운 적\n"
            "nearest_origin  : ROI 좌상단(0,0) 기준 가까운 적\n"
            "oldest           : 화면에 가장 오래 있던 적\n"
            "newest           : 가장 최근에 나타난 적"
        )
        ttk.Label(right_frame, text=hint, foreground="gray", justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", padx=4, pady=6
        )
        row += 1

        # ── extra_frame: 드래그 + SceneMotion ────────────────────────
        erow = 0

        # 드래그 섹션
        ttk.Label(
            extra_frame, text="▶ 드래그", font=("", 9, "bold")
        ).grid(row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))
        erow += 1

        self._drag_enabled = tk.BooleanVar(
            value=bool(self.config.get("pico", {}).get("drag_enabled", False))
        )
        ttk.Checkbutton(
            extra_frame, text="드래그 활성화", variable=self._drag_enabled
        ).grid(row=erow, column=0, columnspan=2, sticky="w", pady=2)
        erow += 1

        for label, path in self.DRAG_NUM_FIELDS:
            ttk.Label(extra_frame, text=label).grid(row=erow, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(extra_frame, textvariable=var, width=10).grid(row=erow, column=1, padx=4, pady=2)
            self.drag_vars[label] = (var, path)
            erow += 1

        drag_hint = (
            "DX > 0 : 오른쪽  DX < 0 : 왼쪽\n"
            "DY > 0 : 아래쪽  DY < 0 : 위쪽\n"
            "Steps : 드래그 중간 단계 수"
        )
        ttk.Label(extra_frame, text=drag_hint, foreground="gray", justify="left").grid(
            row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=4
        )
        erow += 1

        # 구분선
        ttk.Separator(extra_frame, orient="horizontal").grid(
            row=erow, column=0, columnspan=2, sticky="ew", pady=6
        )
        erow += 1

        # SceneMotion 섹션
        ttk.Label(
            extra_frame, text="▶ 이동 감지 필터", font=("", 9, "bold")
        ).grid(row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))
        erow += 1

        self._scene_enabled = tk.BooleanVar(
            value=bool(self.config.get("scene_motion", {}).get("enabled", True))
        )
        ttk.Checkbutton(
            extra_frame, text="이동 중 감지 정지", variable=self._scene_enabled
        ).grid(row=erow, column=0, columnspan=2, sticky="w", pady=2)
        erow += 1

        for label, path in self.SCENE_NUM_FIELDS:
            ttk.Label(extra_frame, text=label).grid(row=erow, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(extra_frame, textvariable=var, width=10).grid(row=erow, column=1, padx=4, pady=2)
            self.scene_vars[label] = (var, path)
            erow += 1

        scene_hint = (
            "Threshold : 높을수록 둔감 (기본 8.0)\n"
            "Settle    : 정지 후 안정화 대기 프레임"
        )
        ttk.Label(extra_frame, text=scene_hint, foreground="gray", justify="left").grid(
            row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=4
        )
        erow += 1

        # ── 상태 표시줄 ───────────────────────────────────────────────
        self.status_label = ttk.Label(
            bot_frame,
            text="대기 중... | Capture: 0.0 fps | Detection: 0.0 fps | Pico: - | Target: -",
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, columnspan=3, sticky="ew", pady=4)

        # ── Start / Stop 버튼 ─────────────────────────────────────────
        ttk.Button(bot_frame, text="▶  Start", command=self.start).grid(
            row=1, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(bot_frame, text="■  Stop",  command=self.stop).grid(
            row=1, column=1, padx=4, pady=4, sticky="ew"
        )
        bot_frame.columnconfigure(0, weight=1)
        bot_frame.columnconfigure(1, weight=1)

    def _refresh_ports(self):
        ports = _get_serial_ports()
        self._port_combo["values"] = ports
        if self._port_var.get() not in ports and ports:
            self._port_var.set(ports[0])

    # ── 설정 적용 / 저장 ─────────────────────────────────────────────

    def _apply_fields_to_config(self):
        # 일반 설정
        for _label, (var, path) in self.vars.items():
            raw = var.get()
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            self._set(path, value)

        # Pico 기본 설정
        pico = self.config.setdefault("pico", {})
        pico["enabled"]     = self._pico_enabled.get()
        pico["serial_port"] = self._port_var.get()

        for _label, (var, path) in self.pico_vars.items():
            raw = var.get()
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            self._set(path, value)

        # SM 설정
        for _label, (var, path) in self.sm_vars.items():
            raw = var.get()
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            self._set(path, value)

        pico["target_priority"] = self._priority_var.get()

        # 드래그 설정
        pico["drag_enabled"] = self._drag_enabled.get()
        for _label, (var, path) in self.drag_vars.items():
            raw = var.get()
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            self._set(path, value)

        # SceneMotion 설정
        sm = self.config.setdefault("scene_motion", {})
        sm["enabled"] = self._scene_enabled.get()
        for _label, (var, path) in self.scene_vars.items():
            raw = var.get()
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            self._set(path, value)

        save_config(self.config, DEFAULT_CONFIG_PATH)

    # ── Start / Stop ─────────────────────────────────────────────────

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self._apply_fields_to_config()
        self.stop_event.clear()

        def worker():
            tracker_main.run(
                self.config,
                stop_event=self.stop_event,
                status_callback=self._on_status,
            )

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()

    # ── 상태 업데이트 콜백 ───────────────────────────────────────────
    # status_callback(enemy_count, capture_fps, detection_fps, pico_connected,
    #                 target_state_name="", current_target_id=None)

    def _on_status(
        self,
        enemy_count: int,
        capture_fps: float,
        detection_fps: float,
        pico_connected: bool = False,
        target_state_name: str = "",
        current_target_id=None,
    ):
        pico_txt   = "Connected" if pico_connected else "Disconnected"
        port_txt   = self._port_var.get() if self._pico_enabled.get() else "disabled"
        target_txt = (
            f"#{current_target_id} [{target_state_name}]"
            if current_target_id is not None
            else f"[{target_state_name}]" if target_state_name else "-"
        )
        text = (
            f"적: {enemy_count} | "
            f"Capture: {capture_fps:.1f} fps | "
            f"Detection: {detection_fps:.1f} fps | "
            f"Pico [{port_txt}]: {pico_txt} | "
            f"Target: {target_txt}"
        )
        self.root.after(0, lambda: self.status_label.config(text=text))

    # ── 실행 / 종료 ──────────────────────────────────────────────────

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    SettingsWindow().run()
