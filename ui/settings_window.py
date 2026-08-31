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

import json
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main as tracker_main
from config.config_loader import DEFAULT_CONFIG_PATH, load_config, save_config

AUTOMATION_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "config_automation.json"
)


def _load_automation_config():
    path = os.path.normpath(AUTOMATION_CONFIG_PATH)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_automation_config(cfg: dict):
    path = os.path.normpath(AUTOMATION_CONFIG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


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
        self.config            = load_config()
        self.automation_config = _load_automation_config()
        self.stop_event        = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self._hunting_sm       = None   # HuntingStateMachine 참조

        self.root = tk.Tk()
        self.root.title("Game Enemy Tracker + Pico  [순차 타겟 SM + Auto Leveling]")
        self.root.resizable(False, False)

        self.vars        = {}   # 일반 설정
        self.pico_vars   = {}   # pico 수치 설정
        self.sm_vars     = {}   # SM 수치 설정
        self.drag_vars   = {}   # 드래그 수치 설정
        self.scene_vars  = {}   # SceneMotion 수치 설정
        self.auto_vars   = {}   # Automation 수치 설정

        # 현재 선택된 모드: "leveling" | "dungeon"
        self._mode = tk.StringVar(value="leveling")

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
        # ── 모드 선택 탭 (최상단) ─────────────────────────────────────
        mode_frame = ttk.Frame(self.root, padding=4)
        mode_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=6, pady=(6, 0))

        ttk.Label(mode_frame, text="🎮 모드 선택:", font=("", 10, "bold")).pack(side="left", padx=(0, 8))

        self._btn_leveling = tk.Button(
            mode_frame, text="⚔  1~10레벨 모드",
            font=("", 10, "bold"), bg="#2a6ead", fg="white",
            relief="raised", padx=12, pady=6,
            command=lambda: self._switch_mode("leveling"),
        )
        self._btn_leveling.pack(side="left", padx=4)

        self._btn_dungeon = tk.Button(
            mode_frame, text="🏰  던전 사냥 모드",
            font=("", 10, "bold"), bg="#555555", fg="white",
            relief="raised", padx=12, pady=6,
            command=lambda: self._switch_mode("dungeon"),
        )
        self._btn_dungeon.pack(side="left", padx=4)

        self._mode_hint = ttk.Label(
            mode_frame, text="현재: 1~10레벨 모드  (드래그 자동사냥 → 5렙)",
            foreground="#2a6ead", font=("", 9),
        )
        self._mode_hint.pack(side="left", padx=12)

        # ── 패널 컨테이너 ─────────────────────────────────────────────
        panels_frame = ttk.Frame(self.root, padding=0)
        panels_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=0, pady=0)

        # 1~10레벨 모드 패널
        self._leveling_panel = ttk.Frame(panels_frame, padding=0)
        # 던전 사냥 모드 패널
        self._dungeon_panel  = ttk.Frame(panels_frame, padding=0)

        # 처음엔 레벨링 패널 표시
        self._leveling_panel.grid(row=0, column=0, sticky="nsew")

        # ── 레벨링 패널 내부 구성 ─────────────────────────────────────
        lv_left  = ttk.LabelFrame(self._leveling_panel, text="감지 설정",            padding=6)
        lv_pico  = ttk.LabelFrame(self._leveling_panel, text="Pico 설정",            padding=6)
        lv_auto  = ttk.LabelFrame(self._leveling_panel, text="⚔ 1~10레벨 자동사냥", padding=6)
        lv_left.grid(row=0, column=0, padx=6, pady=4, sticky="nsew")
        lv_pico.grid(row=0, column=1, padx=6, pady=4, sticky="nsew")
        lv_auto.grid(row=0, column=2, padx=6, pady=4, sticky="nsew")

        # ── 던전 패널 내부 구성 ───────────────────────────────────────
        dn_left  = ttk.LabelFrame(self._dungeon_panel, text="감지 설정",              padding=6)
        dn_right = ttk.LabelFrame(self._dungeon_panel, text="Pico + 순차 타겟 설정",  padding=6)
        dn_extra = ttk.LabelFrame(self._dungeon_panel, text="드래그 / 이동감지 설정", padding=6)
        dn_left.grid(row=0,  column=0, padx=6, pady=4, sticky="nsew")
        dn_right.grid(row=0, column=1, padx=6, pady=4, sticky="nsew")
        dn_extra.grid(row=0, column=2, padx=6, pady=4, sticky="nsew")

        # ── alias: 기존 코드가 left_frame/right_frame/extra_frame/auto_frame을 쓰므로
        left_frame  = dn_left
        right_frame = dn_right
        extra_frame = dn_extra
        auto_frame  = lv_auto

        # bot_frame은 공통 (항상 표시)
        bot_frame = ttk.Frame(self.root, padding=4)
        bot_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=4)

        # ── 레벨링 패널: 감지 설정 (lv_left) ─────────────────────────
        for row, (label, path) in enumerate(self.FIELDS):
            ttk.Label(lv_left, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path)))
            ttk.Entry(lv_left, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)
            # vars는 던전 패널과 공유 (같은 config)
            self.vars[label] = (var, path)

        # ── 레벨링 패널: Pico 설정 (lv_pico) ─────────────────────────
        prow = 0
        self._pico_enabled = tk.BooleanVar(
            value=bool(self.config.get("pico", {}).get("enabled", False))
        )
        ttk.Checkbutton(lv_pico, text="Pico 활성화", variable=self._pico_enabled).grid(
            row=prow, column=0, columnspan=2, sticky="w", pady=4)
        prow += 1

        ttk.Label(lv_pico, text="Serial Port").grid(row=prow, column=0, sticky="w", padx=4)
        self._port_var = tk.StringVar(value=self.config.get("pico", {}).get("serial_port", "COM3"))
        self._port_combo = ttk.Combobox(
            lv_pico, textvariable=self._port_var, width=9,
            values=_get_serial_ports(), state="readonly"
        )
        self._port_combo.grid(row=prow, column=1, padx=4, pady=2)
        ttk.Button(lv_pico, text="⟳", width=3, command=self._refresh_ports).grid(
            row=prow, column=2, padx=2)
        prow += 1

        for label, path in self.PICO_NUM_FIELDS:
            ttk.Label(lv_pico, text=label).grid(row=prow, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(lv_pico, textvariable=var, width=10).grid(row=prow, column=1, padx=4, pady=2)
            self.pico_vars[label] = (var, path)
            prow += 1

        # ── 던전 패널: 감지 설정 (dn_left) ───────────────────────────
        for row, (label, path) in enumerate(self.FIELDS):
            ttk.Label(dn_left, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            # vars는 이미 lv_left에서 등록됨 — 동일 var 재사용
            var = self.vars[label][0]
            ttk.Entry(dn_left, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)

        # ── 던전 패널: Pico + SM 설정 (dn_right) ─────────────────────
        row = 0
        ttk.Checkbutton(
            dn_right, text="Pico 활성화", variable=self._pico_enabled
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1

        ttk.Label(dn_right, text="Serial Port").grid(row=row, column=0, sticky="w", padx=4)
        self._port_combo2 = ttk.Combobox(
            dn_right, textvariable=self._port_var, width=9,
            values=_get_serial_ports(), state="readonly"
        )
        self._port_combo2.grid(row=row, column=1, padx=4, pady=2)
        ttk.Button(
            dn_right, text="⟳", width=3,
            command=self._refresh_ports
        ).grid(row=row, column=2, padx=2)
        row += 1

        # Pico 수치 설정 (pulse, offset) - 던전 패널
        for label, path in self.PICO_NUM_FIELDS:
            ttk.Label(dn_right, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = self.pico_vars.get(label, (tk.StringVar(value=str(self._get(path) or 0)), path))[0]
            ttk.Entry(dn_right, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)
            self.pico_vars[label] = (var, path)
            row += 1

        # ── 구분선 ────────────────────────────────────────────────────
        ttk.Separator(dn_right, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        # ── 순차 타겟 상태머신(SM) 설정 ───────────────────────────────
        ttk.Label(
            dn_right, text="▶ 순차 타겟 상태머신",
            font=("", 9, "bold")
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))
        row += 1

        for label, path in self.SM_NUM_FIELDS:
            ttk.Label(dn_right, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(dn_right, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)
            self.sm_vars[label] = (var, path)
            row += 1

        ttk.Label(dn_right, text="Target Priority").grid(row=row, column=0, sticky="w", padx=4)
        self._priority_var = tk.StringVar(
            value=self.config.get("pico", {}).get("target_priority", "nearest_center")
        )
        ttk.Combobox(
            dn_right, textvariable=self._priority_var,
            values=self.TARGET_PRIORITIES, state="readonly", width=14
        ).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        hint = (
            "nearest_center : ROI 중심에서 가장 가까운 적\n"
            "nearest_origin  : ROI 좌상단(0,0) 기준 가까운 적\n"
            "oldest           : 화면에 가장 오래 있던 적\n"
            "newest           : 가장 최근에 나타난 적"
        )
        ttk.Label(dn_right, text=hint, foreground="gray", justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", padx=4, pady=6
        )
        row += 1

        # ── 던전 패널: 드래그 + SceneMotion (dn_extra) ────────────────
        erow = 0
        ttk.Label(dn_extra, text="▶ 드래그", font=("", 9, "bold")).grid(
            row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))
        erow += 1

        self._drag_enabled = tk.BooleanVar(
            value=bool(self.config.get("pico", {}).get("drag_enabled", False))
        )
        ttk.Checkbutton(dn_extra, text="드래그 활성화", variable=self._drag_enabled).grid(
            row=erow, column=0, columnspan=2, sticky="w", pady=2)
        erow += 1

        for label, path in self.DRAG_NUM_FIELDS:
            ttk.Label(dn_extra, text=label).grid(row=erow, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(dn_extra, textvariable=var, width=10).grid(row=erow, column=1, padx=4, pady=2)
            self.drag_vars[label] = (var, path)
            erow += 1

        drag_hint = (
            "DX > 0 : 오른쪽  DX < 0 : 왼쪽\n"
            "DY > 0 : 아래쪽  DY < 0 : 위쪽\n"
            "Steps : 드래그 중간 단계 수"
        )
        ttk.Label(dn_extra, text=drag_hint, foreground="gray", justify="left").grid(
            row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        erow += 1

        ttk.Separator(dn_extra, orient="horizontal").grid(
            row=erow, column=0, columnspan=2, sticky="ew", pady=6)
        erow += 1

        ttk.Label(dn_extra, text="▶ 이동 감지 필터", font=("", 9, "bold")).grid(
            row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))
        erow += 1

        self._scene_enabled = tk.BooleanVar(
            value=bool(self.config.get("scene_motion", {}).get("enabled", True))
        )
        ttk.Checkbutton(dn_extra, text="이동 중 감지 정지", variable=self._scene_enabled).grid(
            row=erow, column=0, columnspan=2, sticky="w", pady=2)
        erow += 1

        for label, path in self.SCENE_NUM_FIELDS:
            ttk.Label(dn_extra, text=label).grid(row=erow, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(self._get(path) or 0))
            ttk.Entry(dn_extra, textvariable=var, width=10).grid(row=erow, column=1, padx=4, pady=2)
            self.scene_vars[label] = (var, path)
            erow += 1

        scene_hint = (
            "Threshold : 높을수록 둔감 (기본 8.0)\n"
            "Settle    : 정지 후 안정화 대기 프레임"
        )
        ttk.Label(dn_extra, text=scene_hint, foreground="gray", justify="left").grid(
            row=erow, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        erow += 1

        # ── 1~10레벨 모드 자동사냥 패널 ──────────────────────────────
        self._build_auto_panel(lv_auto)

        # ── 공통 하단 영역 ────────────────────────────────────────────
        self.status_label = ttk.Label(
            bot_frame,
            text="대기 중... | Capture: 0.0 fps | Detection: 0.0 fps | Pico: - | Target: -",
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, columnspan=4, sticky="ew", pady=4)

        self.auto_status_label = ttk.Label(
            bot_frame, text="[AUTO] 대기 중", foreground="gray", anchor="w"
        )
        self.auto_status_label.grid(row=1, column=0, columnspan=4, sticky="ew", pady=2)

        # ── 모드별 버튼 영역 ──────────────────────────────────────────
        # 1~10레벨 모드 버튼
        self._lv_btn_frame = ttk.Frame(bot_frame)
        self._lv_btn_frame.grid(row=2, column=0, columnspan=4, sticky="ew")

        ttk.Button(self._lv_btn_frame, text="▶  Start", command=self.start).grid(
            row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(self._lv_btn_frame, text="■  Stop",  command=self.stop).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(self._lv_btn_frame, text="⚔ 레벨링 시작", command=self.auto_start).grid(
            row=0, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(self._lv_btn_frame, text="⏹ 레벨링 중지", command=self.auto_stop).grid(
            row=0, column=3, padx=4, pady=4, sticky="ew")
        self._lv_btn_frame.columnconfigure(0, weight=1)
        self._lv_btn_frame.columnconfigure(1, weight=1)
        self._lv_btn_frame.columnconfigure(2, weight=1)
        self._lv_btn_frame.columnconfigure(3, weight=1)

        # 던전 사냥 모드 버튼
        self._dn_btn_frame = ttk.Frame(bot_frame)

        ttk.Button(self._dn_btn_frame, text="▶  Start", command=self.start).grid(
            row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(self._dn_btn_frame, text="■  Stop",  command=self.stop).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew")
        self._dn_btn_frame.columnconfigure(0, weight=1)
        self._dn_btn_frame.columnconfigure(1, weight=1)

        # 좌표 세팅 도구 (공통)
        ttk.Button(bot_frame, text="📍 좌표 세팅 도구 열기",
                   command=self.open_coord_picker).grid(
            row=3, column=0, columnspan=4, padx=4, pady=6, sticky="ew")
        bot_frame.columnconfigure(0, weight=1)
        bot_frame.columnconfigure(1, weight=1)
        bot_frame.columnconfigure(2, weight=1)
        bot_frame.columnconfigure(3, weight=1)

        # 초기 모드 적용
        self._switch_mode("leveling")

    def _switch_mode(self, mode: str):
        """모드 전환: leveling / dungeon"""
        self._mode.set(mode)

        if mode == "leveling":
            # 패널 교체
            self._dungeon_panel.grid_remove()
            self._leveling_panel.grid(row=0, column=0, sticky="nsew")
            # 버튼 교체
            self._dn_btn_frame.grid_remove()
            self._lv_btn_frame.grid(row=2, column=0, columnspan=4, sticky="ew")
            # 버튼 색상
            self._btn_leveling.config(bg="#2a6ead", relief="sunken")
            self._btn_dungeon.config(bg="#555555", relief="raised")
            self._mode_hint.config(
                text="현재: 1~10레벨 모드  (드래그 자동사냥 → 5렙)",
                foreground="#2a6ead"
            )
        else:
            # 패널 교체
            self._leveling_panel.grid_remove()
            self._dungeon_panel.grid(row=0, column=0, sticky="nsew")
            # 버튼 교체
            self._lv_btn_frame.grid_remove()
            self._dn_btn_frame.grid(row=2, column=0, columnspan=4, sticky="ew")
            # 버튼 색상
            self._btn_dungeon.config(bg="#8B2000", relief="sunken")
            self._btn_leveling.config(bg="#555555", relief="raised")
            self._mode_hint.config(
                text="현재: 던전 사냥 모드  (템플릿 매칭 + 자동 공격)",
                foreground="#8B2000"
            )

    def _build_auto_panel(self, frame):
        """자동 레벨링 설정 패널 (요정 1단계 전용)."""
        ac = self.automation_config
        arow = 0

        # ── 키 설정 ──────────────────────────────────────────────────
        ttk.Label(frame, text="▶ 키 설정", font=("",9,"bold")).grid(
            row=arow, column=0, columnspan=2, sticky="w", padx=4, pady=(0,2))
        arow += 1

        for label, keys, default in [
            ("물약 키 (HP)",    ("keys","potion"),       "F5"),
            ("두루마리 키",      ("keys","scroll"),       "F6"),
            ("속도물약 키",      ("keys","speed_potion"), "F9"),
            ("물약 쿨타임ms",   ("keys","potion_cooldown_ms"), "3000"),
        ]:
            ttk.Label(frame, text=label).grid(row=arow, column=0, sticky="w", padx=4, pady=1)
            node = ac
            for k in keys[:-1]:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            val = str(node.get(keys[-1], default)) if isinstance(node, dict) else default
            var = tk.StringVar(value=val)
            ttk.Entry(frame, textvariable=var, width=10).grid(row=arow, column=1, padx=4, pady=1)
            self.auto_vars[label] = (var, keys)
            arow += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=arow, column=0, columnspan=2, sticky="ew", pady=4)
        arow += 1

        # ── HP 바 설정 ────────────────────────────────────────────────
        ttk.Label(frame, text="▶ HP 바", font=("",9,"bold")).grid(
            row=arow, column=0, columnspan=2, sticky="w", padx=4, pady=(0,2))
        arow += 1

        for label, keys, default in [
            ("HP 바 X",    ("hp_bar","region","x"),      "0"),
            ("HP 바 Y",    ("hp_bar","region","y"),      "0"),
            ("HP 바 W",    ("hp_bar","region","width"),  "200"),
            ("HP 바 H",    ("hp_bar","region","height"), "10"),
            ("HP 경고 %",  ("hp_bar","threshold_pct"),   "50.0"),
        ]:
            ttk.Label(frame, text=label).grid(row=arow, column=0, sticky="w", padx=4, pady=1)
            node = ac
            for k in keys[:-1]:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            val = str(node.get(keys[-1], default)) if isinstance(node, dict) else default
            var = tk.StringVar(value=val)
            ttk.Entry(frame, textvariable=var, width=8).grid(row=arow, column=1, padx=4, pady=1)
            self.auto_vars[label] = (var, keys)
            arow += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=arow, column=0, columnspan=2, sticky="ew", pady=4)
        arow += 1

        # ── 레벨 OCR 설정 ─────────────────────────────────────────────
        ttk.Label(frame, text="▶ 레벨 OCR", font=("",9,"bold")).grid(
            row=arow, column=0, columnspan=2, sticky="w", padx=4, pady=(0,2))
        arow += 1

        for label, keys, default in [
            ("레벨 X",        ("level_ocr","region","x"),      "0"),
            ("레벨 Y",        ("level_ocr","region","y"),      "0"),
            ("레벨 W",        ("level_ocr","region","width"),  "80"),
            ("레벨 H",        ("level_ocr","region","height"), "25"),
            ("허수아비목표Lv", ("level_ocr","target_level_dummy"), "5"),
            ("사냥터목표Lv",   ("level_ocr","target_level_hunt"),  "10"),
        ]:
            ttk.Label(frame, text=label).grid(row=arow, column=0, sticky="w", padx=4, pady=1)
            node = ac
            for k in keys[:-1]:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            val = str(node.get(keys[-1], default)) if isinstance(node, dict) else default
            var = tk.StringVar(value=val)
            ttk.Entry(frame, textvariable=var, width=8).grid(row=arow, column=1, padx=4, pady=1)
            self.auto_vars[label] = (var, keys)
            arow += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=arow, column=0, columnspan=2, sticky="ew", pady=4)
        arow += 1

        # ── 말하는 두루마리 (F6) 설정 ────────────────────────────────
        ttk.Label(frame, text="▶ 두루마리→허수아비", font=("",9,"bold")).grid(
            row=arow, column=0, columnspan=2, sticky="w", padx=4, pady=(0,2))
        arow += 1

        for label, keys, default in [
            ("목적지 창 X",   ("scroll_dummy","destination_region","x"),      "400"),
            ("목적지 창 Y",   ("scroll_dummy","destination_region","y"),      "150"),
            ("목적지 창 W",   ("scroll_dummy","destination_region","width"),  "300"),
            ("목적지 창 H",   ("scroll_dummy","destination_region","height"), "400"),
            ("목적지 텍스트", ("scroll_dummy","destination_text"),            "허수아비"),
        ]:
            ttk.Label(frame, text=label).grid(row=arow, column=0, sticky="w", padx=4, pady=1)
            node = ac
            for k in keys[:-1]:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            val = str(node.get(keys[-1], default)) if isinstance(node, dict) else default
            var = tk.StringVar(value=val)
            ttk.Entry(frame, textvariable=var, width=12).grid(row=arow, column=1, padx=4, pady=1)
            self.auto_vars[label] = (var, keys)
            arow += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=arow, column=0, columnspan=2, sticky="ew", pady=4)
        arow += 1

        # ── 허수아비 드래그 공격 ──────────────────────────────────────
        ttk.Label(frame, text="▶ 허수아비 공격", font=("",9,"bold")).grid(
            row=arow, column=0, columnspan=2, sticky="w", padx=4, pady=(0,2))
        arow += 1

        for label, keys, default in [
            ("드래그 시작 X", ("dummy","drag_from","x"), "960"),
            ("드래그 시작 Y", ("dummy","drag_from","y"), "600"),
            ("드래그 끝 X",   ("dummy","drag_to","x"),   "960"),
            ("드래그 끝 Y",   ("dummy","drag_to","y"),   "400"),
            ("드래그 단계",   ("dummy","drag_steps"),    "8"),
            ("공격 간격ms",   ("dummy","attack_interval_ms"), "500"),
        ]:
            ttk.Label(frame, text=label).grid(row=arow, column=0, sticky="w", padx=4, pady=1)
            node = ac
            for k in keys[:-1]:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            val = str(node.get(keys[-1], default)) if isinstance(node, dict) else default
            var = tk.StringVar(value=val)
            ttk.Entry(frame, textvariable=var, width=8).grid(row=arow, column=1, padx=4, pady=1)
            self.auto_vars[label] = (var, keys)
            arow += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=arow, column=0, columnspan=2, sticky="ew", pady=4)
        arow += 1

        # ── 사냥터 웨이포인트 ─────────────────────────────────────────
        ttk.Label(frame, text="▶ 사냥터 웨이포인트", font=("",9,"bold")).grid(
            row=arow, column=0, columnspan=2, sticky="w", padx=4, pady=(0,2))
        arow += 1

        hwp_cfg = ac.get("hunt_waypoints", {})
        wps = hwp_cfg.get("points", [{"x":960,"y":400,"label":"사냥터-A","wait_ms":2000}])
        self._wp_text = tk.Text(frame, width=24, height=5, font=("Consolas",8))
        self._wp_text.grid(row=arow, column=0, columnspan=2, padx=4, pady=2)
        wp_str = "\n".join(
            f"{wp.get('x',0)},{wp.get('y',0)},{wp.get('label','WP')},{wp.get('wait_ms',2000)}"
            for wp in wps
        )
        self._wp_text.insert("1.0", wp_str)
        arow += 1
        ttk.Label(frame, text="x,y,이름,대기ms (한 줄씩)", foreground="gray").grid(
            row=arow, column=0, columnspan=2, sticky="w", padx=4)
        arow += 1

    def _apply_auto_fields_to_config(self):
        """auto_vars → automation_config 반영."""
        ac = self.automation_config

        def nested_set(d, keys, value):
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        for label, (var, keys) in self.auto_vars.items():
            raw = var.get()
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            nested_set(ac, keys, value)

        # 사냥터 웨이포인트 파싱 → hunt_waypoints.points 에 저장
        wps = []
        for line in self._wp_text.get("1.0", "end").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    wp = {
                        "x": int(parts[0]),
                        "y": int(parts[1]),
                        "label": parts[2] if len(parts) > 2 else f"WP{len(wps)}",
                        "wait_ms": int(parts[3]) if len(parts) > 3 else 2000,
                    }
                    wps.append(wp)
                except ValueError:
                    pass
        if wps:
            ac.setdefault("hunt_waypoints", {})["points"] = wps

        _save_automation_config(ac)
        self.automation_config = ac

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
        current_mode = self._mode.get()  # "leveling" or "dungeon"

        def worker():
            tracker_main.run(
                self.config,
                stop_event=self.stop_event,
                status_callback=self._on_status,
                mode=current_mode,
            )

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        if self._hunting_sm:
            self._hunting_sm.stop()
            self._hunting_sm = None

    def auto_start(self):
        """1~10레벨 자동사냥 시작."""
        self._apply_auto_fields_to_config()

        # 실행 중이면 재시작
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            import time; time.sleep(0.5)

        self._apply_fields_to_config()
        self.stop_event.clear()

        def worker():
            tracker_main.run(
                self.config,
                stop_event=self.stop_event,
                status_callback=self._on_status,
                automation_config=self.automation_config,
                mode="leveling",
            )

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

        # 자동 레벨링 SM을 별도 스레드에서 start() 호출
        def sm_starter():
            import time
            time.sleep(2.0)  # 트래커 초기화 대기
            # SM은 main.py 내부에서 생성되므로 HuntingState 상태 표시만 polling
            self.root.after(0, lambda: self.auto_status_label.config(
                text="[AUTO] 실행 중...", foreground="lime green"
            ))

        threading.Thread(target=sm_starter, daemon=True).start()
        self._poll_auto_status()

    def auto_stop(self):
        """자동 레벨링만 중지 (트래커는 유지)."""
        # 현재 구조상 트래커와 함께 중지
        self.stop()
        self.auto_status_label.config(text="[AUTO] 중지됨", foreground="gray")

    def _poll_auto_status(self):
        """자동 레벨링 상태를 주기적으로 표시 갱신."""
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        # 상태는 오버레이에 그려지므로 간단한 "실행 중" 메시지만 표시
        self.root.after(1000, self._poll_auto_status)

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

    def open_coord_picker(self):
        """좌표 세팅 도구를 별도 창으로 엽니다."""
        import subprocess
        picker_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "coordinate_picker.py")
        )
        # py -3.11 우선, 없으면 현재 인터프리터 fallback
        try:
            subprocess.Popen(["py", "-3.11", picker_path])
        except FileNotFoundError:
            subprocess.Popen([sys.executable, picker_path])

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    SettingsWindow().run()
