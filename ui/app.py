"""
ui/app.py — 리니지 자동화 봇 tkinter GUI.

탭 구성:
  [레벨링]  LevelingMode — 허수아비 → 사냥터 1단계
  [던전]    DungeonMode  — 던전 반복 공략
  [필드]    FieldMode    — 자유 사냥
  [설정]    Settings 편집 (JSON 뷰어 + 저장)
  [로그]    실행 로그 실시간 표시

실행:
  python -m ui.app
  또는
  from ui.app import BotApp; BotApp().run()
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

# ─────────────────────────── 로거 큐 핸들러 ──────────────────────

class _QueueHandler(logging.Handler):
    """logging 레코드를 queue 에 넣어 UI 스레드에서 소비한다."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self._q = log_queue

    def emit(self, record: logging.LogRecord):
        try:
            self._q.put_nowait(self.format(record))
        except queue.Full:
            pass


# ─────────────────────────── 색상 상수 ───────────────────────────

_BG        = "#1e1e2e"
_BG2       = "#2a2a3e"
_FG        = "#cdd6f4"
_ACCENT    = "#89b4fa"
_GREEN     = "#a6e3a1"
_RED       = "#f38ba8"
_YELLOW    = "#f9e2af"
_BORDER    = "#45475a"
_FONT_MONO = ("Consolas", 10)
_FONT_UI   = ("Segoe UI", 10)
_FONT_H    = ("Segoe UI", 12, "bold")


# ═══════════════════════════════════════════════════════════════════
#  메인 애플리케이션
# ═══════════════════════════════════════════════════════════════════

class BotApp:
    """리니지 자동화 봇 메인 GUI."""

    # ────────────────────────────── 생성 ────────────────────────────

    def __init__(self, settings=None):
        """
        Parameters
        ----------
        settings : Settings | None
            None 이면 config/settings.py 의 Settings.load() 를 호출.
        """
        if settings is None:
            try:
                from config.settings import Settings
                settings = Settings.load()
            except Exception as e:
                settings = None
                self._settings_error = str(e)
            else:
                self._settings_error = None
        else:
            self._settings_error = None

        self.settings = settings

        # ── tkinter 루트 ─────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("리니지 자동화 봇 v2")
        self.root.configure(bg=_BG)
        self.root.resizable(True, True)
        self.root.geometry("900x680")

        # ── 로그 큐 ──────────────────────────────────────────────────
        self._log_q: queue.Queue = queue.Queue(maxsize=2000)
        handler = _QueueHandler(self._log_q)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                              datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)

        # ── 런타임 상태 ───────────────────────────────────────────────
        self._mode_thread: Optional[threading.Thread] = None
        self._mode_stop   = threading.Event()
        self._active_mode = None   # LevelingMode | DungeonMode | FieldMode

        # ── Pico / Capturer (지연 초기화) ────────────────────────────
        self._pico     = None
        self._capturer = None

        # ── UI 구성 ───────────────────────────────────────────────────
        self._build_ui()
        self._poll_logs()

        if self._settings_error:
            self._log(f"⚠ 설정 로드 실패: {self._settings_error}", level="warning")

    # ────────────────────────────── UI 구성 ─────────────────────────

    def _build_ui(self):
        # 상단 타이틀
        title_bar = tk.Frame(self.root, bg=_BG, pady=6)
        title_bar.pack(fill="x")
        tk.Label(
            title_bar,
            text="⚔  리니지 자동화 봇  ⚔",
            font=("Segoe UI", 14, "bold"),
            fg=_ACCENT, bg=_BG,
        ).pack()

        # 구분선
        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        # 메인 노트북 (탭)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",       background=_BG,  borderwidth=0)
        style.configure("TNotebook.Tab",   background=_BG2, foreground=_FG,
                        padding=[12, 4], font=_FONT_UI)
        style.map("TNotebook.Tab",
                  background=[("selected", _ACCENT)],
                  foreground=[("selected", _BG)])

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=4, pady=4)
        self._nb = nb

        # 탭 생성
        self._tab_leveling = self._make_tab(nb, "레벨링")
        self._tab_dungeon  = self._make_tab(nb, "던전")
        self._tab_field    = self._make_tab(nb, "필드")
        self._tab_settings = self._make_tab(nb, "설정")
        self._tab_log      = self._make_tab(nb, "로그")

        nb.add(self._tab_leveling, text=" 레벨링 ")
        nb.add(self._tab_dungeon,  text=" 던전 ")
        nb.add(self._tab_field,    text=" 필드 ")
        nb.add(self._tab_settings, text=" 설정 ")
        nb.add(self._tab_log,      text=" 로그 ")

        # 각 탭 내용
        self._build_leveling_tab(self._tab_leveling)
        self._build_dungeon_tab(self._tab_dungeon)
        self._build_field_tab(self._tab_field)
        self._build_settings_tab(self._tab_settings)
        self._build_log_tab(self._tab_log)

        # 하단 상태바
        self._status_var = tk.StringVar(value="대기 중")
        status_bar = tk.Frame(self.root, bg=_BG2, pady=3)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self._status_var,
                 font=_FONT_UI, fg=_FG, bg=_BG2).pack(side="left", padx=8)

        # 전역 단축키
        self.root.bind("<F5>",  lambda e: self._emergency_stop())
        self.root.bind("<Escape>", lambda e: self._emergency_stop())

    def _make_tab(self, nb, name: str) -> tk.Frame:
        f = tk.Frame(nb, bg=_BG)
        return f

    # ── 레벨링 탭 ────────────────────────────────────────────────────

    def _build_leveling_tab(self, parent: tk.Frame):
        self._build_mode_tab(
            parent,
            mode_name="leveling",
            title="레벨링 모드 (허수아비 → 사냥터)",
            color=_GREEN,
            info_lines=[
                "1. 허수아비 수련장 텔레포트 (F6 두루마리)",
                "2. 허수아비 반복 공격 → Lv.목표 달성",
                "3. 속도물약(F9) → 사냥터 이동",
                "4. 사냥터 순찰 + 적 추적·공격 + 루팅",
            ],
        )

    # ── 던전 탭 ─────────────────────────────────────────────────────

    def _build_dungeon_tab(self, parent: tk.Frame):
        self._build_mode_tab(
            parent,
            mode_name="dungeon",
            title="던전 모드 (반복 공략)",
            color=_YELLOW,
            info_lines=[
                "1. 던전 입구 클릭 → 입장",
                "2. 적 탐지·공격 (tracker 주입 시)",
                "3. 아데나/드롭 루팅",
                "4. 던전 퇴장 → 쿨다운 → 반복",
            ],
            extra_widgets=self._build_dungeon_extra,
        )

    def _build_dungeon_extra(self, parent: tk.Frame):
        row = tk.Frame(parent, bg=_BG)
        row.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(row, text="최대 반복 횟수 (0=무한):", font=_FONT_UI,
                 fg=_FG, bg=_BG).pack(side="left")
        self._dungeon_max_runs = tk.IntVar(value=0)
        tk.Spinbox(row, from_=0, to=9999, textvariable=self._dungeon_max_runs,
                   width=6, font=_FONT_UI).pack(side="left", padx=4)

    # ── 필드 탭 ─────────────────────────────────────────────────────

    def _build_field_tab(self, parent: tk.Frame):
        self._build_mode_tab(
            parent,
            mode_name="field",
            title="필드 모드 (자유 사냥)",
            color=_ACCENT,
            info_lines=[
                "1. patrol_waypoints 루프 순찰",
                "2. 적 감지 → tracker SM 공격",
                "3. 루팅 후 순찰 재개",
                "4. max_kills 달성 시 종료",
            ],
            extra_widgets=self._build_field_extra,
        )

    def _build_field_extra(self, parent: tk.Frame):
        row = tk.Frame(parent, bg=_BG)
        row.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(row, text="목표 킬 수 (0=무한):", font=_FONT_UI,
                 fg=_FG, bg=_BG).pack(side="left")
        self._field_max_kills = tk.IntVar(value=0)
        tk.Spinbox(row, from_=0, to=99999, textvariable=self._field_max_kills,
                   width=6, font=_FONT_UI).pack(side="left", padx=4)

    # ── 공통 모드 탭 빌더 ────────────────────────────────────────────

    def _build_mode_tab(
        self,
        parent: tk.Frame,
        mode_name: str,
        title: str,
        color: str,
        info_lines: list,
        extra_widgets=None,
    ):
        # 제목
        tk.Label(parent, text=title, font=_FONT_H, fg=color, bg=_BG,
                 pady=10).pack()

        # 설명
        info_f = tk.LabelFrame(parent, text="흐름", font=_FONT_UI,
                               fg=_FG, bg=_BG, bd=1, relief="groove")
        info_f.pack(fill="x", padx=16, pady=(0, 8))
        for line in info_lines:
            tk.Label(info_f, text=line, font=_FONT_UI, fg=_FG, bg=_BG,
                     anchor="w").pack(fill="x", padx=8, pady=1)

        # 추가 위젯 (모드별)
        if extra_widgets:
            extra_widgets(parent)

        # Pico 연결 체크
        pico_f = tk.Frame(parent, bg=_BG)
        pico_f.pack(fill="x", padx=16, pady=(0, 8))
        tk.Button(
            pico_f, text="Pico 연결 테스트",
            command=self._test_pico_connection,
            font=_FONT_UI, bg=_BG2, fg=_FG, relief="flat", padx=8,
        ).pack(side="left", padx=4)
        pico_lbl = tk.Label(pico_f, text="●", font=("Segoe UI", 14),
                            fg=_RED, bg=_BG)
        pico_lbl.pack(side="left")
        setattr(self, f"_pico_indicator_{mode_name}", pico_lbl)

        # 시작/정지 버튼
        btn_f = tk.Frame(parent, bg=_BG)
        btn_f.pack(pady=12)

        start_btn = tk.Button(
            btn_f, text="▶ 시작",
            command=lambda m=mode_name: self._start_mode(m),
            font=("Segoe UI", 11, "bold"),
            bg=color, fg=_BG, relief="flat", padx=20, pady=6,
            cursor="hand2",
        )
        start_btn.pack(side="left", padx=6)

        stop_btn = tk.Button(
            btn_f, text="■ 정지",
            command=self._emergency_stop,
            font=("Segoe UI", 11, "bold"),
            bg=_RED, fg=_BG, relief="flat", padx=20, pady=6,
            cursor="hand2",
        )
        stop_btn.pack(side="left", padx=6)

        # 상태 표시
        state_var = tk.StringVar(value="IDLE")
        tk.Label(parent, textvariable=state_var, font=_FONT_MONO,
                 fg=_YELLOW, bg=_BG).pack()
        setattr(self, f"_state_var_{mode_name}", state_var)

    # ── 설정 탭 ─────────────────────────────────────────────────────

    def _build_settings_tab(self, parent: tk.Frame):
        tk.Label(parent, text="설정 (config.json)", font=_FONT_H,
                 fg=_ACCENT, bg=_BG, pady=10).pack()

        # 버튼 행
        btn_f = tk.Frame(parent, bg=_BG)
        btn_f.pack(fill="x", padx=16, pady=(0, 4))
        tk.Button(btn_f, text="📂 불러오기", command=self._load_settings_file,
                  font=_FONT_UI, bg=_BG2, fg=_FG, relief="flat", padx=8).pack(side="left", padx=4)
        tk.Button(btn_f, text="💾 저장", command=self._save_settings_file,
                  font=_FONT_UI, bg=_BG2, fg=_FG, relief="flat", padx=8).pack(side="left", padx=4)
        tk.Button(btn_f, text="↺ 새로고침", command=self._refresh_settings_view,
                  font=_FONT_UI, bg=_BG2, fg=_FG, relief="flat", padx=8).pack(side="left", padx=4)

        # JSON 편집기
        self._settings_text = scrolledtext.ScrolledText(
            parent, font=_FONT_MONO, bg=_BG2, fg=_FG,
            insertbackground=_FG, relief="flat",
        )
        self._settings_text.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self._refresh_settings_view()

    def _refresh_settings_view(self):
        if self.settings is None:
            self._settings_text.delete("1.0", "end")
            self._settings_text.insert("end", "# 설정 파일을 불러오세요\n")
            return
        try:
            import dataclasses
            d = dataclasses.asdict(self.settings)
            text = json.dumps(d, ensure_ascii=False, indent=2)
        except Exception as e:
            text = f"# 직렬화 오류: {e}"
        self._settings_text.delete("1.0", "end")
        self._settings_text.insert("end", text)

    def _load_settings_file(self):
        path = filedialog.askopenfilename(
            title="설정 파일 선택",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            from config.settings import Settings
            self.settings = Settings.load(path)
            self._refresh_settings_view()
            self._log(f"설정 파일 로드: {path}")
        except Exception as e:
            messagebox.showerror("오류", f"설정 로드 실패:\n{e}")

    def _save_settings_file(self):
        path = filedialog.asksaveasfilename(
            title="설정 파일 저장",
            defaultextension=".json",
            filetypes=[("JSON 파일", "*.json")],
        )
        if not path:
            return
        try:
            text = self._settings_text.get("1.0", "end").strip()
            data = json.loads(text)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._log(f"설정 저장 완료: {path}")
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{e}")

    # ── 로그 탭 ─────────────────────────────────────────────────────

    def _build_log_tab(self, parent: tk.Frame):
        tk.Label(parent, text="실행 로그", font=_FONT_H,
                 fg=_ACCENT, bg=_BG, pady=8).pack()

        btn_f = tk.Frame(parent, bg=_BG)
        btn_f.pack(fill="x", padx=16)
        tk.Button(btn_f, text="지우기", command=self._clear_log,
                  font=_FONT_UI, bg=_BG2, fg=_FG, relief="flat", padx=8).pack(side="left", padx=4)

        self._log_text = scrolledtext.ScrolledText(
            parent, font=_FONT_MONO, bg=_BG2, fg=_FG,
            state="disabled", insertbackground=_FG, relief="flat",
        )
        self._log_text.pack(fill="both", expand=True, padx=16, pady=8)

        # 태그 색상
        self._log_text.tag_config("INFO",    foreground=_FG)
        self._log_text.tag_config("DEBUG",   foreground=_BORDER)
        self._log_text.tag_config("WARNING", foreground=_YELLOW)
        self._log_text.tag_config("ERROR",   foreground=_RED)
        self._log_text.tag_config("CRITICAL",foreground=_RED)

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    # ────────────────────────────── 모드 제어 ───────────────────────

    def _get_hardware(self):
        """Pico + ScreenCapturer 지연 초기화."""
        if self._pico is None:
            try:
                from hardware.pico import PicoWorker
                self._pico = PicoWorker.from_settings(self.settings)
                self._pico.start()
            except Exception:
                from hardware.pico import NullPicoWorker
                self._pico = NullPicoWorker.from_settings(self.settings)
                self._log("⚠ Pico 연결 실패 — NullPicoWorker 사용", level="warning")

        if self._capturer is None:
            try:
                from hardware.screen import ScreenCapturer
                self._capturer = ScreenCapturer.from_settings(self.settings)
            except Exception as e:
                self._log(f"⚠ 캡처 초기화 실패: {e}", level="warning")
                self._capturer = None

        return self._pico, self._capturer

    def _start_mode(self, mode_name: str):
        if self._mode_thread and self._mode_thread.is_alive():
            messagebox.showwarning("경고", "이미 실행 중입니다.\n■ 정지 후 다시 시작하세요.")
            return

        if self.settings is None:
            messagebox.showerror("오류", "설정이 로드되지 않았습니다.")
            return

        self._mode_stop.clear()
        self._mode_thread = threading.Thread(
            target=self._run_mode,
            args=(mode_name,),
            daemon=True,
            name=f"BotMode-{mode_name}",
        )
        self._mode_thread.start()
        self._status_var.set(f"실행 중: {mode_name}")
        self._log(f"▶ {mode_name} 모드 시작")

    def _run_mode(self, mode_name: str):
        """백그라운드 스레드: 모드 FSM 루프."""
        pico, capturer = self._get_hardware()
        grab = (capturer.grab if capturer else lambda: __import__("numpy").zeros((100, 200, 3), dtype=__import__("numpy").uint8))

        try:
            mode = self._create_mode(mode_name, pico, grab)
        except Exception as e:
            self._log(f"모드 생성 실패: {e}", level="error")
            return

        self._active_mode = mode
        mode.start()

        state_var = getattr(self, f"_state_var_{mode_name}", None)
        last_t = time.monotonic()

        try:
            while not self._mode_stop.is_set():
                now = time.monotonic()
                dt = now - last_t
                last_t = now

                mode.run_tick(dt)

                if state_var:
                    try:
                        self.root.after(0, state_var.set, mode.state.name)
                    except Exception:
                        pass

                if mode.is_done:
                    self._log(f"✅ {mode_name} 완료")
                    break

                time.sleep(1.0 / 30.0)  # ~30 tick/s
        except Exception as e:
            self._log(f"모드 오류: {e}", level="error")
        finally:
            mode.stop()
            self._active_mode = None
            self.root.after(0, self._status_var.set, "대기 중")
            self._log(f"■ {mode_name} 정지")

    def _create_mode(self, mode_name: str, pico, grab):
        """모드 인스턴스 생성."""
        s = self.settings

        # 공통 perception 객체
        try:
            from perception.hp import HpReader
            hp_reader = HpReader.from_settings(s)
        except Exception:
            hp_reader = None

        try:
            from perception.loot import LootDetector
            loot_detector = LootDetector.from_settings(s)
        except Exception:
            loot_detector = None

        # tracker
        try:
            from core.tracking import NearestNeighborTracker
            tracker = NearestNeighborTracker.from_settings(
                s,
                pico_click_callback=lambda x, y: pico.click(x, y),
                pico_drag_callback=lambda fx, fy, tx, ty: pico.drag(fx, fy, tx, ty),
            )
        except Exception:
            tracker = None

        if mode_name == "leveling":
            from modes.leveling import LevelingMode
            try:
                from perception.level import LevelReader
                level_reader = LevelReader.from_settings(s)
            except Exception:
                level_reader = None
            return LevelingMode(
                settings=s, pico=pico, frame_grabber=grab,
                tracker=tracker, hp_reader=hp_reader,
                level_reader=level_reader, loot_detector=loot_detector,
            )
        elif mode_name == "dungeon":
            from modes.dungeon import DungeonMode
            max_runs = getattr(self, "_dungeon_max_runs", None)
            mr = max_runs.get() if max_runs else 0
            return DungeonMode(
                settings=s, pico=pico, frame_grabber=grab,
                tracker=tracker, hp_reader=hp_reader,
                loot_detector=loot_detector, max_runs=mr,
            )
        elif mode_name == "field":
            from modes.field import FieldMode
            max_kills = getattr(self, "_field_max_kills", None)
            mk = max_kills.get() if max_kills else 0
            return FieldMode(
                settings=s, pico=pico, frame_grabber=grab,
                tracker=tracker, hp_reader=hp_reader,
                loot_detector=loot_detector, max_kills=mk,
            )
        else:
            raise ValueError(f"알 수 없는 모드: {mode_name}")

    def _emergency_stop(self):
        """F5 / ESC / 정지 버튼 — 즉시 정지."""
        self._mode_stop.set()
        if self._active_mode:
            try:
                self._active_mode.stop()
            except Exception:
                pass
        self._log("⛔ 긴급 정지")

    def _test_pico_connection(self):
        pico, _ = self._get_hardware()
        connected = getattr(pico, "is_connected", False)
        label = "● 연결됨" if connected else "● 미연결"
        color = _GREEN if connected else _RED
        # 모든 탭의 지시자 갱신
        for mode_name in ("leveling", "dungeon", "field"):
            ind = getattr(self, f"_pico_indicator_{mode_name}", None)
            if ind:
                ind.configure(fg=color)
                ind.configure(text=label)
        self._log(f"Pico 상태: {'연결됨' if connected else '미연결'}")

    # ────────────────────────────── 로그 헬퍼 ───────────────────────

    def _log(self, msg: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        full = f"[{timestamp}] {msg}"
        self._append_log(full, level.upper())

    def _append_log(self, text: str, level: str = "INFO"):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", text + "\n", level)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _poll_logs(self):
        """50ms 마다 로그 큐를 소비해 UI 에 표시한다."""
        try:
            while True:
                msg = self._log_q.get_nowait()
                # 레벨 태그 추출
                level = "INFO"
                for lv in ("DEBUG", "WARNING", "ERROR", "CRITICAL", "INFO"):
                    if f"[{lv}]" in msg:
                        level = lv
                        break
                self._append_log(msg, level)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_logs)

    # ────────────────────────────── 실행 ────────────────────────────

    def run(self):
        """GUI 이벤트 루프 시작."""
        self.root.mainloop()

    def close(self):
        """리소스 정리 후 창 닫기."""
        self._emergency_stop()
        if self._pico:
            try:
                self._pico.stop()
            except Exception:
                pass
        if self._capturer:
            try:
                self._capturer.close()
            except Exception:
                pass
        self.root.destroy()


# ─────────────────────────── 진입점 ──────────────────────────────

def main():
    app = BotApp()
    app.root.protocol("WM_DELETE_WINDOW", app.close)
    app.run()


if __name__ == "__main__":
    main()
