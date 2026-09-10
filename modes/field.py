"""modes/field.py — 필드 자유 사냥 모드.

실제 동작 구조
──────────────
    FieldMode.start()
        └── HuntLoop.start()  ← 별도 스레드: 탐지→추적→공격→루팅 무한 루프
        └── _patrol 루프       ← FSM tick: 순찰 이동 (적 없을 때)

FSM 상태
────────
    IDLE → PATROLLING ↔ ENGAGING → LOOTING → PATROLLING → ... → DONE

변경 이력
─────────
    v2: HuntLoop 연동. detector.detect() 를 FSM 내부에서 호출하던 방식 제거.
        탐지/공격은 전적으로 HuntLoop 가 담당하고,
        FSM 은 순찰·루팅 전환 로직만 관리한다.
"""
from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from core.state import BaseFSM

log = logging.getLogger(__name__)


class FieldState(Enum):
    IDLE       = auto()
    PATROLLING = auto()
    ENGAGING   = auto()
    LOOTING    = auto()
    DONE       = auto()


class FieldMode(BaseFSM):
    """필드 자유 사냥 FSM.

    Parameters
    ----------
    settings      : Settings
    pico          : PicoWorker | NullPicoWorker
    frame_grabber : () -> np.ndarray
    tracker       : NearestNeighborTracker | None   (HuntLoop 내부에서도 사용)
    hp_reader     : HpReader | None
    loot_detector : LootDetector | None
    detector      : RealtimeTemplateDetector | None  (None 이면 build_hunt_loop 내부 생성)
    max_kills     : int   0 = 무한
    base_dir      : str   templates_dir 해석 기준 경로
    """

    def __init__(
        self,
        settings,
        pico,
        frame_grabber: Callable[[], np.ndarray],
        tracker=None,
        hp_reader=None,
        loot_detector=None,
        detector=None,
        max_kills: int = 0,
        base_dir: str = ".",
    ) -> None:
        super().__init__()
        self.settings      = settings
        self.pico          = pico
        self.grab          = frame_grabber
        self.tracker       = tracker
        self.hp_reader     = hp_reader
        self.loot_detector = loot_detector
        self._detector     = detector
        self.max_kills     = max_kills
        self._base_dir     = base_dir

        # ── 키 설정 ──────────────────────────────────────────────────
        keys = getattr(settings, "keys", None)
        self._key_potion = getattr(keys, "potion", "F5") if keys else "F5"
        self._potion_cd  = getattr(keys, "potion_cooldown_ms", 3000) / 1000.0 if keys else 3.0
        self._last_potion = 0.0

        # ── 순찰 무버 ────────────────────────────────────────────────
        from modes.leveling import _build_waypoint_mover
        self._patrol = _build_waypoint_mover(settings, "patrol_waypoints", loop=True)

        # ── 교전 타임아웃 ─────────────────────────────────────────────
        pc_cfg = getattr(settings, "patrol_combat", None)
        self._engage_timeout = (
            getattr(pc_cfg, "engage_timeout_ms", 15000) if pc_cfg else 15000
        ) / 1000.0

        # ── 루팅 설정 ─────────────────────────────────────────────────
        loot_cfg = getattr(settings, "loot", None)
        self._loot_timeout    = getattr(loot_cfg, "timeout_ms", 8000) / 1000.0 if loot_cfg else 8.0
        self._loot_rescan_max = getattr(loot_cfg, "rescan_max", 3) if loot_cfg else 3
        self._loot_rescan_cnt = 0

        # ── HuntLoop (탐지/공격/루팅 엔진) ───────────────────────────
        self._hunt_loop = None   # start() 에서 생성
        self._kill_count = 0
        self._running    = False

    # ── 공개 API ──────────────────────────────────────────────────────

    def start(self) -> None:
        if not self.is_state(FieldState.IDLE):
            return
        self._running = True
        self._kill_count = 0

        # HuntLoop 생성 + 시작
        self._build_and_start_hunt_loop()

        # 순찰 무버 리셋
        self._patrol.start()
        self.transition(FieldState.PATROLLING)

    def stop(self) -> None:
        self._running = False
        if self._hunt_loop:
            self._hunt_loop.stop()
            self._hunt_loop = None
        self.transition(FieldState.IDLE, force=True)

    @property
    def kill_count(self) -> int:
        return self._kill_count

    @property
    def is_done(self) -> bool:
        return self.is_state(FieldState.DONE)

    # ── BaseFSM 구현 ──────────────────────────────────────────────────

    def _initial_state(self):
        return FieldState.IDLE

    def tick(self, dt: float) -> None:
        if not self._running and not self.is_state(FieldState.IDLE):
            return

        s = self.state
        if s == FieldState.PATROLLING:
            self._tick_patrolling()
        elif s == FieldState.ENGAGING:
            self._tick_engaging()
        elif s == FieldState.LOOTING:
            self._tick_looting()

    # ── on_enter 훅 ───────────────────────────────────────────────────

    def on_enter_PATROLLING(self):
        log.info("[Field] PATROLLING 시작")
        if self._hunt_loop:
            self._hunt_loop.set_active(True)

    def on_enter_ENGAGING(self):
        log.info("[Field] ENGAGING 시작")
        if self._hunt_loop:
            self._hunt_loop.set_active(True)

    def on_enter_LOOTING(self):
        log.info("[Field] LOOTING 시작")
        self._loot_rescan_cnt = 0
        # 루팅 중에는 공격 일시 중지 (HuntLoop 루팅 기능은 계속 동작)
        if self.tracker:
            self.tracker.set_active(False)
        if self.loot_detector:
            self.loot_detector.invalidate()

    def on_exit_LOOTING(self):
        if self.loot_detector:
            self.loot_detector.invalidate()
        if self.tracker:
            self.tracker.set_active(True)

    def on_enter_DONE(self):
        log.info("[Field] DONE — 총 %d킬", self._kill_count)
        self._running = False
        if self._hunt_loop:
            self._hunt_loop.stop()

    # ── 상태별 tick ───────────────────────────────────────────────────

    def _tick_patrolling(self):
        """순찰 이동. HuntLoop 가 적을 발견하면 ENGAGING 으로 전환."""
        # HuntLoop 의 tracker SM 이 공격을 시작했으면 ENGAGING
        if self.tracker and self.tracker.target_state.name not in ("IDLE",):
            self.transition(FieldState.ENGAGING)
            return

        # 루팅 아이템 감지
        if self.loot_detector:
            frame = self.grab()
            if frame is not None:
                items = self.loot_detector.find(frame)
                if items:
                    self.transition(FieldState.LOOTING)
                    return

        # 순찰 이동
        self._patrol.tick(self.pico)

    def _tick_engaging(self):
        """HuntLoop 가 자동 공격 중. SM 이 IDLE/COOLDOWN 복귀 시 루팅/순찰."""
        # 적 사라짐 → 킬 카운트 + 순찰 재개
        if self.tracker and self.tracker.target_state.name == "IDLE":
            self._kill_count = self._hunt_loop.kill_count if self._hunt_loop else self._kill_count
            log.info("[Field] 교전 종료 — 누적 %d킬", self._kill_count)
            self._check_done()
            if not self.is_state(FieldState.DONE):
                # 루팅 아이템 체크
                if self.loot_detector:
                    frame = self.grab()
                    if frame is not None and self.loot_detector.find(frame):
                        self.transition(FieldState.LOOTING)
                        return
                self.transition(FieldState.PATROLLING)
            return

        # 타임아웃
        if self.state_elapsed_s >= self._engage_timeout:
            log.warning("[Field] 교전 타임아웃 %.0fs → PATROLLING", self._engage_timeout)
            self.transition(FieldState.PATROLLING)

    def _tick_looting(self):
        """HuntLoop 의 루팅 기능으로 아데나 줍기. 시간 초과 시 순찰 재개."""
        if self.state_elapsed_s >= self._loot_timeout:
            log.info("[Field] 루팅 타임아웃 → PATROLLING")
            self.transition(FieldState.PATROLLING)
            return

        frame = self.grab()
        if frame is None:
            return

        if not self.loot_detector:
            self.transition(FieldState.PATROLLING)
            return

        item = self.loot_detector.find_nearest(frame, frame.shape[1] // 2, frame.shape[0] // 2)
        if item:
            self.pico.click(item.cx, item.cy, pulse_ms=50)
            self.loot_detector.invalidate()
        else:
            self._loot_rescan_cnt += 1
            if self._loot_rescan_cnt >= self._loot_rescan_max:
                log.info("[Field] 루팅 완료 → PATROLLING")
                self.transition(FieldState.PATROLLING)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _build_and_start_hunt_loop(self) -> None:
        """HuntLoop 인스턴스를 생성하고 백그라운드 스레드를 시작한다."""
        from modes.hunt_loop import build_hunt_loop

        def _on_kill(cnt: int):
            self._kill_count = cnt
            self._check_done()

        self._hunt_loop = build_hunt_loop(
            settings    = self.settings,
            pico        = self.pico,
            grab_fn     = self.grab,
            on_kill_cb  = _on_kill,
            base_dir    = self._base_dir,
        )

        # tracker / loot_detector 를 HuntLoop 와 공유
        if self.tracker is None:
            self.tracker = self._hunt_loop._tracker
        if self.loot_detector is None:
            self.loot_detector = self._hunt_loop._loot

        self._hunt_loop.start()
        log.info("[Field] HuntLoop 시작 완료")

    def _check_hp(self) -> None:
        """HP 낮으면 물약 사용. HuntLoop 가 없는 상태에서도 직접 호출 가능."""
        if self.hp_reader is None or not self._running:
            return
        now = time.monotonic()
        if now - self._last_potion < self._potion_cd:
            return
        try:
            frame = self.grab()
            ratio = self.hp_reader.read(frame)
            if ratio is not None and self.hp_reader.is_low():
                self.pico.key_tap_name(self._key_potion, hold_ms=50)
                self._last_potion = now
                log.info("[Field] HP 낮음 → %s 물약 사용", self._key_potion)
        except Exception as e:
            log.debug("[Field] _check_hp 예외: %s", e)

    def _check_done(self):
        if self.max_kills > 0 and self._kill_count >= self.max_kills:
            log.info("[Field] max_kills=%d 달성 → DONE", self.max_kills)
            self.transition(FieldState.DONE)
