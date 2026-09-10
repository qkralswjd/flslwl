"""core/tracking.py — Enemy 데이터클래스 + 최근접 이웃 트래커 + 순차 타겟 상태머신.

기존 tracking/enemy.py + tracking/tracker.py 를 단일 모듈로 통합한다.

개선점 (기존 대비):
  - Enemy / Detection / TargetState / SequentialTargetStateMachine /
    NearestNeighborTracker 모두 이 파일에서 import 가능
  - NearestNeighborTracker.from_settings() 팩토리 추가
  - SequentialTargetStateMachine 에 drag_callback 시그니처 통일
    (from_x, from_y, to_x, to_y) — 기존 동일
  - BaseTracker 는 core/state.py 가 아닌 이 파일에 정의 (의존 관계 단순화)
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from config.settings import Settings
    from core.detection import Detection

logger = logging.getLogger("core.tracking")


# ════════════════════════════════════════════════════════════════════════════
# Enemy 데이터클래스
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Enemy:
    """프레임 간 동일 적 객체. 위치·속도·소실 상태를 유지한다.

    기존 tracking/enemy.py 와 동일 로직. 별도 파일 분리 제거.
    """

    id:          int
    x:           int
    y:           int
    width:       int
    height:      int
    center_x:    int
    center_y:    int
    confidence:  float          = 1.0
    velocity_x:  float          = 0.0   # pixels/frame (last update)
    velocity_y:  float          = 0.0
    pixels_per_second_x: float  = 0.0
    pixels_per_second_y: float  = 0.0
    missing_frames: int         = 0
    predicted:   bool           = False  # True → 이번 프레임은 예측 위치
    created_at:  float          = field(default_factory=time.time)
    last_seen_at: float         = field(default_factory=time.time)
    history:     deque          = field(default_factory=lambda: deque(maxlen=30))

    # 속도 감쇠 계수 — 소실 연속 시 예측 위치가 너무 멀리 벗어나지 않도록
    VELOCITY_DECAY: float       = field(default=0.75, init=False, repr=False, compare=False)

    def update_position(
        self,
        x: int, y: int, w: int, h: int,
        dt: float,
        confidence: float = 1.0,
    ) -> None:
        """실제 탐지 결과로 위치를 갱신한다."""
        cx = x + w // 2
        cy = y + h // 2

        self.velocity_x = cx - self.center_x
        self.velocity_y = cy - self.center_y
        if dt > 0:
            self.pixels_per_second_x = self.velocity_x / dt
            self.pixels_per_second_y = self.velocity_y / dt

        self.x, self.y, self.width, self.height = x, y, w, h
        self.center_x, self.center_y = cx, cy
        self.confidence = confidence
        self.missing_frames = 0
        self.predicted = False
        self.last_seen_at = time.time()
        self.history.append((cx, cy))

    def predict_next_position(self) -> tuple[int, int]:
        """속도로 다음 위치를 외삽한다."""
        return (
            int(self.center_x + self.velocity_x),
            int(self.center_y + self.velocity_y),
        )

    def apply_prediction(self) -> None:
        """탐지 실패 프레임에 속도 기반 예측 위치를 적용한다."""
        pred_x, pred_y = self.predict_next_position()
        self.x       += pred_x - self.center_x
        self.y       += pred_y - self.center_y
        self.center_x, self.center_y = pred_x, pred_y
        self.velocity_x *= self.VELOCITY_DECAY
        self.velocity_y *= self.VELOCITY_DECAY
        self.missing_frames += 1
        self.predicted = True


# ════════════════════════════════════════════════════════════════════════════
# TargetState
# ════════════════════════════════════════════════════════════════════════════

class TargetState(Enum):
    """SequentialTargetStateMachine 상태."""
    IDLE         = auto()   # 감지된 적 없음
    LOCKING      = auto()   # 타겟 잠금 확인 중 (연속 프레임 대기)
    CLICKING     = auto()   # 클릭 신호 발사 직후 (1프레임)
    WAITING_DEAD = auto()   # 타겟이 죽을 때까지 대기
    COOLDOWN     = auto()   # 다음 타겟 전환 전 냉각


# ════════════════════════════════════════════════════════════════════════════
# SequentialTargetStateMachine
# ════════════════════════════════════════════════════════════════════════════

class SequentialTargetStateMachine:
    """한 번에 적 1명만 타겟으로 삼고, 그 적이 완전히 사라지면 다음 적으로 넘어간다.

    상태 흐름:
        IDLE → LOCKING → CLICKING → WAITING_DEAD → COOLDOWN → LOCKING → …

    타겟 선택 우선순위(priority):
        "nearest_center" : ROI 중심에서 가장 가까운 적
        "nearest_origin" : ROI 원점(좌상단)에서 가장 가까운 적
        "oldest"         : 가장 오래 화면에 있던 적 (created_at 기준)
        "newest"         : 가장 최근에 나타난 적
    """

    def __init__(
        self,
        pico_click_callback:  Optional[Callable[[int, int], None]],
        to_screen_fn:         Callable[[int, int], tuple],
        click_pulse_ms:       int   = 20,
        lock_confirm_frames:  int   = 2,
        wait_dead_timeout_ms: float = 5000.0,
        next_target_cooldown_ms: float = 300.0,
        priority:             str   = "nearest_center",
        roi_width:            int   = 1440,
        roi_height:           int   = 780,
        # ── 드래그 파라미터 ─────────────────────────────────────────
        drag_enabled: bool = False,
        drag_dx:      int  = 80,
        drag_dy:      int  = 0,
        drag_steps:   int  = 8,
        pico_drag_callback: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> None:
        self._click_cb            = pico_click_callback
        self._drag_cb             = pico_drag_callback
        self._to_screen           = to_screen_fn
        self._click_pulse_ms      = click_pulse_ms
        self._lock_confirm_frames = max(1, lock_confirm_frames)
        self._wait_dead_timeout_ms      = wait_dead_timeout_ms
        self._next_target_cooldown_ms   = next_target_cooldown_ms
        self._priority            = priority
        self._roi_cx              = roi_width  / 2.0
        self._roi_cy              = roi_height / 2.0
        self._drag_enabled        = drag_enabled
        self._drag_dx             = drag_dx
        self._drag_dy             = drag_dy
        self._drag_steps          = drag_steps

        self.state:     TargetState    = TargetState.IDLE
        self.target_id: Optional[int]  = None
        self._lock_streak:      int    = 0
        self._state_entered_at: float  = time.time()

        # 공격 활성 플래그
        # False 이면 update() 가 즉시 reset+return → Pico 명령 차단
        self.active: bool = True

    # ── 공개 API ───────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """공격 활성 여부 설정. False → 즉시 reset."""
        if not active and self.active:
            self.reset()
            logger.info("[TargetSM] set_active(False) → reset")
        self.active = active

    def reset(self) -> None:
        self.state             = TargetState.IDLE
        self.target_id         = None
        self._lock_streak      = 0
        self._state_entered_at = time.time()

    def update(self, enemies: Dict[int, Enemy]) -> None:
        """매 detection tick 마다 호출.

        active=False 이면 즉시 reset 후 return → Pico 명령 절대 미발생.
        """
        if not self.active:
            if self.state != TargetState.IDLE or self.target_id is not None:
                self.reset()
                logger.debug("[TargetSM] inactive → reset enforced")
            return

        # 실제 탐지(predicted 제외) 적만 공격 대상
        active_enemies = {eid: e for eid, e in enemies.items() if not e.predicted}
        all_tracked    = enemies  # predicted 포함 전체

        now = time.time()

        # ── IDLE ──────────────────────────────────────────────────────────
        if self.state == TargetState.IDLE:
            if active_enemies:
                self._pick_target(active_enemies, now)
                self._enter(TargetState.LOCKING, now)
            return

        # ── LOCKING ───────────────────────────────────────────────────────
        if self.state == TargetState.LOCKING:
            if self.target_id not in active_enemies:
                if active_enemies:
                    prev_id = self.target_id
                    self._pick_target(active_enemies, now)
                    if self.target_id != prev_id:
                        self._lock_streak = 0
                        logger.debug(
                            "[TargetSM] LOCKING 타겟 변경 #%s→#%s streak reset",
                            prev_id, self.target_id,
                        )
                else:
                    self._enter(TargetState.IDLE, now)
                return

            self._lock_streak += 1
            logger.debug(
                "[TargetSM] LOCKING #%d streak=%d/%d",
                self.target_id, self._lock_streak, self._lock_confirm_frames,
            )

            if self._lock_streak >= self._lock_confirm_frames:
                self._fire_click(active_enemies[self.target_id])
                self._enter(TargetState.CLICKING, now)
            return

        # ── CLICKING (1프레임 통과) ────────────────────────────────────────
        if self.state == TargetState.CLICKING:
            self._enter(TargetState.WAITING_DEAD, now)
            return

        # ── WAITING_DEAD ──────────────────────────────────────────────────
        if self.state == TargetState.WAITING_DEAD:
            target_gone = self.target_id not in all_tracked
            timed_out   = (now - self._state_entered_at) * 1000.0 >= self._wait_dead_timeout_ms

            if target_gone:
                logger.info("[TargetSM] 타겟 #%d 사망 확인 → 다음 타겟 탐색", self.target_id)
                self._enter(TargetState.COOLDOWN, now)
            elif timed_out:
                logger.warning("[TargetSM] 타겟 #%d 대기 타임아웃 → 강제 전환", self.target_id)
                self._enter(TargetState.COOLDOWN, now)
            return

        # ── COOLDOWN ──────────────────────────────────────────────────────
        if self.state == TargetState.COOLDOWN:
            elapsed_ms = (now - self._state_entered_at) * 1000.0
            if elapsed_ms >= self._next_target_cooldown_ms:
                if active_enemies:
                    self._pick_target(active_enemies, now)
                    self._enter(TargetState.LOCKING, now)
                else:
                    self._enter(TargetState.IDLE, now)
            return

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _enter(self, new_state: TargetState, now: float) -> None:
        if new_state == TargetState.LOCKING and self.state != TargetState.LOCKING:
            self._lock_streak = 0
        self.state             = new_state
        self._state_entered_at = now
        logger.info("[TargetSM] → %s  target=#%s", new_state.name, self.target_id)

    def _pick_target(self, active: Dict[int, Enemy], _now: float) -> None:
        """우선순위 전략에 따라 타겟 Enemy 를 선택한다."""
        lst = list(active.values())

        if self._priority == "nearest_center":
            best = min(lst, key=lambda e: math.hypot(
                e.center_x - self._roi_cx, e.center_y - self._roi_cy))
        elif self._priority == "nearest_origin":
            best = min(lst, key=lambda e: math.hypot(e.center_x, e.center_y))
        elif self._priority == "oldest":
            best = min(lst, key=lambda e: e.created_at)
        elif self._priority == "newest":
            best = max(lst, key=lambda e: e.created_at)
        else:
            best = min(lst, key=lambda e: math.hypot(
                e.center_x - self._roi_cx, e.center_y - self._roi_cy))

        self.target_id = best.id
        logger.info(
            "[TargetSM] 타겟 선택 #%d (%d,%d) 방식=%s",
            self.target_id, best.center_x, best.center_y, self._priority,
        )

    def _fire_click(self, enemy: Enemy) -> None:
        """Pico 에 클릭(또는 드래그) 신호를 발사한다."""
        sx, sy = self._to_screen(enemy.center_x, enemy.center_y)

        if self._drag_enabled and self._drag_cb is not None:
            tx = sx + self._drag_dx
            ty = sy + self._drag_dy
            logger.info(
                "[TargetSM] DRAG → Enemy #%d screen=(%d,%d)→(%d,%d)",
                enemy.id, sx, sy, tx, ty,
            )
            self._drag_cb(sx, sy, tx, ty)
        elif self._click_cb is not None:
            logger.info(
                "[TargetSM] CLICK → Enemy #%d screen=(%d,%d)",
                enemy.id, sx, sy,
            )
            self._click_cb(sx, sy)


# ════════════════════════════════════════════════════════════════════════════
# BaseTracker
# ════════════════════════════════════════════════════════════════════════════

class BaseTracker:
    """트래커 인터페이스. 하위 클래스에서 update() 를 구현한다."""

    def update(self, detections: List["Detection"], dt: float) -> List[Enemy]:
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# NearestNeighborTracker
# ════════════════════════════════════════════════════════════════════════════

class NearestNeighborTracker(BaseTracker):
    """최근접 이웃 매칭 트래커 + SequentialTargetStateMachine 연동.

    기존 tracker.py 와 동일 로직. from_settings() 팩토리 추가.
    """

    def __init__(
        self,
        max_missing_frames:      int   = 10,
        max_match_distance:      int   = 100,
        # ── Pico 연동 ────────────────────────────────────────────────
        pico_click_callback:     Optional[Callable[[int, int], None]] = None,
        pico_drag_callback:      Optional[Callable[[int, int, int, int], None]] = None,
        roi_offset:              tuple = (0, 0),
        capture_region_offset:   tuple = (0, 0),
        # ── 순차 타겟 상태머신 ────────────────────────────────────────
        lock_confirm_frames:     int   = 2,
        wait_dead_timeout_ms:    float = 5000.0,
        next_target_cooldown_ms: float = 300.0,
        target_priority:         str   = "nearest_center",
        roi_width:               int   = 1440,
        roi_height:              int   = 780,
        # ── 드래그 ───────────────────────────────────────────────────
        drag_enabled: bool = False,
        drag_dx:      int  = 80,
        drag_dy:      int  = 0,
        drag_steps:   int  = 8,
    ) -> None:
        self.max_missing_frames   = max_missing_frames
        self.max_match_distance   = max_match_distance
        self._roi_offset          = roi_offset
        self._capture_region_offset = capture_region_offset

        self.enemies: Dict[int, Enemy] = {}
        self._next_id = 1

        self._sm = SequentialTargetStateMachine(
            pico_click_callback     = pico_click_callback,
            pico_drag_callback      = pico_drag_callback,
            to_screen_fn            = self._to_screen,
            lock_confirm_frames     = lock_confirm_frames,
            wait_dead_timeout_ms    = wait_dead_timeout_ms,
            next_target_cooldown_ms = next_target_cooldown_ms,
            priority                = target_priority,
            roi_width               = roi_width,
            roi_height              = roi_height,
            drag_enabled            = drag_enabled,
            drag_dx                 = drag_dx,
            drag_dy                 = drag_dy,
            drag_steps              = drag_steps,
        )

    # ── 팩토리 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        pico_click_callback: Optional[Callable[[int, int], None]] = None,
        pico_drag_callback:  Optional[Callable[[int, int, int, int], None]] = None,
    ) -> "NearestNeighborTracker":
        """Settings 객체로부터 트래커를 생성한다."""
        t  = settings.tracking
        c  = settings.capture
        d  = settings.detection

        # ROI 오프셋: 탐지 존이 활성화된 경우 존 기준점 사용
        if settings.detection.zone_enabled:
            rect = settings.detection_zone_rect
            roi_off = (rect[0], rect[1]) if rect else (0, 0)
        else:
            roi_off = (0, 0)

        return cls(
            max_missing_frames      = t.max_missing_frames,
            max_match_distance      = t.max_match_distance,
            pico_click_callback     = pico_click_callback,
            pico_drag_callback      = pico_drag_callback,
            roi_offset              = roi_off,
            capture_region_offset   = (c.region_x, c.region_y),
            lock_confirm_frames     = t.lock_confirm_frames,
            wait_dead_timeout_ms    = t.wait_dead_timeout_ms,
            next_target_cooldown_ms = t.next_target_cooldown_ms,
            target_priority         = t.priority,
            roi_width               = d.zone_half_w * 2 if d.zone_enabled else c.region_w,
            roi_height              = d.zone_half_h * 2 if d.zone_enabled else c.region_h,
            drag_enabled            = t.drag_enabled,
            drag_dx                 = t.drag_dx,
            drag_dy                 = t.drag_dy,
            drag_steps              = t.drag_steps,
        )

    # ── 좌표 변환 ──────────────────────────────────────────────────────────

    def _to_screen(self, roi_x: int, roi_y: int) -> tuple:
        """ROI 좌표 → 실제 스크린 절대 좌표."""
        sx = roi_x + self._roi_offset[0] + self._capture_region_offset[0]
        sy = roi_y + self._roi_offset[1] + self._capture_region_offset[1]
        return sx, sy

    # ── 상태 노출 (overlay / UI 용) ────────────────────────────────────────

    @property
    def target_state(self) -> TargetState:
        return self._sm.state

    @property
    def current_target_id(self) -> Optional[int]:
        return self._sm.target_id

    @property
    def attack_active(self) -> bool:
        return self._sm.active

    def set_active(self, active: bool) -> None:
        """공격 활성 여부를 상태머신에 위임한다."""
        self._sm.set_active(active)

    # ── 메인 업데이트 ───────────────────────────────────────────────────────

    def update(self, detections: List["Detection"], dt: float) -> List[Enemy]:
        """매 detection tick 마다 호출한다.

        Args:
            detections: 이번 프레임 탐지 결과
            dt:         이전 프레임과의 시간 간격 (초)

        Returns:
            현재 추적 중인 Enemy 리스트 (predicted 포함)
        """
        # ── 매칭 ──────────────────────────────────────────────────────────
        candidate_pairs = []
        for eid, enemy in self.enemies.items():
            for di, det in enumerate(detections):
                dist = math.hypot(
                    enemy.center_x - det.center_x,
                    enemy.center_y - det.center_y,
                )
                if dist <= self.max_match_distance:
                    candidate_pairs.append((dist, eid, di))
        candidate_pairs.sort(key=lambda p: p[0])

        matched_eids = set()
        matched_dis  = set()
        for dist, eid, di in candidate_pairs:
            if eid in matched_eids or di in matched_dis:
                continue
            matched_eids.add(eid)
            matched_dis.add(di)
            det = detections[di]
            self.enemies[eid].update_position(
                det.x, det.y, det.width, det.height, dt, det.confidence
            )
            e = self.enemies[eid]
            logger.debug("Enemy #%d MOVE (%d,%d)", eid, e.center_x, e.center_y)

        # ── 소실 예측 / 완전 제거 ──────────────────────────────────────────
        for eid in list(self.enemies.keys()):
            if eid in matched_eids:
                continue
            enemy = self.enemies[eid]
            enemy.apply_prediction()
            if enemy.missing_frames > self.max_missing_frames:
                logger.info("Enemy #%d LOST", eid)
                del self.enemies[eid]

        # ── 신규 적 등록 ────────────────────────────────────────────────────
        for di, det in enumerate(detections):
            if di in matched_dis:
                continue
            eid = self._next_id
            self._next_id += 1
            self.enemies[eid] = Enemy(
                id=eid,
                x=det.x, y=det.y, width=det.width, height=det.height,
                center_x=det.center_x, center_y=det.center_y,
                confidence=det.confidence,
            )
            logger.info("Enemy #%d CREATED (%d,%d)", eid, det.center_x, det.center_y)

        # ── 순차 타겟 상태머신 업데이트 ────────────────────────────────────
        self._sm.update(self.enemies)

        return list(self.enemies.values())
