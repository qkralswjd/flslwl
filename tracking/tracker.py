"""Frame-to-frame enemy tracking + Pico 단일 타겟 순차 공격 상태머신.

────────────────────────────────────────────────────────────
단일 타겟 순차 공격 로직 (SequentialTargetStateMachine)
────────────────────────────────────────────────────────────

상태 흐름:

    IDLE
     │  적이 1명 이상 화면에 나타남
     ▼
    LOCKING  ←──────────────────────────────────────┐
     │  lock_confirm_frames 연속으로 동일 적 확인됨  │
     ▼                                              │
    CLICKING                                        │
     │  Pico에 클릭 신호 전송 (1회)                  │
     ▼                                              │
    WAITING_DEAD                                    │
     │  현재 타겟 Enemy가 LOST (화면에서 완전 사라짐)  │
     │  OR  wait_dead_timeout_ms 초과 (타임아웃)     │
     ▼                                              │
    COOLDOWN                                        │
     │  next_target_cooldown_ms 대기                │
     └─────────────────────────────────────────────┘
          다음 적 선택 → LOCKING

타겟 선택 우선순위 (priority):
    "nearest_center"  : 화면 중심(ROI 중심)에서 가장 가까운 적
    "nearest_origin"  : ROI 원점(0,0)에서 가장 가까운 적 (좌상단 우선)
    "oldest"          : 가장 오래 화면에 있었던 적 (created_at 기준)
    "newest"          : 가장 최근에 나타난 적

WAITING_DEAD 단계에서 타겟이 predicted 상태(일시 소실)인지
완전 LOST인지를 구분합니다:
    - predicted=True  : 아직 tracker에 살아있음 → 계속 기다림
    - enemies에 없음   : 완전 소실 → 다음 타겟으로 전환
"""

import logging
import math
import time
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from tracking.enemy import Enemy

logger = logging.getLogger("tracker")


# ────────────────────────────────────────────────────────────
#  상태 정의
# ────────────────────────────────────────────────────────────

class TargetState(Enum):
    IDLE          = auto()   # 감지된 적 없음
    LOCKING       = auto()   # 타겟 잠금 확인 중 (연속 프레임 대기)
    CLICKING      = auto()   # 클릭 신호 발사 직후 (1프레임)
    WAITING_DEAD  = auto()   # 타겟이 죽을 때까지 대기
    COOLDOWN      = auto()   # 다음 타겟 전환 전 냉각


# ────────────────────────────────────────────────────────────
#  순차 타겟 상태머신
# ────────────────────────────────────────────────────────────

class SequentialTargetStateMachine:
    """
    한 번에 적 1명만 타겟으로 삼고,
    그 적이 완전히 사라지면 다음 적으로 넘어갑니다.
    """

    def __init__(
        self,
        pico_click_callback: Optional[Callable[[int, int], None]],
        to_screen_fn: Callable[[int, int], tuple],
        click_pulse_ms: int = 20,
        lock_confirm_frames: int = 2,
        wait_dead_timeout_ms: float = 5000.0,
        next_target_cooldown_ms: float = 300.0,
        priority: str = "nearest_center",
        roi_width: int = 1440,
        roi_height: int = 780,
        # ── 드래그 파라미터 ────────────────────────
        drag_enabled: bool  = False,
        drag_dx: int        = 80,    # 드래그 거리 X (px, 양수=오른쪽)
        drag_dy: int        = 0,     # 드래그 거리 Y (px, 양수=아래)
        drag_steps: int     = 8,     # 드래그 중간 단계 수
        pico_drag_callback: Optional[Callable[[int,int,int,int], None]] = None,
    ):
        self._click_cb              = pico_click_callback
        self._drag_cb               = pico_drag_callback
        self._to_screen             = to_screen_fn
        self._click_pulse_ms        = click_pulse_ms
        self._lock_confirm_frames   = max(1, lock_confirm_frames)
        self._wait_dead_timeout_ms  = wait_dead_timeout_ms
        self._next_target_cooldown_ms = next_target_cooldown_ms
        self._priority              = priority
        self._roi_cx                = roi_width  / 2.0
        self._roi_cy                = roi_height / 2.0
        self._drag_enabled          = drag_enabled
        self._drag_dx               = drag_dx
        self._drag_dy               = drag_dy
        self._drag_steps            = drag_steps

        self.state: TargetState            = TargetState.IDLE
        self.target_id: Optional[int]      = None   # 현재 타겟 Enemy ID
        self._lock_streak: int             = 0      # LOCKING 연속 확인 카운트
        self._state_entered_at: float      = time.time()

        # ── 공격 활성 플래그 ────────────────────────────────────────────
        # set_active(False) 하면 update() 호출 자체가 reset+return 됨
        # HuntingStateMachine이 HUNTING_10일 때만 True로 설정한다
        self.active: bool = True

    # ── 외부 API ─────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """공격 활성 여부를 설정한다.

        active=False 이면 update() 가 즉시 reset+return 하여
        어떤 경로로도 Pico 공격 명령이 나가지 않는다.
        """
        if not active and self.active:
            # 비활성 전환 시 즉시 초기화
            self.reset()
            logger.info("[SM] set_active(False) → reset")
        self.active = active

    def reset(self) -> None:
        self.state      = TargetState.IDLE
        self.target_id  = None
        self._lock_streak = 0
        self._state_entered_at = time.time()

    def update(self, enemies: Dict[int, Enemy]) -> None:
        """매 detection tick마다 호출. enemies = tracker의 현재 Enemy dict.

        active=False 이면 즉시 reset 후 return → Pico 명령 절대 미발생.
        """
        # ── 방어 코드: active=False 이면 공격 일체 차단 ────────────────
        if not self.active:
            if self.state != TargetState.IDLE or self.target_id is not None:
                self.reset()
                logger.debug("[SM] inactive → reset enforced")
            return

        # 살아있는 적만 대상 (predicted 포함 — 잠깐 소실은 계속 추적)
        active = {eid: e for eid, e in enemies.items() if not e.predicted}
        all_tracked = enemies  # predicted 포함 전체

        now = time.time()

        # ── IDLE ──────────────────────────────────────────────────────
        if self.state == TargetState.IDLE:
            if active:
                self._pick_target(active, now)
                self._enter(TargetState.LOCKING, now)
            return

        # ── LOCKING ───────────────────────────────────────────────────
        if self.state == TargetState.LOCKING:
            if self.target_id not in active:
                # 타겟이 사라졌거나 predicted 상태
                if active:
                    prev_id = self.target_id
                    self._pick_target(active, now)
                    # 같은 타겟이면 streak 유지, 다른 타겟이면 리셋
                    if self.target_id != prev_id:
                        self._lock_streak = 0
                        logger.debug(f"[SM] LOCKING 타겟 변경 #{prev_id}→#{self.target_id} streak reset")
                    # else: streak 유지 → 이미 누적된 count 살림
                else:
                    self._enter(TargetState.IDLE, now)
                return

            self._lock_streak += 1
            logger.debug(f"[SM] LOCKING #{self.target_id} streak={self._lock_streak}/{self._lock_confirm_frames}")

            if self._lock_streak >= self._lock_confirm_frames:
                self._fire_click(active[self.target_id])
                self._enter(TargetState.CLICKING, now)
            return

        # ── CLICKING (1프레임 통과) ────────────────────────────────────
        if self.state == TargetState.CLICKING:
            self._enter(TargetState.WAITING_DEAD, now)
            return

        # ── WAITING_DEAD ──────────────────────────────────────────────
        if self.state == TargetState.WAITING_DEAD:
            target_gone = self.target_id not in all_tracked  # tracker에서 완전 삭제됨
            timed_out   = (now - self._state_entered_at) * 1000.0 >= self._wait_dead_timeout_ms

            if target_gone:
                logger.info(f"[SM] 타겟 #{self.target_id} 사망 확인 → 다음 타겟 탐색")
                self._enter(TargetState.COOLDOWN, now)
            elif timed_out:
                logger.warning(f"[SM] 타겟 #{self.target_id} 대기 타임아웃 → 강제 전환")
                self._enter(TargetState.COOLDOWN, now)
            return

        # ── COOLDOWN ──────────────────────────────────────────────────
        if self.state == TargetState.COOLDOWN:
            elapsed_ms = (now - self._state_entered_at) * 1000.0
            if elapsed_ms >= self._next_target_cooldown_ms:
                if active:
                    self._pick_target(active, now)
                    self._enter(TargetState.LOCKING, now)
                else:
                    self._enter(TargetState.IDLE, now)
            return

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _enter(self, new_state: TargetState, now: float) -> None:
        # LOCKING 진입 시 streak 리셋 — 단, 이미 LOCKING 상태에서 같은 타겟
        # 유지 중 재호출되는 경우는 위 LOCKING 블록에서 직접 처리하므로
        # 여기서는 상태 전환(다른 state → LOCKING)일 때만 리셋
        if new_state == TargetState.LOCKING and self.state != TargetState.LOCKING:
            self._lock_streak = 0
        self.state = new_state
        self._state_entered_at = now
        logger.info(f"[SM] → {new_state.name}  target=#{self.target_id}")

    def _pick_target(self, active: Dict[int, Enemy], now: float) -> None:
        """우선순위 전략에 따라 타겟 Enemy를 선택합니다."""
        enemies_list = list(active.values())

        if self._priority == "nearest_center":
            best = min(enemies_list, key=lambda e: math.hypot(
                e.center_x - self._roi_cx, e.center_y - self._roi_cy))

        elif self._priority == "nearest_origin":
            best = min(enemies_list, key=lambda e: math.hypot(e.center_x, e.center_y))

        elif self._priority == "oldest":
            best = min(enemies_list, key=lambda e: e.created_at)

        elif self._priority == "newest":
            best = max(enemies_list, key=lambda e: e.created_at)

        else:  # 기본값 = nearest_center
            best = min(enemies_list, key=lambda e: math.hypot(
                e.center_x - self._roi_cx, e.center_y - self._roi_cy))

        self.target_id = best.id
        logger.info(f"[SM] 타겟 선택 #{self.target_id} ({best.center_x},{best.center_y}) "
                    f"방식={self._priority}")

    def _fire_click(self, enemy: Enemy) -> None:
        """Pico에 클릭(또는 드래그) 신호를 발사합니다."""
        sx, sy = self._to_screen(enemy.center_x, enemy.center_y)

        if self._drag_enabled and self._drag_cb is not None:
            # 드래그 모드: 시작점에서 (dx, dy) 만큼 드래그
            tx = sx + self._drag_dx
            ty = sy + self._drag_dy
            logger.info(
                f"[SM] 🔄 DRAG → Enemy #{enemy.id} "
                f"screen=({sx},{sy}) → ({tx},{ty})"
            )
            self._drag_cb(sx, sy, tx, ty)
        elif self._click_cb is not None:
            # 일반 클릭 모드
            logger.info(f"[SM] 🖱 CLICK → Enemy #{enemy.id} screen=({sx},{sy})")
            self._click_cb(sx, sy)


# ────────────────────────────────────────────────────────────
#  BaseTracker / NearestNeighborTracker
# ────────────────────────────────────────────────────────────

class BaseTracker:
    def update(self, detections, dt) -> List[Enemy]:
        raise NotImplementedError


class NearestNeighborTracker(BaseTracker):
    """최근접 이웃 매칭 트래커 + SequentialTargetStateMachine 연동."""

    def __init__(
        self,
        max_missing_frames: int = 10,
        max_match_distance: int = 100,
        # ── Pico 연동 파라미터 ──────────────────────────────────────
        pico_click_callback: Optional[Callable[[int, int], None]] = None,
        pico_drag_callback: Optional[Callable[[int,int,int,int], None]] = None,
        roi_offset: tuple = (0, 0),
        capture_region_offset: tuple = (0, 0),
        # ── 순차 타겟 상태머신 파라미터 ─────────────────────────────
        lock_confirm_frames: int = 2,
        wait_dead_timeout_ms: float = 5000.0,
        next_target_cooldown_ms: float = 300.0,
        target_priority: str = "nearest_center",
        roi_width: int = 1440,
        roi_height: int = 780,
        # ── 드래그 파라미터 ──────────────────────────────────────────
        drag_enabled: bool = False,
        drag_dx: int = 80,
        drag_dy: int = 0,
        drag_steps: int = 8,
    ):
        self.max_missing_frames = max_missing_frames
        self.max_match_distance = max_match_distance
        self._roi_offset            = roi_offset
        self._capture_region_offset = capture_region_offset

        self.enemies: Dict[int, Enemy] = {}
        self._next_id = 1

        # 순차 타겟 상태머신 초기화
        self._sm = SequentialTargetStateMachine(
            pico_click_callback   = pico_click_callback,
            pico_drag_callback    = pico_drag_callback,
            to_screen_fn          = self._to_screen,
            lock_confirm_frames   = lock_confirm_frames,
            wait_dead_timeout_ms  = wait_dead_timeout_ms,
            next_target_cooldown_ms = next_target_cooldown_ms,
            priority              = target_priority,
            roi_width             = roi_width,
            roi_height            = roi_height,
            drag_enabled          = drag_enabled,
            drag_dx               = drag_dx,
            drag_dy               = drag_dy,
            drag_steps            = drag_steps,
        )

    # ── 좌표 변환 ──────────────────────────────────────────────────

    def _to_screen(self, roi_x: int, roi_y: int) -> tuple:
        """ROI 좌표 → 실제 스크린 절대 좌표."""
        sx = roi_x + self._roi_offset[0] + self._capture_region_offset[0]
        sy = roi_y + self._roi_offset[1] + self._capture_region_offset[1]
        return sx, sy

    # ── 상태머신 노출 (overlay, UI용) ─────────────────────────────

    @property
    def target_state(self) -> TargetState:
        return self._sm.state

    @property
    def current_target_id(self) -> Optional[int]:
        return self._sm.target_id

    # ── 메인 업데이트 ───────────────────────────────────────────────

    def update(self, detections, dt) -> List[Enemy]:

        # ── 기존 매칭 로직 (원본 100% 동일) ─────────────────────────
        candidate_pairs = []
        for eid, enemy in self.enemies.items():
            for di, det in enumerate(detections):
                dist = math.hypot(enemy.center_x - det.center_x,
                                  enemy.center_y - det.center_y)
                if dist <= self.max_match_distance:
                    candidate_pairs.append((dist, eid, di))
        candidate_pairs.sort(key=lambda p: p[0])

        matched_enemy_ids     = set()
        matched_detection_idx = set()
        for dist, eid, di in candidate_pairs:
            if eid in matched_enemy_ids or di in matched_detection_idx:
                continue
            matched_enemy_ids.add(eid)
            matched_detection_idx.add(di)
            det = detections[di]
            self.enemies[eid].update_position(
                det.x, det.y, det.width, det.height, dt, det.confidence)
            e = self.enemies[eid]
            logger.debug(f"Enemy #{eid} MOVE ({e.center_x},{e.center_y})")

        # 소실 예측 / 완전 제거
        for eid in list(self.enemies.keys()):
            if eid in matched_enemy_ids:
                continue
            enemy = self.enemies[eid]
            enemy.apply_prediction()
            if enemy.missing_frames > self.max_missing_frames:
                logger.info(f"Enemy #{eid} LOST")
                del self.enemies[eid]

        # 신규 적 등록
        for di, det in enumerate(detections):
            if di in matched_detection_idx:
                continue
            eid = self._next_id
            self._next_id += 1
            self.enemies[eid] = Enemy(
                id=eid,
                x=det.x, y=det.y, width=det.width, height=det.height,
                center_x=det.center_x, center_y=det.center_y,
                confidence=det.confidence,
            )
            logger.info(f"Enemy #{eid} CREATED ({det.center_x},{det.center_y})")

        # ── 순차 타겟 상태머신 업데이트 ─────────────────────────────
        self._sm.update(self.enemies)

        return list(self.enemies.values())
