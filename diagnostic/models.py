"""DiagRecord + _infer_decision — 순수 로직 모듈 (외부 의존성 없음).

capture/mss/cv2 등 Windows 전용 모듈을 임포트하지 않으므로
sandbox(Linux)에서도 pytest로 직접 검증 가능하다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# DiagRecord
# ──────────────────────────────────────────────────────────────────────────────

class DiagRecord:
    """한 tick에서 수집한 진단 데이터."""

    __slots__ = (
        "timestamp",
        "hp_pct", "level",
        "is_moving",
        "enemy_count", "tracked_count", "loot_count",
        "hunting_state", "target_id", "target_state",
        "decision", "decision_reason",
        "frame_ms",
    )

    def __init__(self):
        self.timestamp:       float         = 0.0
        self.hp_pct:          float         = 100.0
        self.level:           Optional[int] = None
        self.is_moving:       bool          = False
        self.enemy_count:     int           = 0
        self.tracked_count:   int           = 0
        self.loot_count:      int           = 0
        self.hunting_state:   str           = "N/A"
        self.target_id:       Optional[int] = None
        self.target_state:    str           = "IDLE"
        self.decision:        str           = "WAIT"
        self.decision_reason: str           = ""
        self.frame_ms:        float         = 0.0

    def to_log_lines(self) -> list[str]:
        """사람이 읽을 수 있는 로그 블록 (요구사항 포맷)."""
        ts = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]
        lines = [
            f"[{ts}]",
            "[PERCEPTION]",
            f"  hp={self.hp_pct:.1f}",
            f"  level={self.level if self.level is not None else '?'}",
            f"  moving={self.is_moving}",
            f"  enemies={self.enemy_count}",
            f"  tracked={self.tracked_count}",
            f"  loot={self.loot_count}",
            "[STATE]",
            f"  hunting={self.hunting_state}",
            f"  target={self.target_id if self.target_id is not None else '-'}",
            f"  target_state={self.target_state}",
            "[DECISION]",
            f"  {self.decision}",
            f"  reason={self.decision_reason}",
            f"  target_id={self.target_id if self.target_id is not None else '-'}",
            f"  frame_ms={self.frame_ms:.1f}",
            "",
        ]
        return lines


# ──────────────────────────────────────────────────────────────────────────────
# Decision 추론
# ──────────────────────────────────────────────────────────────────────────────

def infer_decision(rec: DiagRecord) -> tuple[str, str]:
    """Dry-Run에서 perception 결과로부터 decision을 추론한다.

    실제 HuntingStateMachine 로직을 복제하지 않는다.
    현재 perception 상태에서 "만약 지금 decision을 내린다면?" 을 단순 규칙으로만 판단.

    우선순위 (높음 → 낮음):
        1. 이동 중          → WAIT
        2. HP < 50%         → USE_POTION
        3. LOOTING 상태     → LOOT
        4. 사냥 중 + 적     → ATTACK
        5. 아데나 발견      → LOOT
        6. 적 있음          → ATTACK
        7. 사냥 중 + 적 없음 → MOVE
        8. 기타             → WAIT

    Returns:
        (decision, reason)
    """
    if rec.is_moving:
        return "WAIT", "scene_moving"

    if rec.hp_pct < 50.0:
        return "USE_POTION", f"hp={rec.hp_pct:.1f}%<50"

    if rec.hunting_state == "LOOTING":
        if rec.loot_count > 0:
            return "LOOT", f"loot_detected={rec.loot_count}"
        return "LOOT", "looting_state"

    if rec.hunting_state == "HUNTING_10" and rec.tracked_count > 0:
        return "ATTACK", "enemy_detected"

    if rec.loot_count > 0:
        return "LOOT", f"loot_detected={rec.loot_count}"

    if rec.tracked_count > 0:
        return "ATTACK", "enemy_detected"

    if rec.hunting_state == "HUNTING_10":
        return "MOVE", "hunting_no_enemy"

    return "WAIT", f"state={rec.hunting_state}"
