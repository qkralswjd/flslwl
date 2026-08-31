"""자동화 모듈 패키지.

요정 1단계 자동 레벨링 시스템:
────────────────────────────────────────────────────────────
  IDLE
   → USE_SCROLL_DUMMY  : F6 말하는 두루마리 → 허수아비 수련장 텔레포트
   → MOVE_TO_DUMMY     : 허수아비 좌표로 이동
   → ATTACKING_DUMMY   : Lv.target_level_dummy(5)까지 허수아비 공격
                          HP < 50% → F5 물약 자동 사용
   → USE_SPEED_POTION  : F9 속도향상물약 사용
   → MOVE_TO_HUNT_ZONE : 사냥터 웨이포인트로 이동
   → HUNTING_10        : Lv.target_level_hunt(10)까지 사냥
                          HP < 50% → F5 물약 자동 사용
                          아데나 발견 → LOOTING → 복귀
   → DONE_PHASE1       : 1단계 완료
────────────────────────────────────────────────────────────
"""
from automation.state_machine import HuntingStateMachine, HuntingState
from automation.hp_reader     import HpReader
from automation.level_reader  import LevelReader

__all__ = [
    "HuntingStateMachine",
    "HuntingState",
    "HpReader",
    "LevelReader",
]
