"""자동화 모듈 패키지.

1단계 자동 레벨링 시스템:
    - 텔레포트 → 허수아비 공격 → 5레벨
    - 좌표 기반 순환 이동 → 몬스터 탐지/공격 → 아데나 줍기
    - 15레벨까지 무한 반복
"""
from automation.state_machine import HuntingStateMachine, HuntingState

__all__ = ["HuntingStateMachine", "HuntingState"]
