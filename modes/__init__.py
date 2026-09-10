"""
modes/ — 게임 자동화 동작 모드 모음.

서브모듈:
  leveling  — LevelingMode : 허수아비 → 사냥터 1단계 자동 레벨링
  dungeon   — DungeonMode  : 던전 반복 공략 모드
  field     — FieldMode    : 필드 자유 사냥 모드
  hunt_loop — HuntLoop     : 탐지→추적→공격→루팅 무한 사냥 엔진 (핵심)
              build_hunt_loop() 팩토리 함수로 생성

각 모드는 core.state.BaseFSM 을 상속하고
tick(dt) 으로 1프레임씩 진행된다.

HuntLoop 는 FSM 과 독립적인 백그라운드 스레드로 실행되며
탐지(RealtimeTemplateDetector) → 추적(NearestNeighborTracker)
→ 공격(SequentialTargetSM) → 루팅(LootDetector) 파이프라인을 담당한다.
"""

__all__ = ["leveling", "dungeon", "field", "hunt_loop"]
