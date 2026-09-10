"""
modes/ — 게임 자동화 동작 모드 모음.

서브모듈:
  leveling — LevelingMode : 허수아비 → 사냥터 1단계 자동 레벨링 (메인 흐름)
  dungeon  — DungeonMode  : 던전 반복 공략 모드
  field    — FieldMode    : 필드 자유 사냥 모드 (tracker 필드 모드)

각 모드는 core.state.BaseFSM 을 상속하고
tick(dt) 으로 1프레임씩 진행된다.
"""

__all__ = ["leveling", "dungeon", "field"]
