"""
perception/ — 화면에서 게임 상태를 읽어내는 모듈 모음.

서브모듈:
  hp    — HpReader   : HP 바 비율 (HSV 빨간색)
  level — LevelReader: 레벨 OCR (easyocr)
  loot  — LootDetector: 아이템 테두리 탐지 (HSV 흰색 + dilation)
"""

# ▸ 이 __init__.py 는 서브모듈을 즉시 import 하지 않는다.
# ▸ cv2 / easyocr / mss 가 설치되지 않은 CI 환경에서도
#   `import perception` 자체는 성공해야 한다.
# ▸ 사용 측에서 필요한 클래스를 직접 import 한다:
#       from perception.hp import HpReader
#       from perception.level import LevelReader
#       from perception.loot import LootDetector

__all__ = ["hp", "level", "loot"]
