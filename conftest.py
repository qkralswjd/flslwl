"""
conftest.py — pytest 전역 설정.

문제:
  test_hardware.py 의 TestScreenCapturer 가 cv2 를 stub 으로 patch.dict 교체.
  tearDown 에서 복원하지만 perception.hp / perception.loot 내부가 이미 stub cv2 를
  참조하는 경우 이후 테스트가 AttributeError 로 실패한다.

해결:
  각 테스트 실행 전에 sys.modules 에서 perception.* 를 제거해 강제 재 import.
  또한 실제 cv2 객체를 저장해두고 stub 로 교체된 경우 복원한다.
"""
import sys
import importlib

# 실제 cv2 모듈을 최초 한 번 저장
_real_cv2 = None


def pytest_configure(config):
    global _real_cv2
    try:
        _real_cv2 = importlib.import_module("cv2")
    except ImportError:
        pass


def pytest_runtest_setup(item):
    """각 테스트 실행 전 처리."""
    # 1. cv2 가 stub 으로 교체되었으면 실제 cv2 로 복원
    if _real_cv2 is not None:
        current_cv2 = sys.modules.get("cv2")
        if current_cv2 is not _real_cv2:
            sys.modules["cv2"] = _real_cv2

    # 2. perception 모듈 캐시 제거 → 재 import 시 올바른 cv2 를 가져감
    to_remove = [k for k in list(sys.modules) if k.startswith("perception")]
    for k in to_remove:
        del sys.modules[k]
