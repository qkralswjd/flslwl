"""hardware — 외부 장치 추상화 레이어.

하위 모듈:
    screen  : mss 기반 화면 캡처 (ScreenCapturer)
    pico    : Pico HID 워커 (PicoWorker, NullPicoWorker, KEY_NAME_MAP)

Note:
    __init__.py 는 서브모듈을 즉시 가져오지 않는다.
    mss / cv2 / pyserial 같은 런타임 의존성이 테스트 환경에 없을 수 있으므로
    각 서브모듈은 사용 측에서 직접 import 한다.

        from hardware.screen import ScreenCapturer
        from hardware.pico  import PicoWorker, NullPicoWorker
"""

__all__ = [
    "ScreenCapturer",
    "PicoWorker",
    "NullPicoWorker",
    "KEY_NAME_MAP",
    "list_serial_ports",
]
