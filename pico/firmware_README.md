# Pico Firmware — Quick Reference

전체 설치/빌드 가이드는 저장소 루트의 `README.md` 5번, 8번 문단을 참고하세요.
여기서는 요약만 정리합니다.

## 파일

- `firmware/boot.py` — 부팅 시 1회 실행. `usb_cdc.data` 채널을 활성화합니다.
  **파일을 복사한 후 반드시 하드 리셋(재연결)해야 적용됩니다.**
- `firmware/code.py` — 메인 루프. `usb_cdc.data`로 들어오는 텍스트 명령을 읽어
  `adafruit_hid.mouse.Mouse`로 HID 마우스 이벤트를 발생시킵니다.

## 설치 순서 (요약)

1. CircuitPython UF2를 Pico에 설치 (BOOTSEL 누른 채 연결 → UF2 복사)
2. Adafruit CircuitPython Bundle에서 `adafruit_hid` 폴더를 `CIRCUITPY/lib/`에 복사
3. `boot.py`, `code.py`를 `CIRCUITPY/` 루트로 복사
4. USB 재연결(하드 리셋)
5. 장치관리자에서 COM 포트 2개(REPL + data) 확인

## 프로토콜

```
PING            -> PONG
MOVE:<dx>:<dy>  -> OK:MOVE | ERR:MOVE   (상대 이동, HID 리포트 한도 +-127을 넘으면 자동 분할)
CLICK[:<ms>]    -> OK:CLICK | ERR:CLICK (좌클릭 pulse, ms 생략 시 기본 20ms)
STOP            -> OK:STOP
그 외           -> ERR:UNKNOWN
```

## 지원 보드

CircuitPython을 지원하는 모든 Pico 계열 (Pico, Pico W, Pico 2 등).
