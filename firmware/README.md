# firmware/ — Raspberry Pi Pico CircuitPython 펌웨어

## 파일 설명

| 파일 | 설명 |
|------|------|
| `boot.py` | USB CDC(시리얼) + HID(마우스/키보드) 동시 활성화 |
| `code.py`  | 메인 펌웨어 — PC 에서 오는 명령을 HID 동작으로 변환 |

## 설치 방법

1. Raspberry Pi Pico 에 [CircuitPython](https://circuitpython.org/board/raspberry_pi_pico/) 설치
2. 필요 라이브러리를 `lib/` 에 복사:
   - `adafruit_hid` (adafruit-circuitpython-hid)
   - `usb_cdc`
3. `boot.py` 와 `code.py` 를 Pico 루트(`/`) 에 복사
4. Pico 재부팅 → COM 포트로 인식됨

## 통신 프로토콜

PC → Pico (텍스트, `\n` 종료):

| 명령 | 설명 |
|------|------|
| `CLICK x y ms\n`          | (x, y) 상대 이동 후 ms 동안 클릭 |
| `CLICK_CURRENT ms\n`      | 현재 위치에서 ms 동안 클릭 |
| `DRAG fx fy tx ty steps\n`| (fx,fy) → (tx,ty) 드래그 |
| `KEY code ms\n`           | HID 키코드 code 를 ms 동안 누름 |
| `STOP\n`                  | 진행 중인 동작 중단 |

Pico → PC: `OK\n` 또는 `ERR msg\n`
