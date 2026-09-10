# bot2 — 리니지 클래식 자동화 봇 v2

## 프로젝트 개요

기존 `flslwl/` 프로젝트를 **전면 재설계**한 신규 아키텍처.  
계층 분리·타입 힌트·단위 테스트 중심으로 재작성되었습니다.

---

## 디렉토리 구조

```
bot2/
├── config/          # 통합 설정 (Settings dataclass, load/save)
├── core/            # 탐지·추적·상태머신 엔진
│   ├── detection.py  → RealtimeTemplateDetector (matchTemplate + NMS + reject)
│   ├── tracking.py   → NearestNeighborTracker + SequentialTargetStateMachine
│   └── state.py      → BaseFSM (on_enter/on_exit 훅, elapsed_since_entry)
├── hardware/        # 하드웨어 추상화 (lazy import)
│   ├── screen.py     → ScreenCapturer (mss, from_settings, context manager)
│   └── pico.py       → PicoWorker + NullPicoWorker (duck-typing)
├── perception/      # 화면 상태 읽기 (cv2 기반)
│   ├── hp.py         → HpReader (HSV 빨간 열비율, 캐시)
│   ├── level.py      → LevelReader (easyocr, min_confidence=0.4, 비동기)
│   └── loot.py       → LootDetector (HSV 흰 테두리 + dilation)
├── modes/           # 자동화 동작 모드 (BaseFSM 상속)
│   ├── leveling.py   → LevelingMode (허수아비→사냥터 1단계)
│   ├── dungeon.py    → DungeonMode (던전 반복 공략)
│   └── field.py      → FieldMode (필드 자유 사냥)
├── ui/              # tkinter GUI
│   └── app.py        → BotApp (레벨링/던전/필드 3탭 + 설정 + 로그)
├── tools/           # 개발·운영 보조 도구
│   ├── capture_monster.py  → 탬플릿 이미지 수집 (cv2 GUI)
│   ├── coordinate_picker.py → 좌표·HSV 픽커 (cv2 GUI)
│   └── diagnostic.py       → 시스템 진단 (패키지/연결/탬플릿 확인)
├── firmware/        → Pico CircuitPython 펌웨어 (boot.py, code.py)
├── tests/           → 단위 테스트 155개 (pytest)
├── conftest.py      → cv2 stub 격리 (test_hardware ↔ test_perception)
└── main.py          → 진입점 (GUI / --headless 모드)
```

---

## 기존 대비 개선 사항

| 항목 | 기존 (flslwl) | 신규 (bot2) |
|------|--------------|-------------|
| 레벨 OCR 신뢰도 | `confidence < 0.1` (사실상 무필터) | `min_confidence = 0.4` |
| 루팅 타임아웃 | 3초 | 8초 |
| HP 리더 | dict region 전용 | dict + tuple 통일 |
| Pico 더미 | null_pico.py 별도 파일 | NullPicoWorker 통합 (단일 pico.py) |
| FieldMode | 없음 | 신규 추가 |
| BaseFSM | 없음 | on_enter/on_exit 훅, elapsed_since_entry |
| 테스트 | 없음 | 155 passed |

---

## 실행 방법

### GUI 실행
```bash
python main.py
python main.py --config custom.json
```

### 헤드리스 실행
```bash
python main.py --headless leveling
python main.py --headless dungeon
python main.py --headless field --verbose
```

### 도구
```bash
# 탬플릿 이미지 수집
python -m tools.capture_monster --monitor 2 --out templates/

# 좌표 픽커
python -m tools.coordinate_picker --monitor 2

# 시스템 진단
python -m tools.diagnostic --full
```

### 테스트
```bash
python -m pytest tests/ -v
```

---

## 설정 파일

`config/settings.py` 의 `Settings.load()` 가 다음 경로를 순서대로 탐색:
1. 인자로 받은 경로
2. `config_automation.json` (기존 호환)
3. `config.json`

설정이 없으면 기본값으로 동작합니다.

---

## Pico 펌웨어

`firmware/` 참조. CircuitPython HID 기반, USB CDC 시리얼 통신.

---

## 의존 패키지

```
opencv-python   # cv2 탐지/인식
numpy           # 배열 연산
mss             # 화면 캡처
pyserial        # Pico 통신
easyocr         # 레벨 OCR (선택)
```

---

## Git 이력

| 커밋 | 내용 |
|------|------|
| `f489e63` | Step 3: core (detection + tracking + state) — 49 tests |
| `950b1ab` | Step 4: perception (hp + level + loot) — 43 tests |
| `eb5f0a4` | Step 5+8: modes + test_modes — 36 tests |
| `397fc74` | Step 6+7+9: ui + main + tools + firmware — 155 passed |
