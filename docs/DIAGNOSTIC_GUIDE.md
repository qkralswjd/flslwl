# Diagnostic Dry-Run 실제 게임 검증 절차

> 이 문서는 `run_diagnostic.py`를 이용한 실제 게임 화면 기반 통합 테스트 절차를 기술한다.

---

## 0. 전제 조건

| 항목 | 요구사항 |
|------|----------|
| OS | Windows 10/11 (게임 실행 환경) |
| Python | 3.11+ |
| 게임 | 리니지 클래식 실행 중, 포커스 활성 상태 |
| Pico | 연결 불필요 (Dry-Run은 NullPicoWorker 사용) |
| 의존성 | `pip install opencv-python mss easyocr` |

---

## 1. 실행 방법

```bash
# 기본 실행 (레벨링 모드, 1초 간격 로그)
python run_diagnostic.py

# 던전 모드 (SM 없이 perception만)
python run_diagnostic.py --mode dungeon

# 0.5초 간격 + screenshot 저장
python run_diagnostic.py --interval 0.5 --screenshot

# DEBUG 레벨 (NullPico BLOCKED 메시지 포함)
python run_diagnostic.py --verbose
```

**종료**: CV2 창에서 `q` 키 또는 창 닫기

**로그 파일**: `logs/diagnostic_YYYYMMDD_HHMMSS.log` 에 자동 저장

---

## 2. 로그 출력 형식

```text
[HH:MM:SS.mmm]
[PERCEPTION]
  hp=87.4
  level=4
  moving=False
  enemies=2
  tracked=2
  loot=0
[STATE]
  hunting=HUNTING_10
  target=3
  target_state=LOCKING
[DECISION]
  ATTACK
  reason=enemy_detected
  target_id=3
  frame_ms=14.3
```

---

## 3. CV2 오버레이 구성

| 요소 | 설명 |
|------|------|
| 녹색 박스 | 일반 감지 적 (tracker ID 표시) |
| 빨간 박스 | 현재 공격 타겟 |
| 좌상단 패널 | PERCEPTION / STATE / DECISION 요약 |
| 우상단 | `[DRY-RUN / NO PICO]` 워터마크 |
| 하단 | frame 처리 시간 ms |
| 화면 상단 | `MOVING — DETECTION PAUSED` (이동 중 표시) |

---

## 4. TEST A — HP / Level

**목적**: HpReader, LevelReader가 실제 게임 UI에서 올바른 값을 읽는지 확인

**준비**:
1. `config_automation.json`의 `hp_bar.region`, `level_ocr.region`을 `coordinate_picker.py`로 정확히 지정
2. 게임 실행 후 `python run_diagnostic.py` 실행

**검증 절차**:
```
1. 게임 화면의 실제 HP 바 상태 확인
   → 로그에서 hp=실제값 과 일치하는지 비교

2. HP 포션 마셔서 HP를 의도적으로 변화시킴
   → 약 0.5초 이내에 hp 값이 갱신되는지 확인

3. 레벨 표시 영역 확인
   → level=N 과 실제 레벨 일치 여부
```

**합격 기준**:
- `hp=` 값이 실제 HP 바 시각적 상태와 ±5% 이내
- `level=` 값이 실제 레벨과 일치

**실패 시 조치**:
- `config_automation.json`의 region 좌표 재조정
- HP 바가 빨간색 계열인지 확인 (`_HP_HSV_RANGES` 범위 조정 필요 가능)

---

## 5. TEST B — Monster 감지

**목적**: ContourDetector + NearestNeighborTracker가 실제 몬스터를 올바르게 감지하는지 확인

**준비**:
- 게임에서 몬스터가 보이는 화면으로 이동

**검증 절차**:
```
1. 몬스터가 1~2마리 보이는 위치에서 진단 실행
   → CV2 창에서 몬스터 위에 녹색 박스 표시 확인
   → enemies=N 이 실제 몬스터 수와 일치하는지 확인

2. 타겟이 설정됐을 때
   → 빨간 박스가 가장 가까운 몬스터에 표시되는지 확인
   → target=ID 값이 박스 ID와 일치하는지 확인

3. 몬스터가 이동할 때
   → 박스가 몬스터를 따라가는지 (Tracker 동작 확인)
   → ID가 유지되는지 확인
```

**합격 기준**:
- `enemies=N` 이 화면의 실제 몬스터 수와 일치
- bbox가 몬스터 외형과 대략 일치
- 이동 중에도 ID가 유지됨 (깜빡임 없음)

---

## 6. TEST C — Motion 감지

**목적**: SceneMotionFilter가 캐릭터 이동 중/정지를 올바르게 분류하는지 확인

**검증 절차**:
```
1. 캐릭터 이동 시작
   → 로그에서 moving=True 전환 확인
   → CV2 창에 "MOVING — DETECTION PAUSED" 배너 표시 확인

2. 캐릭터 이동 정지
   → settle_frames (기본 5프레임) 이내에 moving=False 전환 확인
   → enemies 감지 재개 확인
```

**합격 기준**:
- 이동 중 `moving=True`, 정지 후 `moving=False` 전환이 시각적으로 확인됨
- 이동 중 `enemies=0` (감지 일시정지)

---

## 7. TEST D — Loot 감지

**목적**: LootDetector가 아데나 텍스트를 올바르게 탐지하는지 확인

**준비**:
- 게임에서 몬스터를 잡아 아데나가 바닥에 떨어진 상태

**검증 절차**:
```
1. 아데나가 바닥에 있는 화면에서 진단 실행
   → 로그에서 loot=1 (또는 그 이상) 확인

2. 아데나를 줍거나 사라지면
   → loot=0 으로 복귀 확인
```

**합격 기준**:
- 아데나 텍스트가 화면에 있을 때 `loot>0`
- 텍스트가 없을 때 `loot=0`

**실패 시 조치**:
- `config_automation.json`의 `loot.scan_region` 조정
- easyocr 한국어 모델 로드 확인

---

## 8. TEST E — State 전환

**목적**: HuntingStateMachine 상태 전환이 실제 게임 화면과 일치하는지 확인

> **주의**: TEST E는 HuntingStateMachine을 실제로 `start()` 한 상태에서만 확인 가능하다.
> Dry-Run에서는 SM이 IDLE에서 정지하므로, 아래 검증은 **SM을 실제로 시작한 경우** 해당된다.

**SM을 start()하려면** `diagnostic_runner.py`의 아래 라인을 변경:
```python
# 기본 (관찰 모드): start() 미호출 — IDLE 유지
# hunting_sm.start()  ← 주석 해제 시 실제 SM 동작 (Pico는 여전히 NullPico)
```

**검증 절차**:
```
상태 전환 순서:
IDLE → USE_SCROLL_DUMMY → MOVE_TO_DUMMY → ATTACKING_DUMMY
     → USE_SPEED_POTION → MOVE_TO_HUNT_ZONE → HUNTING_10

각 상태에서:
1. CV2 패널의 hunting=STATE 값 확인
2. 로그의 [STATE] hunting=STATE 확인
3. 실제 게임 화면 상태와 일치하는지 비교
```

**합격 기준**:
- 각 상태가 실제 게임 동작과 동시에 표시됨
- 상태 전환 후 CV2 색상이 변경됨 (`_SM_COLOR_MAP` 참조)

---

## 9. TEST F — Dry-Run Decision

**목적**: `_infer_decision()` 추론이 실제 화면 조건과 일치하는지 확인

> Decision은 Pico 명령을 실행하지 않는다. 로그/패널에만 표시된다.

**검증 절차**:

| 게임 화면 조건 | 기대 decision | 확인 방법 |
|---------------|---------------|-----------|
| 적 화면에 있음 | `ATTACK reason=enemy_detected` | 로그/패널 |
| 적 없음 + 사냥 중 | `MOVE reason=hunting_no_enemy` | 로그/패널 |
| 아데나 바닥에 있음 | `LOOT reason=loot_detected=N` | 로그/패널 |
| HP < 50% | `USE_POTION reason=hp=N%<50` | 로그/패널 |
| 이동 중 | `WAIT reason=scene_moving` | 로그/패널 |

**합격 기준**:
- 각 조건에서 decision이 위 표와 일치

---

## 10. Pico 명령 차단 확인 방법

```bash
# --verbose 옵션으로 실행하면 모든 차단 이벤트가 DEBUG 레벨로 출력됨
python run_diagnostic.py --verbose

# 로그에 다음과 같은 메시지가 나타나면 차단 동작 중
# [null_pico] BLOCKED click(960, 540, pulse_ms=20)
# [null_pico] BLOCKED key_tap_name('F5', hold_ms=80)
```

**차단 확인 기준**:
- 게임 화면에서 실제 마우스 커서가 움직이지 않음
- 키 입력이 게임에 전달되지 않음 (캐릭터 무반응)
- 로그에 `BLOCKED` 메시지가 출력됨

---

## 11. 알려진 제한사항

| 항목 | 설명 |
|------|------|
| 실제 게임 실행 환경 | Windows 전용. 이 sandbox(Linux)에서는 실행 불가 |
| LevelReader | easyocr 초기화에 10~30초 소요 (최초 1회) |
| LootDetector | easyocr 한국어 모델 별도 다운로드 필요 |
| SM start() | Dry-Run에서는 기본 IDLE 유지. 실제 SM 동작 테스트는 별도 |
| COM 포트 | NullPico 사용으로 COM 포트 불필요 |

---

## 12. 파일 구조

```
flslwl/
├── run_diagnostic.py              ← 진입점
├── diagnostic/
│   ├── __init__.py
│   └── diagnostic_runner.py       ← 핵심 로직 (DiagnosticRunner)
├── pico/
│   ├── null_pico.py               ← NullPicoWorker (Pico 차단)
│   └── pico_serial.py             ← 실제 Pico (기존, 변경 없음)
├── logs/
│   ├── diagnostic_YYYYMMDD_HHMMSS.log
│   └── screens_YYYYMMDD_HHMMSS/
│       └── screen_HHMMSS_mmm.jpg
└── docs/
    └── DIAGNOSTIC_GUIDE.md        ← 이 문서
```
