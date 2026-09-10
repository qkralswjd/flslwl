"""
tools/diagnostic.py — 시스템 진단 도구.

사용법:
  python -m tools.diagnostic
  python tools/diagnostic.py --config custom.json --full

기능:
  1. 필수 패키지 설치 여부 확인
  2. 설정 파일 로드 및 유효성 검사
  3. Pico 시리얼 포트 목록 및 연결 테스트
  4. 화면 캡처 테스트 (프레임 1장 저장)
  5. 탐지 엔진 템플릿 로드 확인
  6. perception 모듈 초기화 확인
  7. 전체 결과 요약 출력
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable


# ─────────────────────────── 결과 추적 ──────────────────────────

class _Result:
    def __init__(self):
        self.passed  : list[str] = []
        self.failed  : list[str] = []
        self.warnings: list[str] = []

    def ok(self,   msg: str): self.passed.append(f"  ✅ {msg}")
    def fail(self, msg: str): self.failed.append(f"  ❌ {msg}")
    def warn(self, msg: str): self.warnings.append(f"  ⚠  {msg}")

    def print_summary(self):
        print()
        print("═" * 60)
        print("  진단 결과")
        print("═" * 60)
        for m in self.passed   : print(m)
        for m in self.warnings : print(m)
        for m in self.failed   : print(m)
        print("─" * 60)
        total = len(self.passed) + len(self.failed) + len(self.warnings)
        print(f"  합계: {total}개 검사  |  "
              f"통과 {len(self.passed)}  경고 {len(self.warnings)}  실패 {len(self.failed)}")
        if self.failed:
            print("  → 실패 항목을 수정 후 다시 실행하세요.")
        else:
            print("  → 모든 핵심 검사 통과!")
        print("═" * 60)


R = _Result()


# ─────────────────────────── 검사 함수 ──────────────────────────

def check_packages():
    print("\n[1] 패키지 확인")
    required = {
        "cv2":     "opencv-python",
        "numpy":   "numpy",
        "mss":     "mss",
        "serial":  "pyserial",
    }
    optional = {
        "easyocr": "easyocr",
    }
    for mod, pkg in required.items():
        try:
            __import__(mod)
            R.ok(f"{mod} ({pkg})")
        except ImportError:
            R.fail(f"{mod} 없음 → pip install {pkg}")

    for mod, pkg in optional.items():
        try:
            __import__(mod)
            R.ok(f"{mod} ({pkg}) [선택]")
        except ImportError:
            R.warn(f"{mod} 없음 → pip install {pkg}  (LevelReader 비활성)")


def check_settings(config_path: str = ""):
    print("\n[2] 설정 파일")
    try:
        from config.settings import Settings, DEFAULT_CONFIG_PATH
        path = config_path or DEFAULT_CONFIG_PATH
        if Path(path).exists():
            s = Settings.load(path)
            R.ok(f"설정 로드 성공: {path}")
        else:
            s = Settings()
            R.warn(f"설정 파일 없음({path}) — 기본값 사용")
        # 간단 유효성
        assert s.capture.fps > 0,       "capture.fps <= 0"
        assert s.detection.fps > 0,     "detection.fps <= 0"
        R.ok("설정 유효성 기본 통과")
        return s
    except Exception as e:
        R.fail(f"설정 오류: {e}")
        return None


def check_serial(settings):
    print("\n[3] Pico 시리얼")
    try:
        from hardware.pico import list_serial_ports
        ports = list_serial_ports()
        if ports:
            R.ok(f"시리얼 포트 목록: {ports}")
        else:
            R.warn("사용 가능한 시리얼 포트 없음")
    except Exception as e:
        R.fail(f"포트 목록 오류: {e}")

    if settings is None:
        return

    configured_port = getattr(settings.pico, "serial_port", "")
    if not configured_port:
        R.warn("pico.serial_port 설정 없음")
        return

    try:
        from hardware.pico import PicoWorker
        worker = PicoWorker.from_settings(settings)
        worker.start()
        time.sleep(0.3)
        connected = worker.is_connected
        worker.stop()
        if connected:
            R.ok(f"Pico 연결 성공: {configured_port}")
        else:
            R.warn(f"Pico 응답 없음: {configured_port}")
    except Exception as e:
        R.warn(f"Pico 연결 실패({configured_port}): {e}")


def check_screen(settings, save_frame: bool = False):
    print("\n[4] 화면 캡처")
    try:
        from hardware.screen import ScreenCapturer
        cap = ScreenCapturer.from_settings(settings) if settings else ScreenCapturer()
        frame = cap.grab()
        cap.close()
        h, w = frame.shape[:2]
        R.ok(f"캡처 성공: {w}×{h} px")
        if save_frame:
            import cv2
            fname = "diagnostic_frame.png"
            cv2.imwrite(fname, frame)
            R.ok(f"프레임 저장: {fname}")
    except Exception as e:
        R.fail(f"캡처 실패: {e}")


def check_templates(settings):
    print("\n[5] 탐지 템플릿")
    if settings is None:
        R.warn("설정 없음 — 템플릿 확인 스킵")
        return
    try:
        templates_dir = getattr(settings.detection, "templates_dir", "")
        if not templates_dir:
            R.warn("detection.templates_dir 설정 없음")
            return
        p = Path(templates_dir)
        if not p.exists():
            R.warn(f"템플릿 디렉토리 없음: {p}")
            return
        pngs = list(p.glob("*.png"))
        if pngs:
            R.ok(f"템플릿 {len(pngs)}개 발견: {[f.name for f in pngs[:5]]}")
        else:
            R.warn(f"템플릿 PNG 없음: {p}")
    except Exception as e:
        R.fail(f"템플릿 확인 오류: {e}")

    # RealtimeTemplateDetector 로드
    if settings is None:
        return
    try:
        from core.detection import RealtimeTemplateDetector
        det = RealtimeTemplateDetector.from_settings(settings)
        if det.has_templates():
            R.ok(f"RealtimeTemplateDetector 초기화 완료 (templates={det.has_templates()})")
        else:
            R.warn("RealtimeTemplateDetector 템플릿 0개 로드됨")
    except Exception as e:
        R.fail(f"RealtimeTemplateDetector 오류: {e}")


def check_perception(settings):
    print("\n[6] Perception 모듈")
    if settings is None:
        R.warn("설정 없음 — perception 확인 스킵")
        return

    # HpReader
    try:
        from perception.hp import HpReader
        reader = HpReader.from_settings(settings)
        import numpy as np
        frame = np.zeros((50, 200, 3), dtype=np.uint8)
        pct = reader.read(frame)
        R.ok(f"HpReader 정상 (검정 프레임 → {pct}%)")
    except Exception as e:
        R.fail(f"HpReader 오류: {e}")

    # LevelReader
    try:
        from perception.level import LevelReader
        reader = LevelReader.from_settings(settings)
        R.ok(f"LevelReader 초기화 (ready={reader.is_ready})")
    except Exception as e:
        R.fail(f"LevelReader 오류: {e}")

    # LootDetector
    try:
        from perception.loot import LootDetector
        detector = LootDetector.from_settings(settings)
        import numpy as np
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        items = detector.find(frame)
        R.ok(f"LootDetector 정상 (검정 프레임 → {len(items)}개)")
    except Exception as e:
        R.fail(f"LootDetector 오류: {e}")


def check_modes(settings):
    print("\n[7] 모드 임포트")
    for mod in ["modes.leveling", "modes.dungeon", "modes.field"]:
        try:
            __import__(mod)
            R.ok(f"{mod} 임포트 OK")
        except Exception as e:
            R.fail(f"{mod} 임포트 실패: {e}")


# ─────────────────────────── 진입점 ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="bot2 시스템 진단")
    parser.add_argument("--config", "-c", default="", help="설정 JSON 경로")
    parser.add_argument("--full",   action="store_true",
                        help="화면 캡처 프레임 저장 포함")
    args = parser.parse_args()

    print("=" * 60)
    print("  bot2 시스템 진단 도구")
    print("=" * 60)

    check_packages()
    settings = check_settings(args.config)
    check_serial(settings)
    check_screen(settings, save_frame=args.full)
    check_templates(settings)
    check_perception(settings)
    check_modes(settings)

    R.print_summary()

    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
