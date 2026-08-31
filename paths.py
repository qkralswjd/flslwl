"""EXE / 일반 실행 환경 모두에서 올바른 경로를 반환하는 유틸리티.

PyInstaller --onefile 로 빌드하면 sys.frozen=True 가 되고,
실행 파일 자체는 sys.executable 에 위치합니다.
내부 번들 파일(읽기전용)은 sys._MEIPASS 임시 폴더에 압축 해제됩니다.

규칙:
    - config.json 읽기     → _MEIPASS/config/config.json  (번들 기본값)
    - config.json 쓰기     → EXE 옆 config/config.json    (사용자 설정 저장)
    - templates 읽기       → EXE 옆 config/templates 우선,
                             없으면 _MEIPASS/config/templates (번들 초기값)
    - templates 쓰기(저장) → EXE 옆 config/templates
"""

import os
import sys


def _exe_dir() -> str:
    """EXE 파일(또는 스크립트)이 있는 폴더."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # 일반 실행: 이 파일(paths.py)의 부모 = 프로젝트 루트
    return os.path.dirname(os.path.abspath(__file__))


def _bundle_dir() -> str:
    """PyInstaller 번들 내부 임시 폴더 (없으면 _exe_dir 반환)."""
    return getattr(sys, "_MEIPASS", _exe_dir())


def get_config_path() -> str:
    """config.json 경로 — EXE 옆 우선, 없으면 번들 내부."""
    user_path = os.path.join(_exe_dir(), "config", "config.json")
    if os.path.exists(user_path):
        return user_path
    return os.path.join(_bundle_dir(), "config", "config.json")


def get_config_save_path() -> str:
    """config.json 저장 경로 — 항상 EXE 옆."""
    path = os.path.join(_exe_dir(), "config", "config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_templates_dir() -> str:
    """몬스터 템플릿 폴더 — EXE 옆 우선, 없으면 번들 내부."""
    user_path = os.path.join(_exe_dir(), "config", "templates")
    if os.path.isdir(user_path):
        return user_path
    bundle_path = os.path.join(_bundle_dir(), "config", "templates")
    if os.path.isdir(bundle_path):
        return bundle_path
    # 둘 다 없으면 EXE 옆 경로 반환 (첫 저장 시 생성됨)
    return user_path


def get_reject_templates_dir() -> str:
    """거부 템플릿 폴더 — EXE 옆 우선, 없으면 번들 내부."""
    user_path = os.path.join(_exe_dir(), "config", "templates_reject")
    if os.path.isdir(user_path):
        return user_path
    bundle_path = os.path.join(_bundle_dir(), "config", "templates_reject")
    if os.path.isdir(bundle_path):
        return bundle_path
    return user_path


def get_templates_save_dir() -> str:
    """템플릿 저장 폴더 — 항상 EXE 옆."""
    path = os.path.join(_exe_dir(), "config", "templates")
    os.makedirs(path, exist_ok=True)
    return path


def get_reject_templates_save_dir() -> str:
    """거부 템플릿 저장 폴더 — 항상 EXE 옆."""
    path = os.path.join(_exe_dir(), "config", "templates_reject")
    os.makedirs(path, exist_ok=True)
    return path
