import json
import os
import sys

# paths.py 를 import (프로젝트 루트에 위치)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paths import get_config_path, get_config_save_path

# 하위 호환: DEFAULT_CONFIG_PATH 는 읽기 경로
DEFAULT_CONFIG_PATH = get_config_path()


def load_config(path=None):
    p = path or get_config_path()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, path=None):
    p = path or get_config_save_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
