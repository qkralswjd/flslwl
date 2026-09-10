"""
main.py — 리니지 자동화 봇 v2 진입점.

사용법:
  python main.py               # GUI 실행 (기본)
  python main.py --headless leveling   # GUI 없이 레벨링 모드 실행
  python main.py --headless dungeon
  python main.py --headless field
  python main.py --config custom.json  # 설정 파일 지정
  python main.py --version
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

log = logging.getLogger("main")

# ─────────────────────────── 로깅 초기화 ────────────────────────

def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 외부 라이브러리 로거 조용히
    for name in ("easyocr", "PIL", "urllib3", "serial"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ─────────────────────────── 설정 로드 ──────────────────────────

def _load_settings(config_path: str | None = None):
    from config.settings import Settings
    path = config_path or ""
    try:
        s = Settings.load(path) if path else Settings.load()
        log.info("설정 로드 완료: %s", path or "(기본)")
        return s
    except Exception as e:
        log.error("설정 로드 실패: %s", e)
        log.warning("기본 Settings() 를 사용합니다")
        return Settings()


# ─────────────────────────── 하드웨어 초기화 ────────────────────

def _init_hardware(settings):
    """Pico + ScreenCapturer 초기화. 실패 시 Null/None 반환."""
    # Pico
    try:
        from hardware.pico import PicoWorker
        pico = PicoWorker.from_settings(settings)
        pico.start()
        log.info("Pico 연결 성공: %s", settings.pico.serial_port)
    except Exception as e:
        from hardware.pico import NullPicoWorker
        pico = NullPicoWorker.from_settings(settings)
        log.warning("Pico 연결 실패(%s) — NullPicoWorker 사용", e)

    # ScreenCapturer
    try:
        from hardware.screen import ScreenCapturer
        capturer = ScreenCapturer.from_settings(settings)
        log.info("ScreenCapturer 초기화 완료")
    except Exception as e:
        capturer = None
        log.warning("ScreenCapturer 초기화 실패(%s) — 프레임 없음", e)

    return pico, capturer


# ─────────────────────────── Perception 초기화 ──────────────────

def _init_perception(settings):
    """HpReader / LevelReader / LootDetector 생성."""
    hp_reader     = None
    level_reader  = None
    loot_detector = None

    try:
        from perception.hp import HpReader
        hp_reader = HpReader.from_settings(settings)
    except Exception as e:
        log.warning("HpReader 초기화 실패: %s", e)

    try:
        from perception.level import LevelReader
        level_reader = LevelReader.from_settings(settings)
    except Exception as e:
        log.warning("LevelReader 초기화 실패: %s", e)

    try:
        from perception.loot import LootDetector
        loot_detector = LootDetector.from_settings(settings)
    except Exception as e:
        log.warning("LootDetector 초기화 실패: %s", e)

    return hp_reader, level_reader, loot_detector


# ─────────────────────────── Tracker 초기화 ─────────────────────

def _init_tracker(settings, pico):
    try:
        from core.tracking import NearestNeighborTracker
        tracker = NearestNeighborTracker.from_settings(
            settings,
            pico_click_callback=lambda x, y: pico.click(x, y),
            pico_drag_callback=lambda fx, fy, tx, ty: pico.drag(fx, fy, tx, ty),
        )
        log.info("NearestNeighborTracker 초기화 완료")
        return tracker
    except Exception as e:
        log.warning("Tracker 초기화 실패: %s", e)
        return None


# ─────────────────────────── 헤드리스 실행 ──────────────────────

def _run_headless(mode_name: str, settings, pico, capturer):
    """GUI 없이 지정된 모드를 실행한다."""
    import numpy as np

    grab = (capturer.grab if capturer else
            lambda: np.zeros((100, 200, 3), dtype=np.uint8))

    hp_reader, level_reader, loot_detector = _init_perception(settings)
    tracker = _init_tracker(settings, pico)

    if mode_name == "leveling":
        from modes.leveling import LevelingMode
        mode = LevelingMode(
            settings=settings, pico=pico, frame_grabber=grab,
            tracker=tracker, hp_reader=hp_reader,
            level_reader=level_reader, loot_detector=loot_detector,
        )
    elif mode_name == "dungeon":
        from modes.dungeon import DungeonMode
        mode = DungeonMode(
            settings=settings, pico=pico, frame_grabber=grab,
            tracker=tracker, hp_reader=hp_reader, loot_detector=loot_detector,
        )
    elif mode_name == "field":
        from modes.field import FieldMode
        mode = FieldMode(
            settings=settings, pico=pico, frame_grabber=grab,
            tracker=tracker, hp_reader=hp_reader, loot_detector=loot_detector,
        )
    else:
        log.error("알 수 없는 모드: %s", mode_name)
        return

    log.info("▶ %s 모드 시작 (헤드리스)", mode_name)
    mode.start()

    last_t = time.monotonic()
    try:
        while not mode.is_done:
            now = time.monotonic()
            dt  = now - last_t
            last_t = now
            mode.run_tick(dt)
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        log.info("사용자 중단 (Ctrl+C)")
    finally:
        mode.stop()
        log.info("■ %s 정지", mode_name)


# ─────────────────────────── 진입점 ─────────────────────────────

def main(argv: list | None = None):
    parser = argparse.ArgumentParser(
        prog="bot2",
        description="리니지 자동화 봇 v2",
    )
    parser.add_argument("--config",    metavar="PATH", help="설정 JSON 파일 경로")
    parser.add_argument("--headless",  metavar="MODE",
                        choices=["leveling", "dungeon", "field"],
                        help="GUI 없이 지정 모드 실행")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG 로깅")
    parser.add_argument("--version",       action="version",    version="bot2 v2.0")
    args = parser.parse_args(argv)

    _setup_logging(verbose=args.verbose)

    settings = _load_settings(args.config)

    if args.headless:
        pico, capturer = _init_hardware(settings)
        try:
            _run_headless(args.headless, settings, pico, capturer)
        finally:
            try:
                pico.stop()
            except Exception:
                pass
            if capturer:
                try:
                    capturer.close()
                except Exception:
                    pass
    else:
        # GUI 실행
        try:
            from ui.app import BotApp
            app = BotApp(settings=settings)
            app.root.protocol("WM_DELETE_WINDOW", app.close)
            log.info("GUI 시작")
            app.run()
        except ImportError as e:
            log.error("GUI 실행 실패 (tkinter 없음?): %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
