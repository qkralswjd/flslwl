"""Diagnostic Dry-Run 진입점.

실제 게임 화면을 캡처해 perception / decision 결과를 관찰한다.
Pico 명령(click/drag/key_tap)은 완전히 차단된다.

사용법
──────
    # 레벨링 모드 (HuntingStateMachine 연결)
    python run_diagnostic.py

    # 던전 모드 (SM 없이 perception만)
    python run_diagnostic.py --mode dungeon

    # 로그 간격 0.5초
    python run_diagnostic.py --interval 0.5

    # debug screenshot도 저장
    python run_diagnostic.py --screenshot

    # 모든 옵션
    python run_diagnostic.py --mode leveling --interval 1.0 --screenshot --log-dir logs

종료
────
    CV2 창에서 'q' 키 입력, 또는 창을 닫으면 종료된다.

출력 파일
─────────
    logs/diagnostic_YYYYMMDD_HHMMSS.log
    logs/screens_YYYYMMDD_HHMMSS/screen_HHMMSS_mmm.jpg  (--screenshot 옵션 시)
"""

import argparse
import json
import logging
import os
import sys

# 프로젝트 루트를 PYTHONPATH에 추가
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnostic Dry-Run — 실제 게임 화면 기반 perception 관찰기 (Pico 명령 차단)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["leveling", "dungeon"],
        default="leveling",
        help="leveling: HuntingStateMachine 연결 / dungeon: perception만 (기본: leveling)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="콘솔/파일 로그 출력 간격 초 (기본: 1.0)",
    )
    p.add_argument(
        "--screenshot",
        action="store_true",
        help="지정 시 1초마다 debug screenshot을 저장한다",
    )
    p.add_argument(
        "--log-dir",
        default="logs",
        metavar="DIR",
        help="로그/screenshot 저장 디렉토리 (기본: logs/)",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="config.json 경로 (기본: config/config.json)",
    )
    p.add_argument(
        "--automation-config",
        default=None,
        metavar="PATH",
        help="config_automation.json 경로 (기본: config/config_automation.json)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG 레벨 로그 출력 (NullPico BLOCKED 메시지 포함)",
    )
    return p.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_diagnostic")

    # ── config 로드 ───────────────────────────────────────────────────────────
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(_ROOT, "config", "config.json")

    if not os.path.exists(config_path):
        logger.error(f"config.json을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    config = _load_json(config_path)
    logger.info(f"config 로드: {config_path}")

    # ── automation_config 로드 (leveling 모드일 때만) ─────────────────────────
    automation_config = None
    if args.mode == "leveling":
        if args.automation_config:
            auto_path = args.automation_config
        else:
            auto_path = os.path.join(_ROOT, "config", "config_automation.json")

        if os.path.exists(auto_path):
            automation_config = _load_json(auto_path)
            logger.info(f"automation_config 로드: {auto_path}")
        else:
            logger.warning(
                f"config_automation.json 없음: {auto_path}\n"
                "  → HuntingStateMachine 비활성. perception만 동작합니다."
            )

    # ── 실행 환경 안내 ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Diagnostic Dry-Run 시작")
    print(f"  모드     : {args.mode}")
    print(f"  로그 간격: {args.interval}s")
    print(f"  Screenshot: {'ON' if args.screenshot else 'OFF'}")
    print(f"  로그 저장 : {os.path.abspath(args.log_dir)}/")
    print("  Pico 명령 : 전부 차단 (NullPicoWorker)")
    print("  종료      : CV2 창에서 'q' 키")
    print("=" * 60)

    # ── DiagnosticRunner 실행 ─────────────────────────────────────────────────
    from diagnostic.diagnostic_runner import DiagnosticRunner

    runner = DiagnosticRunner(
        config            = config,
        automation_config = automation_config,
        mode              = args.mode,
        log_interval_s    = args.interval,
        save_screenshots  = args.screenshot,
        log_dir           = args.log_dir,
    )

    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — 종료")
    except Exception as e:
        logger.exception(f"예기치 않은 오류: {e}")
        sys.exit(1)

    logger.info("Diagnostic 종료.")


if __name__ == "__main__":
    main()
