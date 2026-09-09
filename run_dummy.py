"""허수아비/필드 자동사냥 실행 스크립트.

main.py의 run() 파이프라인을 그대로 사용합니다.
(SceneMotionFilter + MOG2 + SVM + NearestNeighborTracker + SequentialTargetSM)

실행:
    python run_dummy.py              # 기본: 허수아비 공격부터 시작 (leveling 모드)
    python run_dummy.py --field      # 필드 이동 사냥 (field 모드, WaypointMover 연동)
    python run_dummy.py --full       # 허수아비 10초 후 강제 사냥터 전환 (leveling 모드)
    python run_dummy.py --waypoint   # 웨이포인트 이동 테스트 (field 모드)

종료:
    OpenCV 창에서 'q' 키  또는  Ctrl+C

모드 설명:
    leveling (기본/--full):
        HuntingStateMachine.start_at_dummy() → ATTACKING_DUMMY → ... → HUNTING_10
        허수아비 단계부터 자동 진행, 목표 레벨 도달 시 종료.

    field (--field/--waypoint):
        HuntingStateMachine.start_at_hunt_zone() → MOVE_TO_HUNT_ZONE → HUNTING_10
        patrol_waypoints 순찰하며 몬스터 탐지 + SequentialTargetSM 공격.
        tracker._sm.state == IDLE일 때만 다음 WP로 이동.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    from config.config_loader import load_config
    from main import load_automation_config, run

    config   = load_config()
    auto_cfg = load_automation_config()

    if auto_cfg is None:
        logger.error("config_automation.json 을 찾을 수 없습니다. 종료합니다.")
        sys.exit(1)

    # ── 실행 모드 결정 ──────────────────────────────────────────────────
    args = sys.argv[1:]

    if "--field" in args or "--waypoint" in args:
        mode = "field"
        logger.info("=== run_dummy.py [field 모드] — WaypointMover + SequentialTargetSM ===")
        logger.info("  patrol_waypoints 순찰 중 몬스터 탐지 + 자동 공격")
        logger.info("  tracker._sm.state == IDLE일 때만 다음 WP 이동")
    elif "--full" in args:
        mode = "leveling"
        # --full: target_level_dummy를 0으로 설정 → 즉시 USE_SPEED_POTION → 사냥터 이동
        auto_cfg.setdefault("level_ocr", {})
        auto_cfg["level_ocr"]["target_level_dummy"] = 0
        logger.info("=== run_dummy.py [leveling 모드 --full] — 허수아비 → 사냥터 ===")
        logger.info("  target_level_dummy=0 → 즉시 사냥터 이동")
    else:
        mode = "leveling"
        logger.info("=== run_dummy.py [leveling 모드] — 허수아비 공격부터 시작 ===")

    auto_cfg_path_info = os.path.join(HERE, "config", "config_automation.json")
    logger.info(f"  config          : {os.path.join(HERE, 'config', 'config.json')}")
    logger.info(f"  automation_cfg  : {auto_cfg_path_info}")
    logger.info(f"  mode            : {mode}")
    logger.info("  'q' 키로 종료 | OpenCV 창")
    logger.info("=" * 50)

    # ── main.py run() 호출 — 탐지 파이프라인 공용 ────────────────────
    # SceneMotionFilter + MOG2 + NearestNeighborTracker + SequentialTargetSM
    # HuntingStateMachine (tracker 주입, field 모드 시)
    run(
        config            = config,
        automation_config = auto_cfg,
        mode              = mode,
    )


if __name__ == "__main__":
    main()
