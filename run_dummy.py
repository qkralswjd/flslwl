"""허수아비 공격 전용 실행 스크립트.

1~5레벨 허수아비 단계를 텔레포트 없이 즉시 시작합니다.
Pico가 연결돼 있으면 실제 드래그 공격이 실행됩니다.

실행:
    python run_dummy.py

종료:
    Ctrl+C

동작:
    - ATTACKING_DUMMY 상태로 즉시 진입
    - drag_from → drag_to 방향으로 attack_interval_ms마다 드래그
    - 레벨이 target_level_dummy(기본 5) 이상이면 자동 종료
    - hp < threshold_pct(기본 50%)이면 USE_POTION 결정 출력
"""

import json
import logging
import os
import sys
import time

# ── 로깅 설정 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_dummy")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # ── 설정 로드 ──────────────────────────────────────────────────────
    cfg_path      = os.path.join(HERE, "config", "config.json")
    auto_cfg_path = os.path.join(HERE, "config", "config_automation.json")

    config   = load_json(cfg_path)
    auto_cfg = load_json(auto_cfg_path)

    dummy_cfg = auto_cfg.get("dummy", {})
    logger.info("=== run_dummy.py 시작 ===")
    logger.info(f"  drag_from      : {dummy_cfg.get('drag_from')}")
    logger.info(f"  drag_to        : {dummy_cfg.get('drag_to')}")
    logger.info(f"  attack_interval: {dummy_cfg.get('attack_interval_ms')}ms")
    logger.info(f"  target_level   : Lv.{auto_cfg.get('level_ocr',{}).get('target_level_dummy',5)}")

    # ── Pico 연결 ──────────────────────────────────────────────────────
    pico_cfg = config.get("pico", {})
    pico_enabled = pico_cfg.get("enabled", False)

    if pico_enabled:
        try:
            from pico.pico_serial import PicoSerialWorker
            port     = pico_cfg.get("port", "COM3")
            baudrate = pico_cfg.get("baudrate", 115200)
            pico     = PicoSerialWorker(port=port, baudrate=baudrate)
            pico.start()
            logger.info(f"  Pico 연결: {port} @ {baudrate}")
        except Exception as e:
            logger.warning(f"  Pico 연결 실패: {e} → NullPicoWorker 사용")
            from pico.null_pico import NullPicoWorker
            pico = NullPicoWorker()
    else:
        logger.info("  Pico 비활성화 → NullPicoWorker (Dry-Run)")
        from pico.null_pico import NullPicoWorker
        pico = NullPicoWorker()

    # ── ScreenCapturer ─────────────────────────────────────────────────
    from capture.screen_capture import ScreenCapturer
    monitor_index = config.get("monitor_index", 2)
    capturer = ScreenCapturer(monitor_index=monitor_index)
    logger.info(f"  monitor_index  : {monitor_index}")

    # ── HuntingStateMachine ────────────────────────────────────────────
    from automation.state_machine import HuntingStateMachine, HuntingState

    sm = HuntingStateMachine(
        config        = auto_cfg,
        pico_worker   = pico,
        frame_grabber = capturer.grab,
    )

    # 텔레포트 없이 허수아비 공격부터 즉시 시작
    sm.start_at_dummy()
    logger.info("  start_at_dummy() 호출 → ATTACKING_DUMMY 진입")
    logger.info("  Ctrl+C 로 종료")
    logger.info("=" * 50)

    # ── 메인 루프 ──────────────────────────────────────────────────────
    try:
        while True:
            frame = capturer.grab()
            sm.update(frame, enemies=[])   # enemies: 적 감지 없이 허수아비 공격만

            status = sm.get_status()
            state  = status.get("state", "?")

            # 완료 또는 IDLE(오류) 감지
            if state in ("DONE_PHASE1", "IDLE"):
                logger.info(f"  상태: {state} → 종료")
                break

            time.sleep(0.05)   # 50ms 간격 (20fps)

    except KeyboardInterrupt:
        logger.info("  Ctrl+C — 종료")
    finally:
        capturer.close()
        if pico_enabled and hasattr(pico, "stop"):
            pico.stop()
        logger.info("=== run_dummy.py 종료 ===")


if __name__ == "__main__":
    main()
