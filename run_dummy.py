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

    # ── monitor_index 먼저 로드 (Pico 오프셋 계산에 필요) ───────────────
    monitor_index = config.get("monitor_index", 2)

    # ── Pico 연결 ──────────────────────────────────────────────────────
    pico_cfg = config.get("pico", {})
    pico_enabled = pico_cfg.get("enabled", False)

    if pico_enabled:
        try:
            from pico.pico_serial import PicoSerialWorker
            import mss as _mss
            with _mss.MSS() as _sct:
                _monitors = _sct.monitors
                _mon = _monitors[monitor_index] if monitor_index < len(_monitors) else _monitors[1]
                mon_left = _mon["left"]
                mon_top  = _mon["top"]
            logger.info(f"  모니터 오프셋: left={mon_left}, top={mon_top}")

            port     = pico_cfg.get("serial_port", pico_cfg.get("port", "COM4"))
            baudrate = pico_cfg.get("baudrate", 115200)
            pico     = PicoSerialWorker(
                port=port,
                baudrate=baudrate,
                on_log=lambda lv, msg: logger.info(f"  [Pico/{lv}] {msg}"),
                on_command_result=lambda cmd, ok: logger.info(f"  [Pico] {cmd} → {'OK' if ok else 'FAIL'}"),
                monitor_offset_x=mon_left,
                monitor_offset_y=mon_top,
            )
            pico.start()
            time.sleep(1.0)   # 연결 안정화 대기
            logger.info(f"  Pico 연결: {port} @ {baudrate}  is_connected={pico.is_connected}")
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
    capturer = ScreenCapturer(monitor_index=monitor_index)
    logger.info(f"  monitor_index  : {monitor_index}")

    # ── HuntingStateMachine ────────────────────────────────────────────
    from automation.state_machine import HuntingStateMachine, HuntingState

    sm = HuntingStateMachine(
        config        = auto_cfg,
        pico_worker   = pico,
        frame_grabber = capturer.grab,
    )

    # 실행 모드 선택
    waypoint_test = "--waypoint" in sys.argv

    full_test = "--full" in sys.argv

    if waypoint_test:
        # 웨이포인트 이동만 테스트 (허수아비 생략)
        wp_cfg = auto_cfg.get("hunt_waypoints", {})
        logger.info("  [MODE] 웨이포인트 이동 테스트")
        logger.info(f"  웨이포인트 {len(wp_cfg.get('points',[]))}개, 대기 {wp_cfg.get('move_timeout_ms',8000)//1000}초/구간")
        sm.start_at_hunt_zone()
        logger.info("  start_at_hunt_zone() 호출 → MOVE_TO_HUNT_ZONE 진입")
    elif full_test:
        # 허수아비 10초 공격 후 강제 사냥터 이동 테스트
        sm.target_level_dummy = 999  # 레벨 달성 방지
        logger.info("  [MODE] 풀 테스트 (허수아비 10초 → 사냥터 이동)")
        logger.info("  target_level_dummy=999 (10초 후 강제 전환)")
        sm.start_at_dummy()
        logger.info("  start_at_dummy() 호출 → ATTACKING_DUMMY 진입")
    else:
        # 허수아비 공격부터 시작 (기본)
        sm.start_at_dummy()
        logger.info("  start_at_dummy() 호출 → ATTACKING_DUMMY 진입")

    logger.info("  Ctrl+C 로 종료")
    logger.info("=" * 50)

    # ── 메인 루프 ──────────────────────────────────────────────────────
    _diag_counter = 0
    _start_t = time.time()
    _full_switched = False
    try:
        while True:
            frame = capturer.grab()
            sm.update(frame, enemies=[])   # enemies: 적 감지 없이 허수아비 공격만

            status = sm.get_status()
            state  = status.get("state", "?")

            # --full 모드: 10초 후 강제 사냥터 이동 전환
            if full_test and not _full_switched and time.time() - _start_t >= 10.0:
                if state == "ATTACKING_DUMMY":
                    logger.info("  [FULL] 10초 경과 → 강제 사냥터 이동 전환")
                    from automation.state_machine import HuntingState
                    sm._enter(HuntingState.USE_SPEED_POTION)
                    _full_switched = True

            # 3초마다 Pico 스레드 상태 출력
            _diag_counter += 1
            if _diag_counter % 60 == 0 and pico_enabled:
                thread = getattr(pico, "_thread", None)
                alive  = thread.is_alive() if thread else False
                qsize  = getattr(pico, "_out_queue", None)
                qsize  = qsize.qsize() if qsize else -1
                conn   = pico.is_connected
                logger.info(f"  [DIAG] thread_alive={alive} queue={qsize} is_connected={conn}")

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
