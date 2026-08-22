"""Entry point: capture -> detect -> track -> overlay -> (optional) Pico click/drag.

Run directly for a plain OpenCV-window version:
    python main.py

`run()` is factored out so ui/settings_window.py can launch/stop the same
loop from a background thread and read live status back into the UI.

Pico 통합:
    config["pico"]["enabled"] == true 일 때
    PicoSerialWorker를 백그라운드 스레드로 시작하고,
    NearestNeighborTracker에 click/drag 콜백을 주입합니다.

SceneMotionFilter 통합:
    플레이어가 이동해 화면 전체가 움직일 때 감지를 자동으로 일시정지합니다.
    config["scene_motion"]["enabled"] == true 일 때 활성화됩니다.

HuntingStateMachine 통합 (1단계 자동 레벨링):
    config_automation.json 기반으로 동작.
    automation_config가 전달되면 자동 레벨링 상태머신이 활성화됩니다.
"""

import json
import logging
import os
import time
from typing import Optional

import cv2

from capture.screen_capture import ScreenCapturer
from config.config_loader import load_config
from paths import get_templates_dir, get_reject_templates_dir
from debug.debug_view import DebugView
from detection.classifier import MonsterClassifier
from detection.contour_detector import ContourDetector
from detection.motion_detector import MotionDetector, SceneMotionFilter
from overlay.overlay import draw_enemies, draw_hud, draw_roi, draw_detection_zone
from tracking.tracker import NearestNeighborTracker


def load_automation_config(path: str = None) -> Optional[dict]:
    """config_automation.json을 로드합니다. 파일 없으면 None 반환."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "config", "config_automation.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("main")

WINDOW_NAME = "Game Enemy Tracker (+ Pico)"


def _slice_roi(frame, roi):
    """Returns (sub_image, (offset_x, offset_y)). roi=None -> whole frame."""
    if roi is None:
        return frame, (0, 0)
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    return frame[y:y + h, x:x + w], (x, y)


def _build_pico_worker(pico_cfg: dict):
    """Pico 워커 인스턴스를 생성하고 시작합니다. 실패 시 None 반환."""
    if not pico_cfg.get("enabled", False):
        return None

    try:
        from pico.pico_serial import PicoSerialWorker
    except ImportError:
        logger.warning("pyserial이 설치되어 있지 않아 Pico 기능을 비활성화합니다.")
        return None

    port     = pico_cfg.get("serial_port", "COM3")
    baudrate = pico_cfg.get("baudrate", 115200)

    worker = PicoSerialWorker(
        port=port,
        baudrate=baudrate,
        on_connected=lambda: logger.info(f"[Pico] 연결됨: {port}"),
        on_disconnected=lambda r: logger.warning(f"[Pico] 연결 끊김: {r}"),
        on_log=lambda lv, msg: logger.info(f"[Pico][{lv}] {msg}"),
        on_command_result=lambda cmd, ok: logger.debug(f"[Pico] {cmd} {'OK' if ok else 'FAIL'}"),
    )
    worker.start()
    logger.info(f"[Pico] 워커 시작 (port={port})")
    return worker


def run(config, stop_event=None, status_callback=None, automation_config=None):
    """Runs the capture/detect/track/overlay loop until 'q' is pressed or stop_event is set."""

    # ── HuntingStateMachine 초기화 (자동 레벨링) ──────────────────────
    hunting_sm = None
    if automation_config is not None:
        try:
            from automation.state_machine import HuntingStateMachine
            # capturer가 아직 없으므로 나중에 grab 함수 주입 (아래에서 처리)
            _hunting_sm_config   = automation_config
            _hunting_sm_pending  = True   # capturer 생성 후 초기화 예정
        except ImportError as e:
            logger.warning(f"[Automation] automation 모듈 로드 실패: {e}")
            _hunting_sm_pending  = False
    else:
        _hunting_sm_pending = False

    # ── Pico 워커 시작 ─────────────────────────────────────────────────
    pico_cfg    = config.get("pico", {})
    pico_worker = _build_pico_worker(pico_cfg)
    pulse_ms    = pico_cfg.get("click_pulse_ms", 20)
    offset_x    = pico_cfg.get("click_offset_x", 0)
    offset_y    = pico_cfg.get("click_offset_y", 0)

    # Pico 클릭 콜백
    def pico_click(screen_x: int, screen_y: int):
        if pico_worker and pico_worker.is_connected:
            pico_worker.click(screen_x + offset_x, screen_y + offset_y, pulse_ms)

    # Pico 드래그 콜백 (from_x, from_y, to_x, to_y)
    def pico_drag(fx: int, fy: int, tx: int, ty: int):
        if pico_worker and pico_worker.is_connected:
            pico_worker.drag(
                fx + offset_x, fy + offset_y,
                tx + offset_x, ty + offset_y,
                steps=pico_cfg.get("drag_steps", 8),
            )

    drag_enabled = pico_cfg.get("drag_enabled", False)
    click_callback = pico_click if pico_worker else None
    drag_callback  = pico_drag  if (pico_worker and drag_enabled) else None

    # ── SceneMotionFilter 초기화 ───────────────────────────────────────
    sm_cfg = config.get("scene_motion", {})
    scene_filter = (
        SceneMotionFilter(
            scene_threshold     = sm_cfg.get("scene_threshold",     12.0),
            settle_frames       = sm_cfg.get("settle_frames",         5),
            downscale           = sm_cfg.get("downscale",             4),
            crop_ratio          = sm_cfg.get("crop_ratio",          0.6),
            move_confirm_frames = sm_cfg.get("move_confirm_frames",   2),
        )
        if sm_cfg.get("enabled", True)
        else None
    )

    # capture_region → 모니터 내 offset
    cap_region = config.get("capture_region")
    cap_offset = (cap_region["x"], cap_region["y"]) if cap_region else (0, 0)

    roi_dict   = config.get("roi")
    roi_offset = (roi_dict["x"], roi_dict["y"]) if roi_dict else (0, 0)

    # ── 각 모듈 초기화 ─────────────────────────────────────────────────
    capturer = ScreenCapturer(config["monitor_index"], config.get("capture_region"))

    # capturer 생성 후 HuntingStateMachine 초기화 + 자동 시작
    if _hunting_sm_pending and pico_worker:
        from automation.state_machine import HuntingStateMachine
        hunting_sm = HuntingStateMachine(
            config        = _hunting_sm_config,
            pico_worker   = pico_worker,
            frame_grabber = capturer.grab,
        )
        hunting_sm.start()   # 자동으로 TELEPORTING 상태부터 시작
        logger.info("[Automation] HuntingStateMachine 시작 — TELEPORTING")
    elif _hunting_sm_pending and not pico_worker:
        logger.warning("[Automation] Pico 미연결 — HuntingStateMachine 비활성화")
    motion_detector = MotionDetector(
        blur_kernel=config["blur_kernel"],
        morph_kernel=config["morph_kernel"],
        **config["motion"],
    )
    contour_detector = ContourDetector(config["min_area"], config["max_area"])

    tracker = NearestNeighborTracker(
        max_missing_frames      = config["max_missing_frames"],
        max_match_distance      = config["max_match_distance"],
        pico_click_callback     = click_callback,
        pico_drag_callback      = drag_callback,
        roi_offset              = roi_offset,
        capture_region_offset   = cap_offset,
        # ── 순차 타겟 상태머신 파라미터 ─────────────────────────
        lock_confirm_frames     = pico_cfg.get("lock_confirm_frames",      2),
        wait_dead_timeout_ms    = pico_cfg.get("wait_dead_timeout_ms",  5000.0),
        next_target_cooldown_ms = pico_cfg.get("next_target_cooldown_ms", 300.0),
        target_priority         = pico_cfg.get("target_priority", "nearest_center"),
        roi_width               = roi_dict.get("width",  1440) if roi_dict else 1440,
        roi_height              = roi_dict.get("height",  780) if roi_dict else  780,
        # ── 드래그 파라미터 ─────────────────────────────────────
        drag_enabled            = drag_enabled,
        drag_dx                 = pico_cfg.get("drag_dx", 80),
        drag_dy                 = pico_cfg.get("drag_dy",  0),
        drag_steps              = pico_cfg.get("drag_steps", 8),
    )

    clf_config = config.get("classifier", {})
    classifier = (
        MonsterClassifier(
            clf_config.get("templates_dir") or get_templates_dir(),
            clf_config.get("reject_templates_dir") or get_reject_templates_dir(),
            confidence_threshold=clf_config.get("confidence_threshold", 0.5),
            excluded_color_ranges=clf_config.get("excluded_color_ranges"),
            excluded_color_ratio=clf_config.get("excluded_color_ratio", 0.2),
            min_size_ratio=clf_config.get("min_size_ratio", 0.6),
        )
        if clf_config.get("enabled")
        else None
    )

    debug_view = (
        DebugView(motion_detector, contour_detector, WINDOW_NAME, classifier=classifier)
        if config.get("show_debug")
        else None
    )

    capture_interval   = 1.0 / max(config["capture_fps"], 1)
    detection_interval = 1.0 / max(config["detection_fps"], 1)

    last_capture_time    = 0.0
    last_detection_time  = 0.0
    last_tracker_update  = time.time()
    capture_fps_smooth   = 0.0
    detection_fps_smooth = 0.0
    enemies              = []
    last_mask            = None
    last_roi_frame       = None
    is_moving            = False   # SceneMotionFilter 결과

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if debug_view:
        debug_view.create_trackbars()

    pico_label = f"port={pico_cfg.get('serial_port','?')}" if pico_cfg.get("enabled") else "disabled"
    scene_label = "ON" if scene_filter else "OFF"
    logger.info(
        "=== Game Enemy Tracker + Pico 시작 (Pico: %s | SceneFilter: %s)"
        " | 종료: 'q' | 몬스터 캡처: 't' | 거부 캡처: 'r' ===",
        pico_label, scene_label,
    )

    try:
        while not (stop_event and stop_event.is_set()):
            loop_start = time.time()

            wait = capture_interval - (loop_start - last_capture_time)
            if wait > 0:
                time.sleep(wait)

            capture_start   = time.time()
            frame           = capturer.grab()
            capture_elapsed = time.time() - capture_start
            last_capture_time = capture_start
            if capture_elapsed > 0:
                capture_fps_smooth = 0.9 * capture_fps_smooth + 0.1 * (1.0 / capture_elapsed)

            roi_frame, roi_offset_val = _slice_roi(frame, config.get("roi"))

            # ── SceneMotionFilter: 이동 중이면 감지 스킵 ──────────────
            if scene_filter is not None:
                is_moving = scene_filter.update(roi_frame)
                if is_moving:
                    # 상태머신도 리셋 (이동 중 쌓인 적 목록 초기화)
                    tracker._sm.reset()

            now = time.time()
            if now - last_detection_time >= detection_interval:
                if not is_moving:
                    # ── 정지 중일 때만 감지 실행 ──────────────────────
                    if debug_view:
                        debug_view.read_trackbars()

                    proc_start = time.time()
                    # 정지 중: 정상 학습 (learningRate=-1 → MOG2 자동)
                    mask       = motion_detector.get_mask(roi_frame, learning_rate=-1.0)
                    detections = contour_detector.detect(mask)

                    # ── detection_zone 필터 ────────────────────────────
                    dz = config.get("detection_zone")
                    if dz and dz.get("enabled", False):
                        rw = roi_dict.get("width",  1440) if roi_dict else 1440
                        rh = roi_dict.get("height",  780) if roi_dict else  780
                        cx = dz.get("center_x", rw // 2)
                        cy = dz.get("center_y", rh // 2)
                        hw = dz.get("half_width",  400)
                        hh = dz.get("half_height", 300)
                        x0, y0 = cx - hw, cy - hh
                        x1, y1 = cx + hw, cy + hh
                        detections = [
                            d for d in detections
                            if x0 <= d.center_x <= x1 and y0 <= d.center_y <= y1
                        ]

                    if classifier is not None:
                        detections = classifier.confirm(detections, roi_frame)

                    dt = now - last_tracker_update
                    last_tracker_update = now
                    enemies = tracker.update(detections, dt if dt > 0 else 1e-3)

                    # ── HuntingStateMachine 업데이트 ──────────────────
                    if hunting_sm is not None:
                        hunting_sm.update(roi_frame, enemies)

                    proc_elapsed_ms = (time.time() - proc_start) * 1000
                    if proc_elapsed_ms > 0:
                        detection_fps_smooth = 0.9 * detection_fps_smooth + 0.1 * (1000.0 / proc_elapsed_ms)

                    last_mask      = mask
                    last_roi_frame = roi_frame
                else:
                    # ── 이동 중: MOG2 학습 차단 + 적 목록 비우기 ───────
                    # learningRate=0 → 배경 모델 동결 (오염 방지)
                    motion_detector.get_mask(roi_frame, learning_rate=0.0)
                    enemies = []

                last_detection_time = now

            # ── 화면 그리기 ────────────────────────────────────────────
            display_frame = frame.copy()
            draw_roi(display_frame, config.get("roi"))
            draw_detection_zone(display_frame, config.get("detection_zone"), roi_offset_val)

            # 이동 중이면 오버레이에 "MOVING" 표시
            if is_moving:
                cv2.putText(
                    display_frame, "MOVING - DETECTION PAUSED",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 165, 255), 2, cv2.LINE_AA,
                )

            # ── HuntingStateMachine 상태 오버레이 ─────────────────────
            if hunting_sm is not None:
                sm_status = hunting_sm.get_status()
                sm_text   = (
                    f"[AUTO] {sm_status['state']}  "
                    f"Lv.{sm_status['level']}  "
                    f"Kill:{sm_status['kills']}  "
                    f"{sm_status['elapsed_min']}min"
                )
                # 상태별 색상
                from automation.state_machine import HuntingState as HS
                color_map = {
                    "IDLE":            (128, 128, 128),
                    "TELEPORTING":     (255, 200,   0),
                    "MOVING_TO_DUMMY": (255, 165,   0),
                    "ATTACKING_DUMMY": (0,   200, 255),
                    "PATROLLING":      (0,   165, 255),
                    "HUNTING":         (0,   255,   0),
                    "LOOTING":         (0,   255, 200),
                    "DONE":            (255, 255,   0),
                }
                sm_color = color_map.get(sm_status["state"], (200, 200, 200))
                cv2.putText(
                    display_frame, sm_text,
                    (20, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    sm_color, 2, cv2.LINE_AA,
                )

            draw_enemies(
                display_frame, enemies, roi_offset_val,
                current_target_id = tracker.current_target_id,
                target_state_name = tracker.target_state.name,
            )

            pico_connected = pico_worker.is_connected if pico_worker else False
            draw_hud(
                display_frame,
                capture_fps_smooth,
                detection_fps_smooth,
                (time.time() - loop_start) * 1000,
                len(enemies),
                pico_connected    = pico_connected,
                pico_port         = pico_cfg.get("serial_port", "") if pico_cfg.get("enabled") else None,
                target_state_name = tracker.target_state.name,
                current_target_id = tracker.current_target_id,
            )
            if debug_view and last_mask is not None:
                debug_view.draw_into(display_frame, last_roi_frame, last_mask)

            cv2.imshow(WINDOW_NAME, display_frame)

            if status_callback:
                status_callback(
                    len(enemies),
                    capture_fps_smooth,
                    detection_fps_smooth,
                    pico_connected,
                    target_state_name = tracker.target_state.name,
                    current_target_id = tracker.current_target_id,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t") and classifier is not None:
                x, y, w, h = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=True)
                if w > 0 and h > 0:
                    classifier.save_template(frame[y:y + h, x:x + w])
            if key == ord("r") and classifier is not None:
                x, y, w, h = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=True)
                if w > 0 and h > 0:
                    classifier.save_reject_template(frame[y:y + h, x:x + w])
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if pico_worker:
            pico_worker.stop_target()
            pico_worker.stop()
            logger.info("[Pico] 워커 종료")
        capturer.close()
        cv2.destroyAllWindows()
        logger.info("=== 종료 ===")


if __name__ == "__main__":
    run(load_config())
