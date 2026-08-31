"""DiagnosticRunner — 실제 게임 화면 기반 Dry-Run 진단 모드.

목적
────
Pico 명령을 전송하지 않으면서 실제 게임 화면을 캡처해
perception / decision 결과를 관찰한다.

실행 방법
─────────
    python run_diagnostic.py [--mode leveling|dungeon] [--interval 1.0]

절대 조건
─────────
1. 기존 main.py / run() 경로와 완전히 분리된다.
2. NullPicoWorker를 주입해 click/drag/key_tap_name이 실제로 실행되지 않는다.
3. 실제 ScreenCapturer.grab()을 사용한다.
4. logs/ 에 진단 로그를 저장한다.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# 프로젝트 루트를 sys.path에 추가 (직접 실행 지원)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# NullPicoWorker는 순수 Python — 플랫폼 의존 없으므로 모듈 레벨 임포트 OK
# capture / tracker 등 Windows 전용 모듈은 setup() 내부에서 임포트한다
from pico.null_pico import NullPicoWorker
# DiagRecord / infer_decision 은 순수 로직 모듈에서 가져온다
from diagnostic.models import DiagRecord, infer_decision as _infer_decision

logger = logging.getLogger("diagnostic")


# ──────────────────────────────────────────────────────────────────────────────
# LootDetector 백그라운드 스레드 (easyocr 호출을 메인 루프에서 분리)
# ──────────────────────────────────────────────────────────────────────────────

class _LootThread(threading.Thread):
    """LootDetector.find()를 별도 스레드에서 실행, 결과를 캐싱한다.

    메인 루프가 매 프레임 OCR을 기다리지 않도록 분리.
    scan_interval_s(기본 1.5s) 마다 OCR 재실행.
    """

    def __init__(self, loot_detector, interval_s: float = 1.5):
        super().__init__(daemon=True, name="LootDetectorThread")
        self._detector   = loot_detector
        self._interval   = interval_s
        self._lock       = threading.Lock()
        self._result: list = []
        self._stop_evt   = threading.Event()
        self._frame: Optional[np.ndarray] = None

    def push_frame(self, frame: np.ndarray) -> None:
        """메인 루프가 처리할 최신 프레임을 전달한다 (shallow copy 방지)."""
        with self._lock:
            self._frame = frame

    def get_result(self) -> list:
        """가장 최근 OCR 결과를 반환한다 (blocking 없음)."""
        with self._lock:
            return list(self._result)

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        logger.info("[LootThread] 시작")
        while not self._stop_evt.is_set():
            frame = None
            with self._lock:
                if self._frame is not None:
                    frame = self._frame.copy()

            if frame is not None:
                try:
                    # LootDetector 내부 캐시 무시하고 강제 재실행하려면
                    # _last_scan_time을 0으로 초기화
                    self._detector._last_scan_time = 0.0
                    result = self._detector.find(frame)
                    with self._lock:
                        self._result = result
                except Exception as e:
                    logger.warning(f"[LootThread] OCR 오류: {e}")

            self._stop_evt.wait(self._interval)
        logger.info("[LootThread] 종료")

# ── 오버레이 색상 상수 ─────────────────────────────────────────────────────
_COLOR_ENEMY       = (0, 255, 80)       # 일반 적: 초록
_COLOR_TARGET      = (0, 80, 255)       # 현재 타겟: 빨강
_COLOR_TEXT        = (255, 255, 255)    # 기본 흰색 텍스트
_COLOR_PANEL_BG    = (20, 20, 20)       # 패널 배경
_COLOR_MOVING      = (0, 165, 255)      # 이동 중 주황
_COLOR_HUNTING     = (0, 255, 0)        # HUNTING 녹색
_COLOR_IDLE        = (128, 128, 128)    # IDLE 회색
_COLOR_LOOTING     = (0, 255, 200)      # LOOTING 청록
_COLOR_OTHER       = (200, 200, 200)    # 기타 상태

# HuntingState별 오버레이 색상
_SM_COLOR_MAP = {
    "IDLE":              _COLOR_IDLE,
    "USE_SCROLL_DUMMY":  (255, 200,   0),
    "MOVE_TO_DUMMY":     (255, 165,   0),
    "ATTACKING_DUMMY":   (0,   200, 255),
    "USE_SPEED_POTION":  (200, 100, 255),
    "MOVE_TO_HUNT_ZONE": (0,   165, 255),
    "HUNTING_10":        _COLOR_HUNTING,
    "LOOTING":           _COLOR_LOOTING,
    "DONE_PHASE1":       (255, 255,   0),
}



# ──────────────────────────────────────────────────────────────────────────────
# CV2 오버레이 렌더
# ──────────────────────────────────────────────────────────────────────────────

def _draw_overlay(
    frame: np.ndarray,
    rec: DiagRecord,
    enemies: list,
    tracker: NearestNeighborTracker,
    roi_offset: tuple[int, int],
) -> np.ndarray:
    """진단 정보를 프레임에 그려 반환한다."""
    out = frame.copy()
    ox, oy = roi_offset

    # ── Enemy bounding boxes ─────────────────────────────────────────────────
    current_id = tracker.current_target_id
    for e in enemies:
        # Enemy 속성: x, y, width, height (ROI 좌표 기준)
        bx, by, bw, bh = e.x, e.y, e.width, e.height
        sx, sy = bx + ox, by + oy
        color = _COLOR_TARGET if e.id == current_id else _COLOR_ENEMY
        cv2.rectangle(out, (sx, sy), (sx + bw, sy + bh), color, 2)
        label = f"ID:{e.id}"
        cv2.putText(
            out, label,
            (sx, sy - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )

    # ── 이동 중 배너 ─────────────────────────────────────────────────────────
    if rec.is_moving:
        cv2.putText(
            out, "MOVING — DETECTION PAUSED",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, _COLOR_MOVING, 2, cv2.LINE_AA,
        )

    # ── DRY-RUN 워터마크 ─────────────────────────────────────────────────────
    h, w = out.shape[:2]
    cv2.putText(
        out, "[DRY-RUN / NO PICO]",
        (w - 340, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA,
    )

    # ── 왼쪽 정보 패널 ───────────────────────────────────────────────────────
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 340, 220
    overlay = out.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  _COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    sm_color = _SM_COLOR_MAP.get(rec.hunting_state, _COLOR_OTHER)

    rows = [
        ("PERCEPTION", _COLOR_TEXT),
        (f"  HP       : {rec.hp_pct:5.1f}%",  _COLOR_TEXT),
        (f"  Level    : {rec.level if rec.level is not None else '?'}",  _COLOR_TEXT),
        (f"  Moving   : {rec.is_moving}",      _COLOR_MOVING if rec.is_moving else _COLOR_TEXT),
        (f"  Enemies  : {rec.enemy_count}",    _COLOR_TEXT),
        (f"  Tracked  : {rec.tracked_count}",  _COLOR_TEXT),
        (f"  Loot     : {rec.loot_count}",     _COLOR_TEXT),
        ("STATE", _COLOR_TEXT),
        (f"  Hunting  : {rec.hunting_state}", sm_color),
        (f"  Target   : {rec.target_id if rec.target_id is not None else '-'}", _COLOR_TEXT),
        (f"  TargetSM : {rec.target_state}", _COLOR_TEXT),
        ("DECISION", _COLOR_TEXT),
        (f"  {rec.decision} — {rec.decision_reason}", (80, 255, 80)),
    ]

    line_h = 16
    for i, (text, color) in enumerate(rows):
        y_pos = panel_y + 16 + i * line_h
        cv2.putText(
            out, text,
            (panel_x + 6, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA,
        )

    # ── 하단 frame_ms ────────────────────────────────────────────────────────
    cv2.putText(
        out, f"frame {rec.frame_ms:.1f}ms",
        (w - 180, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
    )

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 클래스
# ──────────────────────────────────────────────────────────────────────────────

class DiagnosticRunner:
    """실제 게임 화면을 캡처해 perception/decision 결과를 관찰한다.

    Pico 명령은 NullPicoWorker를 통해 완전히 차단된다.
    HuntingStateMachine은 선택적으로 연결할 수 있다 (SM 없이도 동작).
    """

    WINDOW_NAME = "Diagnostic Dry-Run"

    def __init__(
        self,
        config: dict,
        automation_config: Optional[dict] = None,
        mode: str = "leveling",
        log_interval_s: float = 1.0,
        save_screenshots: bool = False,
        log_dir: str = "logs",
    ):
        """
        Args:
            config:           config.json 내용
            automation_config: config_automation.json 내용 (없으면 SM 비활성)
            mode:             "leveling" | "dungeon"
            log_interval_s:   콘솔/파일 로그 출력 간격 (초)
            save_screenshots: True면 1초마다 debug screenshot 저장
            log_dir:          로그 파일 저장 디렉토리
        """
        self.cfg              = config
        self.auto_cfg         = automation_config
        self.mode             = mode
        self.log_interval     = log_interval_s
        self.save_screenshots = save_screenshots
        self.log_dir          = log_dir

        # NullPico — 명령 차단
        self.pico = NullPicoWorker()

        # 로그 파일 설정
        os.makedirs(log_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path        = os.path.join(log_dir, f"diagnostic_{ts_str}.log")
        self._screenshot_dir  = os.path.join(log_dir, f"screens_{ts_str}")
        self._file_handler: Optional[logging.FileHandler] = None
        self._last_log_time   = 0.0
        self._last_screen_time = 0.0
        self._tick_count      = 0

        # 내부 모듈 (setup()에서 초기화)
        self._capturer:       Optional[ScreenCapturer]          = None
        self._motion_det:     Optional[MotionDetector]          = None
        self._contour_det:    Optional[ContourDetector]         = None
        self._scene_filter:   Optional[SceneMotionFilter]       = None
        self._tracker:        Optional[NearestNeighborTracker]  = None
        self._hunting_sm                                        = None
        self._hp_reader                                         = None
        self._level_reader                                      = None
        self._loot_detector                                     = None
        # LootDetector 백그라운드 스레드 (easyocr 분리로 frame 속도 개선)
        self._loot_thread: Optional[_LootThread]                = None

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """모든 perception 모듈을 초기화한다."""
        # Windows 전용 모듈 — setup() 시점에 임포트 (pytest 모듈 레벨 실패 방지)
        from capture.screen_capture import ScreenCapturer
        from detection.contour_detector import ContourDetector
        from detection.motion_detector import MotionDetector, SceneMotionFilter
        from tracking.tracker import NearestNeighborTracker

        cfg = self.cfg
        is_lev = (self.mode == "leveling")

        # 파일 로거
        fh = logging.FileHandler(self._log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)
        self._file_handler = fh
        logger.info(f"[Diagnostic] 로그 파일: {self._log_path}")
        logger.info(f"[Diagnostic] 모드: {self.mode}  |  NullPico 주입됨 (Pico 명령 차단)")

        # ScreenCapturer
        self._capturer = ScreenCapturer(
            cfg["monitor_index"],
            cfg.get("capture_region"),
        )

        # MotionDetector / ContourDetector
        self._motion_det = MotionDetector(
            blur_kernel  = cfg["blur_kernel"],
            morph_kernel = cfg["morph_kernel"],
            **cfg["motion"],
        )
        self._contour_det = ContourDetector(cfg["min_area"], cfg["max_area"])

        # SceneMotionFilter
        sm_cfg = cfg.get("scene_motion", {})
        if sm_cfg.get("enabled", True):
            self._scene_filter = SceneMotionFilter(
                scene_threshold     = sm_cfg.get("scene_threshold",     12.0),
                settle_frames       = sm_cfg.get("settle_frames",         5),
                downscale           = sm_cfg.get("downscale",             4),
                crop_ratio          = sm_cfg.get("crop_ratio",          0.6),
                move_confirm_frames = sm_cfg.get("move_confirm_frames",   2),
            )

        # roi/capture offset
        cap_region  = cfg.get("capture_region")
        cap_offset  = (cap_region["x"], cap_region["y"]) if cap_region else (0, 0)
        roi_dict    = cfg.get("roi")
        roi_offset  = (roi_dict["x"], roi_dict["y"]) if roi_dict else (0, 0)
        self._roi_offset = roi_offset
        self._roi_dict   = roi_dict

        # NearestNeighborTracker — NullPico 콜백 (실제로는 아무것도 안 함)
        pico_cfg = cfg.get("pico", {})

        def _null_click(x, y):
            self.pico.click(x, y)

        def _null_drag(fx, fy, tx, ty):
            self.pico.drag(fx, fy, tx, ty)

        drag_enabled = pico_cfg.get("drag_enabled", False)

        self._tracker = NearestNeighborTracker(
            max_missing_frames      = cfg["max_missing_frames"],
            max_match_distance      = cfg["max_match_distance"],
            pico_click_callback     = _null_click,
            pico_drag_callback      = _null_drag if drag_enabled else None,
            roi_offset              = roi_offset,
            capture_region_offset   = cap_offset,
            lock_confirm_frames     = pico_cfg.get("lock_confirm_frames",      2),
            wait_dead_timeout_ms    = pico_cfg.get("wait_dead_timeout_ms",  5000.0),
            next_target_cooldown_ms = pico_cfg.get("next_target_cooldown_ms", 300.0),
            target_priority         = pico_cfg.get("target_priority", "nearest_center"),
            roi_width               = roi_dict.get("width",  1440) if roi_dict else 1440,
            roi_height              = roi_dict.get("height",  780) if roi_dict else  780,
            drag_enabled            = drag_enabled,
            drag_dx                 = pico_cfg.get("drag_dx", 80),
            drag_dy                 = pico_cfg.get("drag_dy",  0),
            drag_steps              = pico_cfg.get("drag_steps", 8),
        )

        # HuntingStateMachine (optional — automation_config가 있고 leveling 모드일 때)
        if is_lev and self.auto_cfg is not None:
            try:
                from automation.state_machine import HuntingStateMachine
                self._hunting_sm = HuntingStateMachine(
                    config        = self.auto_cfg,
                    pico_worker   = self.pico,          # NullPico 주입
                    frame_grabber = self._capturer.grab,
                )
                # start() 호출하지 않음 — Diagnostic에서는 SM을 관찰만 한다.
                # SM을 실제로 구동하려면 start()를 호출하면 되지만,
                # Dry-Run 목적상 IDLE 상태에서 get_status()만 읽는다.
                logger.info("[Diagnostic] HuntingStateMachine 연결됨 (IDLE, start() 미호출)")

                # HpReader / LevelReader / LootDetector는 SM 내부에서 생성되므로
                # 별도 접근을 위해 직접 참조
                self._hp_reader    = self._hunting_sm.hp_reader
                self._level_reader = self._hunting_sm.level_reader
                self._loot_detector = self._hunting_sm.loot_detector
            except Exception as e:
                logger.warning(f"[Diagnostic] HuntingStateMachine 초기화 실패: {e}")
                self._hunting_sm = None

        # LootDetector 백그라운드 스레드 시작 (easyocr 분리)
        if self._loot_detector is not None:
            loot_interval = (
                self.auto_cfg.get("loot", {}).get("scan_interval_s", 1.5)
                if self.auto_cfg else 1.5
            )
            self._loot_thread = _LootThread(self._loot_detector, interval_s=max(loot_interval, 1.0))
            self._loot_thread.start()
            logger.info(f"[Diagnostic] LootThread 시작 (간격={max(loot_interval,1.0):.1f}s)")

        logger.info("[Diagnostic] setup 완료. 'q' 키로 종료.")

    def run(self, stop_event=None) -> None:
        """메인 진단 루프."""
        if self._capturer is None:
            self.setup()

        cfg         = self.cfg
        roi_dict    = self._roi_dict
        roi_offset  = self._roi_offset

        capture_interval  = 1.0 / max(cfg["capture_fps"], 1)
        detection_interval = 1.0 / max(cfg["detection_fps"], 1)

        last_capture_time   = 0.0
        last_detection_time = 0.0
        last_tracker_update = time.time()
        is_moving           = False
        enemies             = []

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while not (stop_event and stop_event.is_set()):
                loop_start = time.time()

                # ── 캡처 간격 대기 ────────────────────────────────────────
                wait = capture_interval - (loop_start - last_capture_time)
                if wait > 0:
                    time.sleep(wait)

                t0    = time.time()
                frame = self._capturer.grab()
                last_capture_time = t0

                # LootThread에 최신 프레임 전달 (non-blocking)
                if self._loot_thread is not None:
                    self._loot_thread.push_frame(frame)

                # ROI 슬라이스
                if roi_dict:
                    rx, ry = roi_dict["x"], roi_dict["y"]
                    rw, rh = roi_dict["width"], roi_dict["height"]
                    roi_frame = frame[ry:ry + rh, rx:rx + rw]
                else:
                    roi_frame = frame
                    rx, ry = 0, 0

                # ── SceneMotionFilter ──────────────────────────────────────
                if self._scene_filter is not None:
                    is_moving = self._scene_filter.update(roi_frame)
                    if is_moving:
                        self._tracker._sm.reset()

                # ── Detection (정지 중일 때만) ─────────────────────────────
                now = time.time()
                if now - last_detection_time >= detection_interval:
                    if not is_moving:
                        mask       = self._motion_det.get_mask(roi_frame, learning_rate=-1.0)
                        detections = self._contour_det.detect(mask)

                        # detection_zone 필터
                        dz = cfg.get("detection_zone")
                        if dz and dz.get("enabled", False):
                            rw_ = roi_dict.get("width",  1440) if roi_dict else 1440
                            rh_ = roi_dict.get("height",  780) if roi_dict else  780
                            cx  = dz.get("center_x", rw_ // 2)
                            cy  = dz.get("center_y", rh_ // 2)
                            hw  = dz.get("half_width",  400)
                            hh  = dz.get("half_height", 300)
                            x0, y0, x1, y1 = cx - hw, cy - hh, cx + hw, cy + hh
                            detections = [
                                d for d in detections
                                if x0 <= d.center_x <= x1 and y0 <= d.center_y <= y1
                            ]

                        dt = now - last_tracker_update
                        last_tracker_update = now

                        # TargetSM 활성 여부 (HUNTING_10일 때만)
                        if self._hunting_sm is not None:
                            from automation.state_machine import HuntingState
                            _is_hunting = (
                                self._hunting_sm.state == HuntingState.HUNTING_10
                            )
                            self._tracker._sm.set_active(_is_hunting)

                        enemies = self._tracker.update(detections, dt if dt > 0 else 1e-3)

                        # HuntingStateMachine update — SM이 start()된 상태에서만 의미 있음
                        if self._hunting_sm is not None:
                            self._hunting_sm.update(roi_frame, enemies)
                    else:
                        self._motion_det.get_mask(roi_frame, learning_rate=0.0)
                        enemies = []

                    last_detection_time = now

                # ── DiagRecord 빌드 ────────────────────────────────────────
                rec = self._build_record(frame, enemies, is_moving, now)
                rec.frame_ms = (time.time() - loop_start) * 1000
                self._tick_count += 1

                # ── 로그 출력 (interval) ───────────────────────────────────
                if now - self._last_log_time >= self.log_interval:
                    self._last_log_time = now
                    self._emit_log(rec)

                # ── Screenshot 저장 ────────────────────────────────────────
                if self.save_screenshots and (now - self._last_screen_time >= 1.0):
                    self._last_screen_time = now
                    self._save_screenshot(frame, rec, now)

                # ── CV2 오버레이 표시 ──────────────────────────────────────
                display = _draw_overlay(frame, rec, enemies, self._tracker, roi_offset)
                cv2.imshow(self.WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("[Diagnostic] 'q' 입력 — 종료")
                    break
                if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("[Diagnostic] 창 닫힘 — 종료")
                    break

        finally:
            self._shutdown()

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _build_record(
        self,
        frame: np.ndarray,
        enemies: list,
        is_moving: bool,
        now: float,
    ) -> DiagRecord:
        rec = DiagRecord()
        rec.timestamp     = now
        rec.is_moving     = is_moving
        rec.enemy_count   = len(enemies)
        rec.tracked_count = len(enemies)

        # HP / Level / Loot — SM이 연결돼 있으면 내부 reader 사용
        if self._hp_reader is not None:
            rec.hp_pct = self._hp_reader.read(frame)
        if self._level_reader is not None:
            lv = self._level_reader.read(frame)   # OCR 실행 (2초 캐시 적용)
            rec.level = lv
        if self._loot_thread is not None:
            # 백그라운드 스레드 결과 읽기 (non-blocking)
            rec.loot_count = len(self._loot_thread.get_result())
        elif self._loot_detector is not None:
            rec.loot_count = len(self._loot_detector.find(frame))

        # HuntingStateMachine 상태
        if self._hunting_sm is not None:
            st = self._hunting_sm.get_status()
            rec.hunting_state = st.get("state", "N/A")
            rec.hp_pct        = st.get("hp_pct", rec.hp_pct)
            lv_st = st.get("level")
            if lv_st != "?":
                rec.level = lv_st
        else:
            rec.hunting_state = "N/A"

        # Tracker / TargetSM 상태
        rec.target_id    = self._tracker.current_target_id
        rec.target_state = self._tracker.target_state.name

        # Decision 추론
        rec.decision, rec.decision_reason = _infer_decision(rec)

        return rec

    def _emit_log(self, rec: DiagRecord) -> None:
        """콘솔 + 파일에 structured block 출력."""
        lines = rec.to_log_lines()
        block = "\n".join(lines)
        # 콘솔 출력 (logger 경유 — 타임스탬프 없이 raw 출력)
        print(block, flush=True)
        # 파일에는 logger 경유로 기록 (FileHandler가 수신)
        logger.info(block)

    def _save_screenshot(
        self,
        frame: np.ndarray,
        rec: DiagRecord,
        now: float,
    ) -> None:
        """debug screenshot을 파일로 저장한다."""
        os.makedirs(self._screenshot_dir, exist_ok=True)
        ts_str = datetime.fromtimestamp(now).strftime("%H%M%S_%f")[:-3]
        fname  = os.path.join(self._screenshot_dir, f"screen_{ts_str}.jpg")
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        logger.debug(f"[Diagnostic] screenshot → {fname}")

    def _shutdown(self) -> None:
        """종료 정리."""
        logger.info(
            f"[Diagnostic] 종료. "
            f"총 tick={self._tick_count}  "
            f"로그={self._log_path}"
        )
        # LootThread 종료 (daemon이지만 명시적으로 stop)
        if self._loot_thread is not None:
            self._loot_thread.stop()
            self._loot_thread.join(timeout=3.0)
        if self._capturer:
            self._capturer.close()
        cv2.destroyAllWindows()
        if self._file_handler:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
