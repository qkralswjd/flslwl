"""modes/hunt_loop.py — 탐지 → 추적 → 공격 → 루팅 무한 사냥 루프.

                    ┌──────────────────────────────────────────┐
                    │            HuntLoop (백그라운드 스레드)         │
                    │                                          │
                    │  grab()  →  detector.detect()            │
                    │        →  tracker.update(detections, dt) │
                    │        →  SequentialTargetSM 자동 클릭    │
                    │        →  loot_detector → pico.click()   │
                    └──────────────────────────────────────────┘

특징
────
- **완전 독립 스레드**: FSM 모드(LevelingMode, FieldMode 등) 와 별도 스레드로 실행
- **탐지 FPS / 표시 FPS 분리**: 탐지는 `detection_fps` 속도로, 미리보기 콜백은 별도
- **자동 재시작**: 예외 발생 시 `retry_delay_s` 후 자동 재시도 (최대 `max_retries` 회)
- **외부 정지**: `stop()` 또는 `stop_event`(threading.Event) 로 즉시 중단
- **luot 우선**: loot 아이템이 감지되면 즉시 클릭 후 계속 사냥
- **HP 감시**: HpReader 를 통해 HP 낮으면 물약 사용 (pico.key_tap_name)

사용법
──────
    loop = HuntLoop(
        grab        = capturer.grab,
        detector    = RealtimeTemplateDetector.from_settings(settings),
        tracker     = NearestNeighborTracker.from_settings(settings, pico.click, pico.drag),
        loot_detector = LootDetector.from_settings(settings),
        hp_reader   = HpReader.from_settings(settings),
        pico        = pico,
        settings    = settings,
    )
    loop.start()          # 백그라운드 스레드 시작
    ...
    loop.stop()           # 정지

콜백 연동 (PreviewWindow 등)
────────────────────────────
    loop.on_frame_cb = lambda frame, enemies: preview_win.update_enemies(enemies)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config.settings import Settings
    from core.detection import Detection, RealtimeTemplateDetector
    from core.tracking import Enemy, NearestNeighborTracker
    from hardware.pico import PicoWorker
    from hardware.screen import ScreenCapturer
    from perception.hp import HpReader
    from perception.loot import LootDetector

logger = logging.getLogger("modes.hunt_loop")


class HuntLoop:
    """탐지 → 추적 → 공격 → 루팅 무한 사냥 루프.

    Parameters
    ----------
    grab            : () -> np.ndarray    화면 캡처 함수
    detector        : RealtimeTemplateDetector    몬스터 탐지기
    tracker         : NearestNeighborTracker     추적기 + 순차 공격 SM
    loot_detector   : LootDetector | None        아데나/루팅 탐지기
    hp_reader       : HpReader | None            HP 감시
    pico            : PicoWorker                 클릭/키 입력 장치
    settings        : Settings                  설정 (FPS, HP threshold 등)
    detection_fps   : float                     탐지 루프 목표 FPS (기본 6)
    max_retries     : int                       예외 시 자동 재시도 횟수 (기본 5)
    retry_delay_s   : float                     재시도 전 대기 시간 (기본 2.0)
    on_frame_cb     : (frame, enemies) -> None  매 프레임 후 콜백 (미리보기 연동)
    on_kill_cb      : (kill_count) -> None      적 처치 시 콜백
    """

    def __init__(
        self,
        grab: Callable[[], Optional[np.ndarray]],
        detector: "RealtimeTemplateDetector",
        tracker: "NearestNeighborTracker",
        loot_detector: Optional["LootDetector"] = None,
        hp_reader: Optional["HpReader"] = None,
        pico=None,
        settings=None,
        detection_fps: float = 6.0,
        max_retries: int = 5,
        retry_delay_s: float = 2.0,
        on_frame_cb: Optional[Callable[[np.ndarray, List["Enemy"]], None]] = None,
        on_kill_cb:  Optional[Callable[[int], None]] = None,
    ) -> None:
        self._grab          = grab
        self._detector      = detector
        self._tracker       = tracker
        self._loot          = loot_detector
        self._hp_reader     = hp_reader
        self._pico          = pico
        self._settings      = settings
        self._det_interval  = 1.0 / max(detection_fps, 1.0)
        self._max_retries   = max_retries
        self._retry_delay   = retry_delay_s

        # 콜백
        self.on_frame_cb = on_frame_cb
        self.on_kill_cb  = on_kill_cb

        # HP 관련 설정
        self._key_potion   = "F5"
        self._potion_cd    = 3.0
        self._last_potion  = 0.0
        if settings is not None:
            keys = getattr(settings, "keys", None)
            if keys:
                self._key_potion  = getattr(keys, "potion", "F5")
                self._potion_cd   = getattr(keys, "potion_cooldown_ms", 3000) / 1000.0

        # 루팅 설정
        self._loot_click_delay = 0.05   # 루팅 클릭 후 대기 (초)
        if settings is not None:
            loot_cfg = getattr(settings, "loot", None)
            if loot_cfg:
                self._loot_click_delay = getattr(loot_cfg, "click_delay_ms", 50) / 1000.0

        # 제어
        self._stop_event   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active       = False

        # 통계
        self._kill_count   = 0
        self._total_loot   = 0
        self._last_det_fps = 0.0
        self._last_proc_ms = 0.0
        self._enemies: List["Enemy"] = []

    # ── 공개 API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """백그라운드 스레드로 사냥 루프를 시작한다."""
        if self._thread and self._thread.is_alive():
            logger.warning("HuntLoop 이미 실행 중")
            return
        self._stop_event.clear()
        self._active = True
        self._kill_count = 0
        self._total_loot = 0
        self._thread = threading.Thread(
            target=self._runner,
            name="HuntLoop",
            daemon=True,
        )
        self._thread.start()
        logger.info("HuntLoop 시작 (detection_fps=%.1f)", 1.0 / self._det_interval)

    def stop(self) -> None:
        """루프를 정지하고 스레드가 종료될 때까지 최대 3초 대기한다."""
        self._active = False
        self._stop_event.set()
        if self._tracker:
            self._tracker.set_active(False)
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("HuntLoop 정지 — 킬 %d / 루팅 %d", self._kill_count, self._total_loot)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def kill_count(self) -> int:
        return self._kill_count

    @property
    def enemies(self) -> List["Enemy"]:
        return list(self._enemies)

    @property
    def det_fps(self) -> float:
        return self._last_det_fps

    @property
    def proc_ms(self) -> float:
        return self._last_proc_ms

    def set_active(self, active: bool) -> None:
        """공격 SM 활성 여부 제어 (루팅 중 비활성화 등)."""
        if self._tracker:
            self._tracker.set_active(active)

    # ── 내부 실행 ─────────────────────────────────────────────────────────

    def _runner(self) -> None:
        """재시도 래퍼 — 예외 발생 시 max_retries 회 재시작."""
        retries = 0
        while not self._stop_event.is_set() and retries <= self._max_retries:
            try:
                self._loop()
            except Exception as e:
                if self._stop_event.is_set():
                    break
                retries += 1
                logger.error(
                    "HuntLoop 예외 (시도 %d/%d): %s",
                    retries, self._max_retries, e, exc_info=True,
                )
                if retries <= self._max_retries:
                    logger.info("%.1f초 후 재시작...", self._retry_delay)
                    time.sleep(self._retry_delay)
                else:
                    logger.critical("최대 재시도 초과 — HuntLoop 종료")
                    break
        logger.info("HuntLoop 스레드 종료")

    def _loop(self) -> None:
        """탐지 → 추적 → 공격 → 루팅 핵심 루프."""
        prev_target_id   = None
        last_det_time    = time.monotonic()
        det_fps_smooth   = 0.0

        # 트래커 활성화
        if self._tracker:
            self._tracker.set_active(True)

        logger.info("=== HuntLoop 핵심 루프 진입 ===")

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            # ── 1. 화면 캡처 ────────────────────────────────────────────
            frame = self._grab()
            if frame is None or frame.size == 0:
                time.sleep(0.05)
                continue

            # ── 2. 템플릿 매칭 탐지 ─────────────────────────────────────
            t0 = time.monotonic()
            try:
                detections = self._detector.detect(frame)
            except Exception as e:
                logger.warning("탐지 오류: %s", e)
                detections = []

            # ── 3. 트래커 업데이트 → SequentialTargetSM 자동 클릭 ────────
            dt = max(time.monotonic() - last_det_time, 1e-4)
            last_det_time = time.monotonic()

            try:
                enemies = self._tracker.update(detections, dt)
            except Exception as e:
                logger.warning("트래커 업데이트 오류: %s", e)
                enemies = []

            self._enemies = enemies

            # ── 4. 킬 카운트 갱신 ───────────────────────────────────────
            cur_target_id = self._tracker.current_target_id if self._tracker else None
            if (
                prev_target_id is not None
                and cur_target_id != prev_target_id
                and self._tracker.target_state.name in ("COOLDOWN", "IDLE")
            ):
                self._kill_count += 1
                logger.info(
                    "[HuntLoop] 적 처치 확인 #%d → 누적 %d킬",
                    prev_target_id, self._kill_count,
                )
                if self.on_kill_cb:
                    try:
                        self.on_kill_cb(self._kill_count)
                    except Exception:
                        pass
            prev_target_id = cur_target_id

            # ── 5. 루팅 (아데나 줍기) ────────────────────────────────────
            if self._loot is not None:
                try:
                    self._do_loot(frame)
                except Exception as e:
                    logger.warning("루팅 오류: %s", e)

            # ── 6. HP 감시 ────────────────────────────────────────────────
            self._check_hp(frame)

            # ── 7. 통계 / 콜백 ───────────────────────────────────────────
            proc_elapsed = time.monotonic() - t0
            self._last_proc_ms = proc_elapsed * 1000.0
            elapsed_since_last = time.monotonic() - loop_start
            if elapsed_since_last > 0:
                det_fps_smooth = 0.9 * det_fps_smooth + 0.1 * (1.0 / elapsed_since_last)
            self._last_det_fps = det_fps_smooth

            if self.on_frame_cb:
                try:
                    self.on_frame_cb(frame, enemies)
                except Exception:
                    pass

            # ── 8. FPS 제한 대기 ─────────────────────────────────────────
            sleep_t = self._det_interval - (time.monotonic() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _do_loot(self, frame: np.ndarray) -> None:
        """아데나/아이템 탐지 후 클릭 (루팅)."""
        items = self._loot.find(frame)
        if not items:
            return

        # 가장 가까운 아이템부터 줍기
        ref_x = frame.shape[1] // 2
        ref_y = frame.shape[0] // 2
        item = self._loot.find_nearest(frame, ref_x, ref_y)
        if item is None:
            return

        logger.info("[HuntLoop] 루팅: cx=%d cy=%d", item.cx, item.cy)
        if self._pico:
            self._pico.click(item.cx, item.cy, pulse_ms=50)
        self._total_loot += 1
        self._loot.invalidate()
        time.sleep(self._loot_click_delay)

    def _check_hp(self, frame: np.ndarray) -> None:
        """HP 낮으면 물약 사용."""
        if self._hp_reader is None or self._pico is None:
            return
        now = time.monotonic()
        if now - self._last_potion < self._potion_cd:
            return
        try:
            ratio = self._hp_reader.read(frame)
            if ratio is not None and self._hp_reader.is_low():
                self._pico.key_tap_name(self._key_potion, hold_ms=50)
                self._last_potion = now
                logger.info("[HuntLoop] HP 낮음 → %s 물약 사용", self._key_potion)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# 팩토리 함수
# ════════════════════════════════════════════════════════════════════════════

def build_hunt_loop(
    settings,
    pico,
    grab_fn: Callable[[], Optional[np.ndarray]],
    on_frame_cb=None,
    on_kill_cb=None,
    base_dir: str = ".",
) -> HuntLoop:
    """Settings 객체로부터 HuntLoop 를 완전히 구성해 반환한다.

    모든 컴포넌트(detector, tracker, loot, hp)를 Settings 에서 생성하므로
    호출 측에서 개별 객체를 직접 생성할 필요가 없다.

    Parameters
    ----------
    settings    : Settings
    pico        : PicoWorker | NullPicoWorker
    grab_fn     : () -> np.ndarray
    on_frame_cb : (frame, enemies) -> None  미리보기 연동 콜백
    on_kill_cb  : (kill_count) -> None      킬 카운트 콜백
    base_dir    : str  config.json 기준 디렉토리 (templates_dir 해석에 사용)
    """
    # ── 탐지기 ──────────────────────────────────────────────────────────
    from core.detection import RealtimeTemplateDetector
    detector = RealtimeTemplateDetector.from_settings(settings, base_dir=base_dir)

    logger.info(
        "탐지기 로드: templates=%d개, reject=%d개",
        len(detector._prepared),
        len(detector._reject),
    )

    if len(detector._prepared) == 0:
        logger.warning(
            "⚠ 템플릿 이미지가 0개입니다! config/templates/ 에 PNG를 넣어주세요."
        )

    # ── 트래커 ──────────────────────────────────────────────────────────
    from core.tracking import NearestNeighborTracker
    tracker = NearestNeighborTracker.from_settings(
        settings,
        pico_click_callback=lambda x, y: pico.click(x, y) if pico else None,
        pico_drag_callback=(
            lambda fx, fy, tx, ty: pico.drag(fx, fy, tx, ty)
            if pico and getattr(pico, "drag", None) else None
        ),
    )

    # ── 루팅 탐지기 ─────────────────────────────────────────────────────
    loot = None
    try:
        from perception.loot import LootDetector
        loot = LootDetector.from_settings(settings)
        logger.info("LootDetector 초기화 완료")
    except Exception as e:
        logger.warning("LootDetector 초기화 실패: %s", e)

    # ── HP 리더 ─────────────────────────────────────────────────────────
    hp_reader = None
    try:
        from perception.hp import HpReader
        hp_reader = HpReader.from_settings(settings)
        logger.info("HpReader 초기화 완료")
    except Exception as e:
        logger.warning("HpReader 초기화 실패: %s", e)

    # ── detection FPS ───────────────────────────────────────────────────
    det_fps = getattr(getattr(settings, "detection", None), "fps", 6.0)

    return HuntLoop(
        grab          = grab_fn,
        detector      = detector,
        tracker       = tracker,
        loot_detector = loot,
        hp_reader     = hp_reader,
        pico          = pico,
        settings      = settings,
        detection_fps = det_fps,
        on_frame_cb   = on_frame_cb,
        on_kill_cb    = on_kill_cb,
    )
