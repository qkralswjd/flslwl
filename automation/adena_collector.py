"""YOLO 기반 아데나 자동 수집 모듈.

monster-adena-detector 프로젝트의 adena_collector.py 를
flslwl_master/pico_worker 연동 방식으로 이식.

흐름:
    1. 몬스터 사망 확정 시 → record_kill(cx, cy) 호출
       - 사망 위치와 탐색 반경을 기억
    2. 매 프레임 → check_and_collect(yolo_detections, pico_worker) 호출
       - 기억된 위치 주변에서 adena(class_id==1) 탐색
       - 발견 시 pico_worker.click() 으로 줍기
       - 없으면 collect_timeout_sec 후 포기

설정값 예시 (config_automation.json → "yolo_adena"):
    enabled             : bool  - ON/OFF (기본 true)
    search_radius       : int   - 사망 위치 기준 탐색 반경 px (기본 120)
    collect_timeout_sec : float - 탐색 대기 최대 시간 (기본 3.0)
    click_hold_ms       : int   - 줍기 클릭 pulse ms (기본 50)
    max_picks           : int   - 한 자리 최대 줍기 횟수 (기본 5)
"""

import logging
import math
import time
from typing import List, Optional

logger = logging.getLogger("adena_collector")


def _dist(ax: int, ay: int, bx: int, by: int) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


class YoloAdenaCollector:
    """YOLO adena(class_id==1) 탐지 기반 아데나 자동 수집기.

    SequentialTargetStateMachine 이 COOLDOWN(사망 확정) 전환 시
    record_kill() 을 호출하고, 매 프레임 check_and_collect() 를 호출한다.

    pico_worker 는 flslwl_master 의 PicoSerialWorker 인스턴스.
    click() 메서드만 사용한다.
    """

    def __init__(
        self,
        enabled: bool           = True,
        search_radius: int      = 120,
        collect_timeout_sec: float = 3.0,
        click_hold_ms: int      = 50,
        max_picks: int          = 5,
    ):
        """
        Args:
            enabled             : False이면 record_kill/check_and_collect 모두 무시
            search_radius       : 사망 위치 기준 탐색 반경 (픽셀)
            collect_timeout_sec : 아데나 탐색 최대 대기 시간 (초)
            click_hold_ms       : 아데나 클릭 pulse ms (PicoSerialWorker.click pulse)
            max_picks           : 한 사망 자리에서 최대 줍기 횟수
        """
        self._enabled         = enabled
        self._search_radius   = search_radius
        self._collect_timeout = collect_timeout_sec
        self._click_hold_ms   = click_hold_ms
        self._max_picks       = max_picks

        # 현재 수집 대기 상태
        self._kill_x: Optional[int]   = None
        self._kill_y: Optional[int]   = None
        self._kill_time: Optional[float] = None
        self._picks_done: int          = 0
        self._collecting: bool         = False

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def record_kill(self, cx: int, cy: int) -> None:
        """몬스터 사망 확정 시 호출.

        SequentialTargetStateMachine._fire_click() / COOLDOWN 진입 시점에
        tracker 또는 main.py 에서 호출한다.

        Args:
            cx, cy : 사망한 몬스터의 ROI 기준 중심 좌표
                     (나중에 pico_worker.click() 에 그대로 전달됨 — 스크린 좌표로 변환 필요 시
                      호출 측에서 변환 후 전달할 것)
        """
        if not self._enabled:
            return

        self._kill_x    = cx
        self._kill_y    = cy
        self._kill_time  = time.time()
        self._picks_done = 0
        self._collecting = True

        logger.info(
            f"[YoloAdenaCollector] 사망 위치 기억: ({cx},{cy})  "
            f"반경={self._search_radius}px  "
            f"최대대기={self._collect_timeout}s"
        )

    def check_and_collect(
        self,
        yolo_detections: list,
        pico_worker,
        roi_offset: tuple = (0, 0),
        capture_offset: tuple = (0, 0),
    ) -> bool:
        """매 프레임 호출. 아데나가 탐지되면 pico_worker.click() 으로 줍는다.

        Args:
            yolo_detections : YoloDetection 리스트 (class_id=1 adena 포함)
            pico_worker     : PicoSerialWorker (click(x,y,pulse_ms) 사용)
            roi_offset      : (ox, oy) — ROI 오프셋 (ROI 좌표 → 캡처 좌표 변환)
            capture_offset  : (ox, oy) — 캡처 오프셋 (캡처 좌표 → 화면 절대좌표 변환)

        Returns:
            True  : 수집 완료 또는 타임아웃 (더 이상 대기 불필요)
            False : 아직 수집 중 또는 비활성
        """
        if not self._enabled or not self._collecting:
            return False

        now     = time.time()
        elapsed = now - self._kill_time

        # ── 타임아웃: 아데나 없는 몬스터 ────────────────────────────────
        if elapsed > self._collect_timeout:
            logger.info(
                f"[YoloAdenaCollector] 타임아웃({self._collect_timeout}s) "
                "→ 아데나 없음, 수집 종료"
            )
            self._reset()
            return True

        # ── adena(class_id==1) 필터 ──────────────────────────────────────
        adenas = [d for d in yolo_detections if d.class_id == 1]
        if not adenas:
            return False  # 아직 아데나 탐지 안 됨

        # ── 사망 위치 반경 내 후보만 ──────────────────────────────────────
        nearby = [
            d for d in adenas
            if _dist(d.cx, d.cy, self._kill_x, self._kill_y) <= self._search_radius
        ]
        if not nearby:
            logger.debug(
                f"[YoloAdenaCollector] adena {len(adenas)}개 탐지됐으나 "
                f"반경({self._search_radius}px) 밖 → 대기 중"
            )
            return False

        # ── 가장 가까운 아데나 줍기 ──────────────────────────────────────
        target_adena = min(
            nearby,
            key=lambda d: _dist(d.cx, d.cy, self._kill_x, self._kill_y)
        )

        # ROI 좌표 → 화면 절대좌표 변환
        screen_x = target_adena.cx + roi_offset[0] + capture_offset[0]
        screen_y = target_adena.cy + roi_offset[1] + capture_offset[1]

        logger.info(
            f"[YoloAdenaCollector] 아데나 발견! "
            f"ROI({target_adena.cx},{target_adena.cy}) "
            f"→ screen({screen_x},{screen_y})  "
            f"conf={target_adena.confidence:.2f}"
        )

        pico_worker.click(screen_x, screen_y, self._click_hold_ms)
        self._picks_done += 1

        logger.info(
            f"[YoloAdenaCollector] 줍기 완료 "
            f"({self._picks_done}/{self._max_picks})"
        )

        # ── 최대 횟수 도달 ───────────────────────────────────────────────
        if self._picks_done >= self._max_picks:
            logger.info("[YoloAdenaCollector] 최대 줍기 횟수 → 수집 종료")
            self._reset()
            return True

        return False

    def invalidate(self) -> None:
        """수집 상태를 강제 종료한다. 다음 사냥 시작 전 호출하면 안전."""
        if self._collecting:
            logger.debug("[YoloAdenaCollector] 강제 종료 (invalidate)")
        self._reset()

    # ── 프로퍼티 ─────────────────────────────────────────────────────────────

    @property
    def is_collecting(self) -> bool:
        """현재 아데나 수집 대기 중인지."""
        return self._collecting

    @property
    def elapsed(self) -> float:
        """수집 대기 경과 시간(초). 대기 중 아닐 때는 0."""
        if self._kill_time is None:
            return 0.0
        return time.time() - self._kill_time

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._kill_x    = None
        self._kill_y    = None
        self._kill_time  = None
        self._picks_done = 0
        self._collecting = False
