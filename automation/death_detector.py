"""타겟 몬스터 사망/소실 감지 모듈 (YOLO 기반).

monster-adena-detector 프로젝트의 death_detector.py 를
flslwl_master 스타일로 이식.

판정 기준:
    1. 현재 프레임 YOLO 탐지 목록에서 이전 타겟 위치와 IoU >= thresh 인 박스가 없음
    2. 위 상태가 death_timeout_sec 이상 지속 → 사망 확정

사용법:
    dd = YoloDeathDetector(timeout_sec=1.5, iou_thresh=0.3)
    dd.reset()                               # 새 타겟 지정 시 초기화
    is_dead = dd.update(yolo_detections, current_target)
"""

import logging
import time
from typing import List, Optional

logger = logging.getLogger("death_detector")


# ── IoU 유틸 ─────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    """두 Detection(또는 YoloDetection)의 IoU 계산.

    tlbr 프로퍼티 또는 (x, y, width, height) 속성을 사용.
    """
    if hasattr(a, "tlbr"):
        ax1, ay1, ax2, ay2 = a.tlbr
    else:
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.width, a.y + a.height

    if hasattr(b, "tlbr"):
        bx1, by1, bx2, by2 = b.tlbr
    else:
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.width, b.y + b.height

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union  = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def find_matching_yolo(detections: list, target, iou_thresh: float = 0.3):
    """YOLO 탐지 목록에서 target 과 IoU 가장 높은 것 반환.

    thresh 미만이면 None (타겟 소실/사망 판정).
    """
    if not detections or target is None:
        return None
    best = max(detections, key=lambda d: _iou(d, target))
    if _iou(best, target) >= iou_thresh:
        return best
    return None


# ── DeathDetector ─────────────────────────────────────────────────────────────

class YoloDeathDetector:
    """YOLO 탐지 기반 몬스터 사망/소실 감지기.

    매 프레임 update() 를 호출하면 타겟이 소실된 상태가
    timeout_sec 이상 지속될 때 True(사망 확정)를 반환한다.
    """

    def __init__(self, timeout_sec: float = 1.5, iou_thresh: float = 0.3):
        """
        Args:
            timeout_sec : 소실 지속 시간 임계값 (초). 이 이상이면 사망 확정.
            iou_thresh  : IoU 임계값. 이 미만이면 소실로 판정.
        """
        self._timeout    = timeout_sec
        self._iou_thresh = iou_thresh
        self._miss_since: Optional[float] = None  # 소실 시작 시각

    def reset(self) -> None:
        """새 타겟 지정 시 반드시 호출. 소실 타이머를 초기화한다."""
        self._miss_since = None

    def update(self, yolo_detections: list, target) -> bool:
        """매 프레임 호출.

        Args:
            yolo_detections : YoloDetection 리스트 (class_id=0 monster 만 넘겨도 됨)
            target          : 현재 추적 중인 YoloDetection (None이면 즉시 True)

        Returns:
            True  → 사망/소실 확정 (새 타겟 선택 필요)
            False → 타겟 살아있음
        """
        if target is None:
            return True

        matched = find_matching_yolo(yolo_detections, target, self._iou_thresh)

        if matched is not None:
            # 탐지됨 → 살아있음
            self._miss_since = None
            return False
        else:
            # 소실 → 타이머 시작
            now = time.time()
            if self._miss_since is None:
                self._miss_since = now

            elapsed = now - self._miss_since
            if elapsed >= self._timeout:
                logger.debug(
                    f"[YoloDeathDetector] 사망 확정 "
                    f"(소실 {elapsed:.1f}s >= {self._timeout}s)"
                )
                self._miss_since = None
                return True  # 사망 확정
            return False

    @property
    def miss_elapsed(self) -> float:
        """현재 소실 중이면 경과 시간(초), 아니면 0."""
        if self._miss_since is None:
            return 0.0
        return time.time() - self._miss_since
