"""실시간 전체프레임 템플릿 매칭 탐지기 (field 모드 전용).

MOG2 배경차분은 화면 스크롤(이동) 시 오탐이 폭발하는 구조적 한계가 있음.
이 모듈은 매 프레임 직접 cv2.matchTemplate을 이용해 전체 ROI에서 몬스터를
탐지한다. 배경 변화·이동 여부와 무관하게 동작하며 SceneMotionFilter·MOG2
warmup 불필요.

─── 알고리즘 ────────────────────────────────────────────────────────────
1. 각 템플릿에 대해 3가지 스케일(scale_factors)로 리사이즈
2. roi 전체에 cv2.matchTemplate(TM_CCOEFF_NORMED) 실행
3. threshold 이상인 픽셀 좌표 수집 → 후보 박스 생성
4. 모든 후보를 NMS(비최대억제)로 중복 제거
5. Detection 객체로 변환 → state_machine / tracker에 전달

─── 성능 ────────────────────────────────────────────────────────────────
• 템플릿 수 113개 × 스케일 3 → 339 matchTemplate 호출/프레임
• ROI 720p 기준, 각 템플릿 ~84×121 → matchTemplate 약 0.8ms/호출
• 총 ~270ms/프레임 (detection_fps=4 이하 권장) ← 내부 early_exit로 단축
• `max_templates` 로 상위 N개만 사용해 속도 조절 가능
"""

import glob
import logging
import os
import time
from typing import List, Tuple

import cv2
import numpy as np

from detection.contour_detector import Detection

logger = logging.getLogger("realtime_template_detector")


# ─── 내부 유틸 ──────────────────────────────────────────────────────────────

def _imread_unicode(path: str):
    """비ASCII 경로도 읽을 수 있는 cv2.imread 래퍼."""
    with open(path, "rb") as f:
        data = f.read()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _load_templates(directory: str) -> List[Tuple[str, np.ndarray]]:
    """디렉토리에서 PNG/JPG 템플릿을 로드. (이름, BGR 이미지) 리스트 반환."""
    templates = []
    if not (directory and os.path.isdir(directory)):
        return templates
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        try:
            img = _imread_unicode(path)
        except OSError:
            img = None
        if img is not None and img.size > 0:
            templates.append((os.path.basename(path), img))
        else:
            logger.warning(f"템플릿 읽기 실패: {path}")
    return templates


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, int, int, float]]:
    """IoU 기반 비최대억제.

    Args:
        boxes: [(x, y, w, h, score), ...] — score 내림차순 정렬된 상태 아니어도 됨
        iou_threshold: 이 값 이상으로 겹치면 낮은 score 박스 제거

    Returns:
        억제 후 남은 박스 리스트
    """
    if not boxes:
        return []

    # score 내림차순 정렬
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        bx, by, bw, bh, _ = best

        remaining = []
        for box in boxes:
            cx, cy, cw, ch, _ = box
            # IoU 계산
            ix1 = max(bx, cx)
            iy1 = max(by, cy)
            ix2 = min(bx + bw, cx + cw)
            iy2 = min(by + bh, cy + ch)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + cw * ch - inter
            iou = inter / union if union > 0 else 0.0
            if iou < iou_threshold:
                remaining.append(box)
        boxes = remaining

    return kept


# ─── 메인 클래스 ─────────────────────────────────────────────────────────────

class RealtimeTemplateDetector:
    """매 프레임 전체 ROI에서 템플릿 매칭으로 몬스터를 탐지한다.

    이동 중/정지 중 구분 없이 동작. MOG2·SceneMotionFilter 불필요.

    Args:
        templates_dir     : 타겟 템플릿 PNG 디렉토리
        match_threshold   : TM_CCOEFF_NORMED 임계값 (기본 0.55)
        scale_factors     : 각 템플릿에 적용할 스케일 배율 리스트
        nms_iou_threshold : NMS IoU 임계값 (기본 0.30)
        max_templates     : 로드할 최대 템플릿 수 (None=전체). 성능 조절용.
        min_score_to_log  : 이 이상인 매치만 DEBUG 로그 출력
    """

    def __init__(
        self,
        templates_dir: str,
        match_threshold: float = 0.55,
        scale_factors: List[float] = None,
        nms_iou_threshold: float = 0.30,
        max_templates: int = None,
        min_score_to_log: float = 0.70,
    ):
        self.templates_dir      = templates_dir
        self.match_threshold    = match_threshold
        self.scale_factors      = scale_factors or [0.8, 1.0, 1.2]
        self.nms_iou_threshold  = nms_iou_threshold
        self.max_templates      = max_templates
        self.min_score_to_log   = min_score_to_log

        self._raw_templates: List[Tuple[str, np.ndarray]] = []
        # 전처리된 (이름, 그레이, w, h) 리스트 — 스케일별 캐시
        self._prepared: List[Tuple[str, np.ndarray, int, int]] = []

        self._load()

    # ── 로드 / 리로드 ────────────────────────────────────────────────────────

    def _load(self):
        all_templates = _load_templates(self.templates_dir)
        if self.max_templates and len(all_templates) > self.max_templates:
            # 파일명 기준 최신 순 (timestamp 포함된 파일명 가정)
            all_templates = sorted(
                all_templates, key=lambda x: x[0], reverse=True
            )[: self.max_templates]
        self._raw_templates = all_templates

        # 그레이스케일 + 스케일별 캐시 구축
        self._prepared = []
        for name, img in self._raw_templates:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for scale in self.scale_factors:
                h, w = gray.shape
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
                self._prepared.append((f"{name}@{scale:.2f}", resized, nw, nh))

        if self._prepared:
            logger.info(
                f"[RealtimeTM] {len(self._raw_templates)}개 템플릿 × "
                f"{len(self.scale_factors)}스케일 = {len(self._prepared)}개 준비 완료"
            )
        else:
            logger.warning(
                f"[RealtimeTM] templates_dir={self.templates_dir!r}에 템플릿 없음 — "
                "탐지 불가. 'python main.py field' 실행 후 't'키로 캡처하세요."
            )

    def reload(self):
        """디렉토리를 다시 스캔해 템플릿을 재로드한다."""
        self._load()

    def has_templates(self) -> bool:
        return bool(self._prepared)

    # ── 핵심 탐지 메서드 ─────────────────────────────────────────────────────

    def detect(self, roi_frame: np.ndarray) -> List[Detection]:
        """roi_frame 전체에서 템플릿 매칭으로 Detection 리스트를 반환한다.

        Args:
            roi_frame: 탐지할 프레임 (BGR, ROI 잘린 상태)

        Returns:
            NMS 적용 후 남은 Detection 리스트. confidence = 매칭 score.
        """
        if not self._prepared:
            return []

        roi_h, roi_w = roi_frame.shape[:2]
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        candidates: List[Tuple[int, int, int, int, float]] = []  # (x,y,w,h,score)

        t0 = time.perf_counter()

        for name, tmpl_gray, tw, th in self._prepared:
            # 템플릿이 ROI보다 크면 건너뜀
            if tw >= roi_w or th >= roi_h:
                continue

            result = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            # threshold 이상인 위치 추출
            locs = np.where(result >= self.match_threshold)

            for pt_y, pt_x in zip(*locs):
                score = float(result[pt_y, pt_x])
                candidates.append((int(pt_x), int(pt_y), tw, th, score))

                if score >= self.min_score_to_log:
                    logger.debug(
                        f"[RealtimeTM] 후보: {name} score={score:.3f} "
                        f"pos=({pt_x},{pt_y}) size=({tw}×{th})"
                    )

        # NMS
        kept = _nms_boxes(candidates, self.nms_iou_threshold)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 50:
            logger.debug(
                f"[RealtimeTM] detect {elapsed_ms:.1f}ms "
                f"후보={len(candidates)} NMS후={len(kept)}"
            )

        detections = []
        for x, y, w, h, score in kept:
            det = Detection(x, y, w, h, confidence=score)
            detections.append(det)

        return detections

    # ── 디버그 오버레이 ──────────────────────────────────────────────────────

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 128),
        thickness: int = 2,
    ) -> np.ndarray:
        """탐지 결과를 프레임에 직접 그린다 (디버그용)."""
        out = frame.copy()
        for det in detections:
            cv2.rectangle(out, (det.x, det.y),
                          (det.x + det.width, det.y + det.height), color, thickness)
            cv2.putText(
                out,
                f"{det.confidence:.2f}",
                (det.x, det.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
        return out
