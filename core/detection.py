"""core/detection.py — Detection 데이터 + 실시간 템플릿 매칭 탐지기.

기존 detection/contour_detector.py + detection/realtime_template_detector.py 를
단일 모듈로 통합한다.

개선점 (기존 대비):
  - Detection 클래스를 이 모듈에 직접 정의 (외부 contour_detector 의존 제거)
  - reject_templates 지원: 탐지된 영역을 reject 템플릿으로 2차 검증 후 억제
  - from_settings() 팩토리 추가
  - paths.py 없이 절대경로 유틸(_abs_path) 내장
  - max_templates=None 이면 전체 사용 (파일명 내림차순 정렬)
  - detect() 반환 타입 List[Detection] 명시

알고리즘 (변경 없음):
  1. 각 템플릿 × scale_factors 로 리사이즈 (캐시)
  2. ROI 전체에 cv2.matchTemplate(TM_CCOEFF_NORMED)
  3. threshold 이상 픽셀 → 후보 박스
  4. NMS(IoU 기반) → 중복 제거
  5. Detection 리스트 반환
"""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger("core.detection")


# ════════════════════════════════════════════════════════════════════════════
# Detection 데이터 클래스
# ════════════════════════════════════════════════════════════════════════════

class Detection:
    """단일 프레임 탐지 결과 (ID 할당 전).

    기존 contour_detector.Detection 과 동일한 인터페이스.
    __slots__ 로 메모리 절약.
    """

    __slots__ = ("x", "y", "width", "height", "center_x", "center_y", "area", "confidence")

    def __init__(self, x: int, y: int, w: int, h: int, confidence: float = 1.0) -> None:
        self.x          = x
        self.y          = y
        self.width      = w
        self.height     = h
        self.center_x   = x + w // 2
        self.center_y   = y + h // 2
        self.area       = w * h
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"Detection(x={self.x}, y={self.y}, w={self.width}, h={self.height}, "
            f"cx={self.center_x}, cy={self.center_y}, conf={self.confidence:.3f})"
        )


# ════════════════════════════════════════════════════════════════════════════
# 내부 유틸
# ════════════════════════════════════════════════════════════════════════════

def _imread_unicode(path: str) -> Optional[np.ndarray]:
    """비ASCII 경로도 읽을 수 있는 cv2.imread 래퍼."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img if img is not None and img.size > 0 else None
    except OSError:
        return None


def _load_templates(directory: str) -> List[Tuple[str, np.ndarray]]:
    """디렉토리에서 PNG/JPG 템플릿을 로드한다.

    Returns:
        [(파일명, BGR 이미지), ...] 리스트
    """
    templates: List[Tuple[str, np.ndarray]] = []
    if not (directory and os.path.isdir(directory)):
        return templates
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        img = _imread_unicode(path)
        if img is not None:
            templates.append((os.path.basename(path), img))
        else:
            logger.warning("템플릿 읽기 실패: %s", path)
    return templates


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, int, int, float]]:
    """IoU 기반 비최대억제(NMS).

    Args:
        boxes: [(x, y, w, h, score), ...] — 정렬 불필요
        iou_threshold: 이 값 이상으로 겹치면 낮은 score 박스 제거

    Returns:
        억제 후 남은 박스 리스트 (score 내림차순)
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: List[Tuple[int, int, int, int, float]] = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        bx, by, bw, bh, _ = best

        remaining = []
        for box in boxes:
            cx, cy, cw, ch, _ = box
            ix1 = max(bx, cx);  iy1 = max(by, cy)
            ix2 = min(bx + bw, cx + cw); iy2 = min(by + bh, cy + ch)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + cw * ch - inter
            iou = inter / union if union > 0 else 0.0
            if iou < iou_threshold:
                remaining.append(box)
        boxes = remaining

    return kept


def _abs_path(base_dir: str, rel_or_abs: str) -> str:
    """상대경로를 base_dir 기준 절대경로로 변환. 이미 절대경로면 그대로."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(base_dir, rel_or_abs))


# ════════════════════════════════════════════════════════════════════════════
# RealtimeTemplateDetector
# ════════════════════════════════════════════════════════════════════════════

class RealtimeTemplateDetector:
    """매 프레임 전체 ROI에서 템플릿 매칭으로 몬스터를 탐지한다.

    이동·정지 구분 없이 동작. MOG2 / SceneMotionFilter 불필요.

    기존 대비 개선:
      - from_settings(settings, base_dir) 팩토리
      - reject_templates_dir 지원: 탐지 후보를 reject 패턴으로 억제
      - reload() 시 reject 템플릿도 재로드

    Args:
        templates_dir        : 타겟 템플릿 PNG 디렉토리
        reject_templates_dir : 거부 템플릿 디렉토리 (None=사용 안 함)
        match_threshold      : TM_CCOEFF_NORMED 임계값
        reject_threshold     : reject 패턴 매칭 임계값 (이상이면 후보 제거)
        scale_factors        : 각 템플릿에 적용할 스케일 배율 리스트
        nms_iou_threshold    : NMS IoU 임계값
        max_templates        : 로드할 최대 타겟 템플릿 수 (None=전체)
        min_score_to_log     : 이 이상인 매치만 DEBUG 로그 출력
    """

    def __init__(
        self,
        templates_dir: str,
        reject_templates_dir: Optional[str] = None,
        match_threshold: float = 0.55,
        reject_threshold: float = 0.60,
        scale_factors: Optional[List[float]] = None,
        nms_iou_threshold: float = 0.30,
        max_templates: Optional[int] = None,
        min_score_to_log: float = 0.70,
    ) -> None:
        self.templates_dir        = templates_dir
        self.reject_templates_dir = reject_templates_dir
        self.match_threshold      = match_threshold
        self.reject_threshold     = reject_threshold
        self.scale_factors        = scale_factors or [0.8, 1.0, 1.2]
        self.nms_iou_threshold    = nms_iou_threshold
        self.max_templates        = max_templates
        self.min_score_to_log     = min_score_to_log

        # 전처리된 (이름, 그레이스케일, w, h) — 스케일별 플래튼
        self._prepared: List[Tuple[str, np.ndarray, int, int]] = []
        # reject 패턴 (이름, 그레이스케일, w, h)
        self._reject: List[Tuple[str, np.ndarray, int, int]] = []

        self._load()

    # ── 팩토리 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        base_dir: str = "",
    ) -> "RealtimeTemplateDetector":
        """Settings 객체로부터 탐지기를 생성한다.

        Args:
            settings : 통합 설정 객체
            base_dir : 상대 경로의 기준 디렉토리 (보통 프로젝트 루트)
        """
        d = settings.detection
        tmpl_dir    = _abs_path(base_dir, d.templates_dir)
        reject_dir: Optional[str] = None
        if d.reject_templates_dir:
            reject_dir = _abs_path(base_dir, d.reject_templates_dir)
        return cls(
            templates_dir        = tmpl_dir,
            reject_templates_dir = reject_dir,
            match_threshold      = d.match_threshold,
            scale_factors        = list(d.scale_factors),
            max_templates        = d.max_templates if d.max_templates > 0 else None,
        )

    # ── 로드 / 리로드 ───────────────────────────────────────────────────────

    def _build_prepared(
        self,
        raw: List[Tuple[str, np.ndarray]],
    ) -> List[Tuple[str, np.ndarray, int, int]]:
        """원본 BGR 이미지 리스트 → 그레이+스케일별 플래튼 리스트."""
        result: List[Tuple[str, np.ndarray, int, int]] = []
        for name, img in raw:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for scale in self.scale_factors:
                h, w = gray.shape
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
                result.append((f"{name}@{scale:.2f}", resized, nw, nh))
        return result

    def _load(self) -> None:
        # ── 타겟 템플릿 ────────────────────────────────────────────────
        all_templates = _load_templates(self.templates_dir)
        if self.max_templates and len(all_templates) > self.max_templates:
            # 파일명 내림차순 (타임스탬프 포함 파일명 기준 최신 우선)
            all_templates = sorted(
                all_templates, key=lambda x: x[0], reverse=True
            )[: self.max_templates]

        self._prepared = self._build_prepared(all_templates)

        if self._prepared:
            logger.info(
                "[RealtimeTM] %d개 템플릿 × %d스케일 = %d개 준비",
                len(all_templates), len(self.scale_factors), len(self._prepared),
            )
        else:
            logger.warning(
                "[RealtimeTM] templates_dir=%r 에 템플릿 없음 — 탐지 불가.",
                self.templates_dir,
            )

        # ── reject 템플릿 ──────────────────────────────────────────────
        if self.reject_templates_dir:
            reject_raw = _load_templates(self.reject_templates_dir)
            # reject 는 scale 1.0 고정 (빠른 검증용)
            self._reject = []
            for name, img in reject_raw:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                self._reject.append((name, gray, w, h))
            if self._reject:
                logger.info("[RealtimeTM] reject 템플릿 %d개 로드", len(self._reject))
        else:
            self._reject = []

    def reload(self) -> None:
        """디렉토리를 다시 스캔해 템플릿을 재로드한다."""
        self._load()

    def has_templates(self) -> bool:
        return bool(self._prepared)

    # ── 핵심 탐지 ──────────────────────────────────────────────────────────

    def detect(self, roi_frame: np.ndarray) -> List[Detection]:
        """roi_frame 전체에서 템플릿 매칭으로 Detection 리스트를 반환한다.

        Args:
            roi_frame: 탐지할 프레임 (BGR, ROI 잘린 상태)

        Returns:
            NMS + reject 억제 후 남은 Detection 리스트.
            confidence = TM_CCOEFF_NORMED 매칭 score.
        """
        if not self._prepared:
            return []

        roi_h, roi_w = roi_frame.shape[:2]
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        candidates: List[Tuple[int, int, int, int, float]] = []

        t0 = time.perf_counter()

        for name, tmpl_gray, tw, th in self._prepared:
            if tw >= roi_w or th >= roi_h:
                continue
            result = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            locs = np.where(result >= self.match_threshold)
            for pt_y, pt_x in zip(*locs):
                score = float(result[pt_y, pt_x])
                candidates.append((int(pt_x), int(pt_y), tw, th, score))
                if score >= self.min_score_to_log:
                    logger.debug(
                        "[RealtimeTM] 후보: %s score=%.3f pos=(%d,%d) size=(%d×%d)",
                        name, score, pt_x, pt_y, tw, th,
                    )

        # NMS
        kept = _nms_boxes(candidates, self.nms_iou_threshold)

        # reject 억제
        if self._reject and kept:
            kept = self._apply_reject(roi_gray, roi_w, roi_h, kept)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 50:
            logger.debug(
                "[RealtimeTM] detect %.1fms 후보=%d NMS후=%d",
                elapsed_ms, len(candidates), len(kept),
            )

        return [Detection(x, y, w, h, confidence=score) for x, y, w, h, score in kept]

    def _apply_reject(
        self,
        roi_gray: np.ndarray,
        roi_w: int,
        roi_h: int,
        boxes: List[Tuple[int, int, int, int, float]],
    ) -> List[Tuple[int, int, int, int, float]]:
        """각 후보 박스 영역에 reject 패턴을 매칭해 일치하면 제거한다."""
        result = []
        for bx, by, bw, bh, score in boxes:
            # 후보 영역 크롭
            x1 = max(0, bx); y1 = max(0, by)
            x2 = min(roi_w, bx + bw); y2 = min(roi_h, by + bh)
            crop = roi_gray[y1:y2, x1:x2]
            if crop.size == 0:
                result.append((bx, by, bw, bh, score))
                continue

            rejected = False
            for rname, rtmpl, rw, rh in self._reject:
                if rw > crop.shape[1] or rh > crop.shape[0]:
                    continue
                rv = cv2.matchTemplate(crop, rtmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(rv)
                if max_val >= self.reject_threshold:
                    logger.debug(
                        "[RealtimeTM] reject 억제: %s score=%.3f → 제거",
                        rname, max_val,
                    )
                    rejected = True
                    break
            if not rejected:
                result.append((bx, by, bw, bh, score))
        return result

    # ── 디버그 오버레이 ────────────────────────────────────────────────────

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
            cv2.rectangle(
                out,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color, thickness,
            )
            cv2.putText(
                out,
                f"{det.confidence:.2f}",
                (det.x, det.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
        return out
