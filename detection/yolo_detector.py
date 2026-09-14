"""YOLOv8 커스텀 모델 탐지기 (monster / adena 2클래스).

monster-adena-detector 프로젝트의 detector.py 를 flslwl_master 스타일로 이식.

학습된 커스텀 모델 (best.pt) 전용:
    class_id=0 → monster
    class_id=1 → adena

사용법:
    detector = YoloDetector(model_path="runs/detect/monster_v1/weights/best.pt")
    if detector.is_loaded:
        results = detector.detect(roi_frame)
        monsters = [d for d in results if d.class_id == YoloDetector.CLASS_MONSTER]
        adenas   = [d for d in results if d.class_id == YoloDetector.CLASS_ADENA]
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger("yolo_detector")

# ── 커스텀 모델 클래스 ID ────────────────────────────────────────────────────
CLASS_MONSTER = 0
CLASS_ADENA   = 1
KNOWN_CLASSES = {CLASS_MONSTER: "monster", CLASS_ADENA: "adena"}


@dataclass
class YoloDetection:
    """YOLO 탐지 결과 하나.

    기존 Detection(contour_detector.py)과 구분하기 위해 YoloDetection으로 명명.
    class_id로 monster/adena를 구분한다.
    """
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_id: int = CLASS_MONSTER
    class_name: str = "monster"

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    # monster-adena-detector 호환 alias
    @property
    def cx(self) -> int:
        return self.center_x

    @property
    def cy(self) -> int:
        return self.center_y

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def tlbr(self):
        """(x1, y1, x2, y2) 형식."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class YoloDetector:
    """커스텀 best.pt 전용 YOLO 탐지기 (monster=0 / adena=1 2클래스).

    enabled=False 이거나 ultralytics 미설치이면 항상 빈 리스트를 반환하므로
    기존 RealtimeTemplateDetector 흐름에 영향 없음.

    Args:
        model_path    : best.pt 경로
        confidence    : 탐지 신뢰도 임계값 (기본 0.4)
        iou_threshold : NMS IoU 임계값 (기본 0.45)
        device        : "cuda" 또는 "cpu"
        img_size      : 추론 이미지 크기 (기본 640)
        enabled       : False이면 detect()가 항상 [] 반환 (기능 OFF)
    """

    CLASS_MONSTER = CLASS_MONSTER
    CLASS_ADENA   = CLASS_ADENA

    def __init__(
        self,
        model_path: str = "runs/detect/monster_v1/weights/best.pt",
        confidence: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        img_size: int = 640,
        enabled: bool = True,
    ):
        self._conf       = confidence
        self._iou        = iou_threshold
        self._device     = device
        self._img_size   = img_size
        self._enabled    = enabled
        self._model      = None
        self._model_names: dict = {}
        self._fps_ticks: list = []
        self._fps: float = 0.0

        if self._enabled:
            self._load_model(model_path)
        else:
            logger.info("[YoloDetector] enabled=False — 탐지 비활성")

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model_names = self._model.names  # {0:'monster', 1:'adena'}

            logger.info(f"[YoloDetector] 모델 로드: {model_path} @ {self._device}")
            logger.info(
                "[YoloDetector] 탐지 클래스: "
                + ", ".join(f"{k}={v}" for k, v in self._model_names.items())
            )

            # 클래스 검증
            if self._model_names.get(0) != "monster":
                logger.warning(
                    f"[YoloDetector] ⚠ class_id=0 이 'monster' 가 아님: "
                    f"{self._model_names.get(0)} — best.pt 확인 필요"
                )
            if self._model_names.get(1) != "adena":
                logger.warning(
                    f"[YoloDetector] ⚠ class_id=1 이 'adena' 가 아님: "
                    f"{self._model_names.get(1)} — best.pt 확인 필요"
                )

            # GPU 워밍업
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False, device=self._device)
            logger.info("[YoloDetector] 워밍업 완료 — 준비됨")

        except FileNotFoundError:
            logger.warning(
                f"[YoloDetector] 모델 파일 없음: {model_path} "
                "— YoloDetector 비활성 (기존 템플릿 탐지만 사용)"
            )
            self._model = None
        except ImportError:
            logger.warning(
                "[YoloDetector] ultralytics 미설치 — pip install ultralytics "
                "— YoloDetector 비활성"
            )
            self._model = None
        except Exception as e:
            logger.error(f"[YoloDetector] 모델 로드 실패: {e}")
            self._model = None

    # ── 공개 API ─────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """탐지 활성화 여부."""
        return self._enabled

    @property
    def is_loaded(self) -> bool:
        """모델이 정상 로드된 경우 True."""
        return self._model is not None

    @property
    def fps(self) -> float:
        return self._fps

    def detect(self, frame: np.ndarray) -> List[YoloDetection]:
        """프레임에서 monster/adena를 탐지하여 YoloDetection 리스트 반환.

        enabled=False 또는 모델 미로드 시 항상 [] 반환.

        Args:
            frame: BGR numpy array (ROI 잘린 상태 또는 전체 프레임)

        Returns:
            YoloDetection 리스트. class_id=0=monster, class_id=1=adena.
        """
        if not self._enabled or self._model is None or frame is None:
            return []

        try:
            results = self._model(
                frame,
                conf    = self._conf,
                iou     = self._iou,
                imgsz   = self._img_size,
                device  = self._device,
                verbose = False,
            )
            detections: List[YoloDetection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf   = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    # 커스텀 모델 클래스만 허용 (0=monster, 1=adena)
                    if cls_id not in KNOWN_CLASSES:
                        continue

                    cls_name = self._model_names.get(cls_id, KNOWN_CLASSES[cls_id])
                    detections.append(YoloDetection(
                        x          = x1,
                        y          = y1,
                        width      = x2 - x1,
                        height     = y2 - y1,
                        confidence = round(conf, 3),
                        class_id   = cls_id,
                        class_name = cls_name,
                    ))

            self._tick_fps()
            if detections:
                n_mon = sum(1 for d in detections if d.class_id == CLASS_MONSTER)
                n_ade = sum(1 for d in detections if d.class_id == CLASS_ADENA)
                logger.debug(
                    f"[YoloDetector] monster={n_mon} adena={n_ade} "
                    f"(fps={self._fps:.1f})"
                )
            return detections

        except Exception as e:
            logger.error(f"[YoloDetector] 탐지 실패: {e}")
            return []

    def reload(self, model_path: str) -> None:
        """모델을 재로드합니다."""
        self._model = None
        self._load_model(model_path)

    # ── 내부 ──────────────────────────────────────────────────────────────────

    def _tick_fps(self) -> None:
        now = time.time()
        self._fps_ticks.append(now)
        if len(self._fps_ticks) > 30:
            self._fps_ticks.pop(0)
        if len(self._fps_ticks) >= 2:
            elapsed = self._fps_ticks[-1] - self._fps_ticks[0]
            if elapsed > 0:
                self._fps = round((len(self._fps_ticks) - 1) / elapsed, 1)
