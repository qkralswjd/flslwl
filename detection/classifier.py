"""Learned classifier: "does this crop look like a monster or not?"

Trained on the same data you were already collecting by hand -- every image
in templates_dir is a positive (monster) example, every image in
reject_templates_dir is a negative (not-monster, e.g. another player)
example. Features are HOG (edge/shape structure, via cv2.HOGDescriptor) +
an HSV color histogram (same idea as the earlier rule-based color check),
concatenated into one vector and fed to a linear SVM.

This is classical machine learning, not a deep neural network -- no GPU,
trains in well under a second on a few hundred images, and re-trains
automatically every time you capture a new example with 't' / 'r' so the
model keeps improving live.

Before classification, two cheap deterministic checks still run first
(same as the earlier rule-based matcher): a candidate must be at least
min_size_ratio of the average monster template's size, and must not be
dominated by a known non-target color (player-armor blue/pink).
"""

import glob
import logging
import os
import sys
import time

import cv2
import numpy as np

# paths.py (프로젝트 루트) import
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths import get_templates_save_dir, get_reject_templates_save_dir
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

logger = logging.getLogger("classifier")

FEATURE_SIZE = 64  # images are resized to FEATURE_SIZE x FEATURE_SIZE before feature extraction

_HOG = cv2.HOGDescriptor(
    _winSize=(FEATURE_SIZE, FEATURE_SIZE),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

DEFAULT_EXCLUDED_COLOR_RANGES = [
    # OpenCV hue is 0..180. h_min > h_max means the band wraps around 0/180 (for red/pink).
    {"name": "blue", "h_min": 95, "h_max": 135, "s_min": 60, "v_min": 60},
    {"name": "pink_red", "h_min": 145, "h_max": 10, "s_min": 60, "v_min": 60},
]


def _imread_unicode(path):
    """cv2.imread() can't open non-ASCII (e.g. Korean) paths on Windows -- it silently
    returns None instead of raising. Read the bytes ourselves and let cv2 decode the
    in-memory buffer instead."""
    with open(path, "rb") as f:
        data = f.read()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def _load_images(directory):
    images = []
    if directory and os.path.isdir(directory):
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            try:
                img = _imread_unicode(path)
            except OSError:
                img = None
            if img is not None and img.size > 0:
                images.append(img)
    return images


def _extract_features(img_bgr):
    resized = cv2.resize(img_bgr, (FEATURE_SIZE, FEATURE_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_features = _HOG.compute(gray).flatten()

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(color_hist, color_hist, 0, 1, cv2.NORM_MINMAX)

    return np.concatenate([hog_features, color_hist.flatten()])


def _is_excluded_color(crop_bgr, ranges, min_ratio):
    if not ranges or crop_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return False
    for rng in ranges:
        h_min, h_max = rng["h_min"], rng["h_max"]
        s_min, v_min = rng.get("s_min", 60), rng.get("v_min", 60)
        if h_min > h_max:
            mask = cv2.inRange(hsv, (0, s_min, v_min), (h_max, 255, 255)) | \
                   cv2.inRange(hsv, (h_min, s_min, v_min), (180, 255, 255))
        else:
            mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))
        if cv2.countNonZero(mask) / total >= min_ratio:
            return True
    return False


class MonsterClassifier:
    def __init__(
        self,
        templates_dir,
        reject_templates_dir,
        confidence_threshold=0.5,
        excluded_color_ranges=None,
        excluded_color_ratio=0.2,
        min_size_ratio=0.6,
    ):
        self.templates_dir = templates_dir
        self.reject_templates_dir = reject_templates_dir
        self.confidence_threshold = confidence_threshold
        self.excluded_color_ranges = (
            DEFAULT_EXCLUDED_COLOR_RANGES if excluded_color_ranges is None else excluded_color_ranges
        )
        self.excluded_color_ratio = excluded_color_ratio
        self.min_size_ratio = min_size_ratio

        self.model = None
        self.avg_w, self.avg_h = 0.0, 0.0
        self.positive_count, self.negative_count = 0, 0
        self.train()

    def train(self):
        positives = _load_images(self.templates_dir)
        negatives = _load_images(self.reject_templates_dir)
        self.positive_count, self.negative_count = len(positives), len(negatives)

        if positives:
            self.avg_w = sum(im.shape[1] for im in positives) / len(positives)
            self.avg_h = sum(im.shape[0] for im in positives) / len(positives)
        else:
            self.avg_w, self.avg_h = 0.0, 0.0

        if len(positives) < 2 or len(negatives) < 2:
            logger.info(
                f"Not enough data to train yet (monster={len(positives)}, reject={len(negatives)}, "
                "need >=2 of each) - pass-through until more are captured"
            )
            self.model = None
            return

        X = [_extract_features(img) for img in positives + negatives]
        y = [1] * len(positives) + [0] * len(negatives)
        # cv can't exceed the smaller class's example count -- keep it small so this
        # still works with a handful of examples right after the first few captures.
        cv = max(2, min(3, len(positives), len(negatives)))
        model = CalibratedClassifierCV(SVC(kernel="linear", class_weight="balanced"), ensemble=False, cv=cv)
        model.fit(X, y)
        self.model = model
        logger.info(f"Classifier trained on {len(positives)} monster / {len(negatives)} reject example(s)")

    def size_ok(self, width, height):
        if self.avg_w == 0 or self.avg_h == 0:
            return True
        return width >= self.avg_w * self.min_size_ratio and height >= self.avg_h * self.min_size_ratio

    def is_excluded_color(self, crop_bgr):
        return _is_excluded_color(crop_bgr, self.excluded_color_ranges, self.excluded_color_ratio)

    def predict(self, crop_bgr):
        """Returns (is_monster: bool, confidence 0..1)."""
        if self.model is None or crop_bgr.size == 0:
            return True, 1.0  # not trained yet -- pass-through
        features = _extract_features(crop_bgr).reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        monster_proba = float(proba[list(self.model.classes_).index(1)])
        return monster_proba >= self.confidence_threshold, monster_proba

    def confirm(self, detections, roi_frame):
        """Same interface as the old TemplateMatcher.confirm(): filters Detection
        objects down to ones the classifier thinks are a monster, stamping confidence."""
        if self.model is None:
            return detections

        confirmed = []
        for det in detections:
            if not self.size_ok(det.width, det.height):
                continue
            crop = roi_frame[det.y:det.y + det.height, det.x:det.x + det.width]
            if self.is_excluded_color(crop):
                continue
            is_monster, confidence = self.predict(crop)
            if not is_monster:
                continue
            det.confidence = confidence
            confirmed.append(det)
        return confirmed

    def save_template(self, image_bgr, name=None):
        """Saves a cropped region as a new positive (monster) example and retrains."""
        save_dir = get_templates_save_dir()  # 항상 EXE 옆에 저장
        if not name:
            name = f"template_{int(time.time() * 1000)}.png"
        path = os.path.join(save_dir, name)
        _imwrite_unicode(path, image_bgr)
        logger.info(f"Template saved: {path}")
        # 저장 후 읽기 경로도 save_dir 로 갱신
        self.templates_dir = save_dir
        self.train()
        return path

    def save_reject_template(self, image_bgr, name=None):
        """Saves a cropped region as a new negative (non-target) example and retrains."""
        save_dir = get_reject_templates_save_dir()  # 항상 EXE 옆에 저장
        if not name:
            name = f"reject_{int(time.time() * 1000)}.png"
        path = os.path.join(save_dir, name)
        _imwrite_unicode(path, image_bgr)
        logger.info(f"Reject template saved: {path}")
        # 저장 후 읽기 경로도 save_dir 로 갱신
        self.reject_templates_dir = save_dir
        self.train()
        return path
