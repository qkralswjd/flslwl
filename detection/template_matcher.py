"""Confirms a motion candidate actually looks like a known target, instead
of trusting motion alone (motion also fires on non-enemy things: other
players, pets, floating damage numbers, ...).

Reference crops (PNG/JPG) live in templates_dir -- each file is one
template. A candidate bounding box is kept only if its cropped image
best-matches one of them above match_threshold, using normalized
cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED). Both images are
resized to a small fixed resolution before comparing, so distance/zoom
differences wash out and each comparison stays cheap even with a large
template library (size_ok() already screens out wildly wrong-sized
candidates before any of this runs).

reject_templates_dir is the same idea in reverse: crops of things that
LOOK like a target but aren't (e.g. other players, who are also moving
humanoid sprites and can outscore a loose target match). A candidate is
rejected if it matches a reject template strongly on its own, OR matches
a reject template better than it matches any target template -- checked
twice, independently, on shape (matchTemplate) AND on color (an HSV
histogram of the whole crop), since two humanoid sprites can share a
silhouette but differ clearly in armor/skin color, or vice versa.

Before any of that, a candidate whose crop is dominated by a known
non-target color (default: player-armor blue/pink) is vetoed outright --
see DEFAULT_EXCLUDED_COLOR_RANGES -- since that's a much cheaper and more
reliable signal than appearance matching when it applies.

If templates_dir has no images yet, confirm() passes every candidate
through unfiltered -- so motion detection keeps working standalone until
you actually add templates (press 't' in the main window to capture a
target, 'r' to capture a reject/non-target).
"""

import glob
import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger("template_matcher")


def _imread_unicode(path):
    """cv2.imread() can't open non-ASCII (e.g. Korean) paths on Windows -- it silently
    returns None instead of raising. Read the bytes ourselves (Python's open() handles
    any Unicode path correctly) and let cv2 decode the in-memory buffer instead."""
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


COLOR_HIST_BINS = (30, 32)  # (Hue, Saturation) bins -- Value/brightness left out, so
#                             lighting/shadow differences matter less than hue/saturation


def _color_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(COLOR_HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _load_templates_from(directory, label):
    templates = []
    if directory and os.path.isdir(directory):
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            try:
                img = _imread_unicode(path)
            except OSError:
                img = None
            if img is not None and img.size > 0:
                templates.append((os.path.basename(path), img, _color_hist(img)))
            else:
                logger.warning(f"Could not read {label} template image: {path}")
    return templates


DEFAULT_EXCLUDED_COLOR_RANGES = [
    # OpenCV hue is 0..180. h_min > h_max means the band wraps around 0/180 (for red/pink).
    {"name": "blue", "h_min": 95, "h_max": 135, "s_min": 60, "v_min": 60},
    {"name": "pink_red", "h_min": 145, "h_max": 10, "s_min": 60, "v_min": 60},
]


def _is_excluded_color(crop_bgr, ranges, min_ratio):
    """True if at least `min_ratio` of the crop's pixels fall in one of the excluded
    hue bands (e.g. player-armor blue/pink) -- an absolute veto, independent of how
    well the crop otherwise matches a target template."""
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


MATCH_SIZE = 80  # fixed comparison resolution -- higher keeps more distinguishing detail
#                  (monster vs. similarly-shaped player) at some extra per-comparison cost
EARLY_EXIT_SCORE = 0.92  # a match this strong is decisive; stop scanning the remaining templates


def _prep(img, size=MATCH_SIZE):
    return cv2.resize(img, (size, size))


def _best_match_against(candidate_bgr, templates):
    """Returns (best_score 0..1, template_name) for the best shape match in `templates`.

    Both images are resized to the same fixed MATCH_SIZE before matchTemplate, so
    each comparison is a single correlation (no sliding search, no per-template
    multi-scale loop) -- size differences are already handled upstream by
    TemplateMatcher.size_ok(), so re-testing several scales here was redundant.
    """
    if not templates or candidate_bgr.size == 0:
        return 0.0, None
    if candidate_bgr.std() < 5:
        # Near-flat crop (e.g. empty background): TM_CCOEFF_NORMED divides by std-dev,
        # which is ~0 here and produces numerically unstable/meaningless scores.
        return 0.0, None

    candidate_small = _prep(candidate_bgr)
    best_score, best_name = 0.0, None

    for name, template, _hist in templates:
        result = cv2.matchTemplate(candidate_small, _prep(template), cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        if score > best_score:
            best_score, best_name = score, name
            if best_score >= EARLY_EXIT_SCORE:
                break

    return best_score, best_name


def _best_color_match_against(candidate_bgr, templates):
    """Returns (best_score 0..1, template_name) for the best HSV-histogram match."""
    if not templates or candidate_bgr.size == 0:
        return 0.0, None

    candidate_hist = _color_hist(candidate_bgr)
    best_score, best_name = 0.0, None

    for name, _img, hist in templates:
        score = max(0.0, float(cv2.compareHist(candidate_hist, hist, cv2.HISTCMP_CORREL)))
        if score > best_score:
            best_score, best_name = score, name

    return best_score, best_name


class TemplateMatcher:
    def __init__(
        self,
        templates_dir,
        reject_templates_dir=None,
        match_threshold=0.35,
        color_match_threshold=0.35,
        size_tolerance=0.4,
        reject_threshold=0.4,
        color_reject_threshold=0.7,
        excluded_color_ranges=None,
        excluded_color_ratio=0.2,
        min_size_ratio=0.6,
    ):
        self.templates_dir = templates_dir
        self.reject_templates_dir = reject_templates_dir
        self.match_threshold = match_threshold  # shape: primary gate, see confirm()
        self.color_match_threshold = color_match_threshold  # color: secondary confirmation, see confirm()
        self.size_tolerance = size_tolerance  # allowed deviation from a template's own w/h, e.g. 0.4 = +/-40%
        self.reject_threshold = reject_threshold  # shape: this alone kills a candidate, regardless of target score
        self.color_reject_threshold = color_reject_threshold  # same idea, on the HSV color histogram
        self.excluded_color_ranges = (
            DEFAULT_EXCLUDED_COLOR_RANGES if excluded_color_ranges is None else excluded_color_ranges
        )
        self.excluded_color_ratio = excluded_color_ratio  # min fraction of pixels in a band to veto outright
        self.min_size_ratio = min_size_ratio  # candidate must be at least this fraction of the AVERAGE template
        #                                       size, regardless of any single template's own size (see size_ok)
        self.templates = _load_templates_from(self.templates_dir, "target")
        self.reject_templates = _load_templates_from(self.reject_templates_dir, "reject")
        self._recompute_avg_size()
        self._log_loaded()

    def _log_loaded(self):
        if not self.templates:
            logger.info(f"No templates found in {self.templates_dir} - matching runs pass-through until some are added")
        else:
            logger.info(f"Loaded {len(self.templates)} target template(s): {', '.join(n for n, _, _ in self.templates)}")
        if self.reject_templates:
            logger.info(f"Loaded {len(self.reject_templates)} reject template(s): {', '.join(n for n, _, _ in self.reject_templates)}")

    def _recompute_avg_size(self):
        """Average (width, height) across all target templates -- used as a minimum-size
        floor so a small artifact (a flickering torch flame, a floating item icon) can't
        pass just because it happens to fall within some individual template's tolerance."""
        if not self.templates:
            self.avg_template_w, self.avg_template_h = 0.0, 0.0
            return
        self.avg_template_w = sum(t.shape[1] for _n, t, _h in self.templates) / len(self.templates)
        self.avg_template_h = sum(t.shape[0] for _n, t, _h in self.templates) / len(self.templates)

    def reload(self):
        self.templates = _load_templates_from(self.templates_dir, "target")
        self.reject_templates = _load_templates_from(self.reject_templates_dir, "reject")
        self._recompute_avg_size()
        self._log_loaded()

    def save_template(self, image_bgr, name=None):
        """Saves a cropped region as a new target template and registers it immediately."""
        os.makedirs(self.templates_dir, exist_ok=True)
        if not name:
            name = f"template_{int(time.time() * 1000)}.png"
        path = os.path.join(self.templates_dir, name)
        _imwrite_unicode(path, image_bgr)
        self.templates.append((name, image_bgr, _color_hist(image_bgr)))
        self._recompute_avg_size()
        logger.info(f"Template saved: {path}")
        return path

    def save_reject_template(self, image_bgr, name=None):
        """Saves a cropped region as a new reject/non-target template (e.g. another player)."""
        if not self.reject_templates_dir:
            raise ValueError("reject_templates_dir is not configured")
        os.makedirs(self.reject_templates_dir, exist_ok=True)
        if not name:
            name = f"reject_{int(time.time() * 1000)}.png"
        path = os.path.join(self.reject_templates_dir, name)
        _imwrite_unicode(path, image_bgr)
        self.reject_templates.append((name, image_bgr, _color_hist(image_bgr)))
        logger.info(f"Reject template saved: {path}")
        return path

    def size_ok(self, width, height):
        """True if (width, height) is close enough to at least one template's own
        pixel size (within size_tolerance) -- e.g. width=200 can never be the same
        target as a template that's 30px wide, no matter how similar it looks.
        Also enforces a floor relative to the AVERAGE template size, so something
        much smaller than any real target (a torch flicker, an item icon) can't
        sneak through just because it's within tolerance of one unusually small template."""
        if not self.templates:
            return True
        if width < self.avg_template_w * self.min_size_ratio or height < self.avg_template_h * self.min_size_ratio:
            return False
        lo, hi = 1.0 - self.size_tolerance, 1.0 + self.size_tolerance
        for _name, template, _hist in self.templates:
            th, tw = template.shape[:2]
            if lo <= width / tw <= hi and lo <= height / th <= hi:
                return True
        return False

    def is_excluded_color(self, crop_bgr):
        """True if the crop is dominated by a known non-target color (e.g. player armor
        blue/pink) -- see DEFAULT_EXCLUDED_COLOR_RANGES / excluded_color_ratio."""
        return _is_excluded_color(crop_bgr, self.excluded_color_ranges, self.excluded_color_ratio)

    def best_match(self, candidate_bgr):
        """Returns (best_score 0..1, template_name) for the best-matching target template (shape)."""
        return _best_match_against(candidate_bgr, self.templates)

    def best_reject_match(self, candidate_bgr):
        """Returns (best_score 0..1, template_name) for the best-matching reject template (shape)."""
        return _best_match_against(candidate_bgr, self.reject_templates)

    def best_color_match(self, candidate_bgr):
        """Returns (best_score 0..1, template_name) for the best-matching target template (color)."""
        return _best_color_match_against(candidate_bgr, self.templates)

    def best_color_reject_match(self, candidate_bgr):
        """Returns (best_score 0..1, template_name) for the best-matching reject template (color)."""
        return _best_color_match_against(candidate_bgr, self.reject_templates)

    def confirm(self, detections, roi_frame):
        """Filters Detection objects down to ones that:
        1) are close enough to a known target's size AND at least min_size_ratio of
           the AVERAGE target size (rules out small artifacts like torch flicker),
        2) aren't dominated by a known non-target color (blue/pink armor),
        3) shape-match a target template above match_threshold -- the primary gate,
        4) color-match a target template above color_match_threshold -- a secondary
           confirmation on top of the shape match,
        5) don't color- or shape-match a reject template too strongly (absolute floor
           or relative loss), if any reject templates are loaded.
        Stamps confidence with max(color_score, shape_score).

        Both match_threshold and color_match_threshold default low (see TemplateMatcher
        __init__) -- requiring two independent noisy signals to each clear their own bar
        compounds failure, so neither should be set "safely strict" on its own.
        """
        if not self.templates:
            return detections

        confirmed = []
        for det in detections:
            if not self.size_ok(det.width, det.height):
                continue  # wrong size for any known target -- skip the (expensive) appearance check entirely

            # Un-padded crop for both shape and color -- padding pulls in background
            # pixels (floor/wall) that dilute both the color histogram AND the shape
            # correlation (a padded crop resized to MATCH_SIZE looks structurally
            # different from a template that's just the character, tanking the score
            # even for an exact match). Tried padding to recover full-silhouette
            # motion boxes; net effect was worse on the common case, so dropped it.
            tight_crop = roi_frame[det.y:det.y + det.height, det.x:det.x + det.width]

            if self.is_excluded_color(tight_crop):
                continue  # dominated by a known non-target color (player armor blue/pink) -- veto outright

            shape_score, _ = self.best_match(tight_crop)
            if shape_score < self.match_threshold:
                continue  # doesn't look like any known target shape-wise

            color_score, _ = self.best_color_match(tight_crop)
            if color_score < self.color_match_threshold:
                continue  # ...or doesn't look like any known target color-wise

            if self.reject_templates:
                reject_score, _ = self.best_reject_match(tight_crop)
                if reject_score >= self.reject_threshold or reject_score >= shape_score:
                    continue  # shape looks enough like (or more like) a known non-target

                reject_color_score, _ = self.best_color_reject_match(tight_crop)
                if reject_color_score >= self.color_reject_threshold or reject_color_score >= color_score:
                    continue  # color looks enough like (or more like) a known non-target

            det.confidence = max(color_score, shape_score)
            confirmed.append(det)
        return confirmed
