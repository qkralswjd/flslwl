"""In-window trackbars + a mask/ROI thumbnail composited into the corner
of the main frame -- everything lives in ONE OpenCV window instead of
separate debug windows.

Tune detection live by dragging the trackbars until the mask thumbnail
cleanly isolates the moving target, then copy the values into
config/config.json.
"""

import cv2
import numpy as np


class DebugView:
    def __init__(self, motion_detector, contour_detector, window_name, classifier=None, thumb_size=(220, 150)):
        self.motion_detector = motion_detector
        self.contour_detector = contour_detector
        self.classifier = classifier
        self.window_name = window_name
        self.thumb_w, self.thumb_h = thumb_size
        self._trackbars_created = False

    @staticmethod
    def _noop(_value):
        pass

    def create_trackbars(self):
        """Call once, after the main window already exists (cv2.namedWindow)."""
        cv2.createTrackbar("Var Threshold", self.window_name, self.motion_detector.var_threshold, 100, self._noop)
        cv2.createTrackbar("Min Area", self.window_name, min(self.contour_detector.min_area, 20000), 20000, self._noop)
        cv2.createTrackbar("Max Area", self.window_name, min(self.contour_detector.max_area, 200000), 200000, self._noop)
        if self.classifier is not None:
            cv2.createTrackbar(
                "Confidence %", self.window_name, int(self.classifier.confidence_threshold * 100), 100, self._noop
            )
            cv2.createTrackbar(
                "Excl Color %", self.window_name, int(self.classifier.excluded_color_ratio * 100), 100, self._noop
            )
            cv2.createTrackbar(
                "Min Size %", self.window_name, int(self.classifier.min_size_ratio * 100), 100, self._noop
            )
        self._trackbars_created = True

    def read_trackbars(self):
        """Pull current trackbar values back into the detector objects (call once per detection tick)."""
        if not self._trackbars_created:
            return
        self.motion_detector.var_threshold = max(1, cv2.getTrackbarPos("Var Threshold", self.window_name))
        self.contour_detector.min_area = cv2.getTrackbarPos("Min Area", self.window_name)
        self.contour_detector.max_area = cv2.getTrackbarPos("Max Area", self.window_name)
        if self.classifier is not None:
            self.classifier.confidence_threshold = cv2.getTrackbarPos("Confidence %", self.window_name) / 100.0
            self.classifier.excluded_color_ratio = cv2.getTrackbarPos("Excl Color %", self.window_name) / 100.0
            self.classifier.min_size_ratio = cv2.getTrackbarPos("Min Size %", self.window_name) / 100.0

    def draw_into(self, display_frame, roi_frame, mask):
        """Composites a small [ROI | Mask] thumbnail into the top-right corner of display_frame."""
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        roi_small = cv2.resize(roi_frame, (self.thumb_w, self.thumb_h))
        mask_small = cv2.resize(mask_bgr, (self.thumb_w, self.thumb_h))

        cv2.putText(roi_small, "ROI", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(mask_small, "Motion Mask", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        thumb = np.hstack([roi_small, mask_small])

        th, tw = thumb.shape[:2]
        frame_h, frame_w = display_frame.shape[:2]
        x0, y0 = frame_w - tw - 10, 10
        if x0 < 0 or y0 + th > frame_h:
            return  # main window too small to fit the thumbnail, skip silently

        display_frame[y0:y0 + th, x0:x0 + tw] = thumb
        cv2.rectangle(display_frame, (x0, y0), (x0 + tw, y0 + th), (255, 255, 255), 1)
