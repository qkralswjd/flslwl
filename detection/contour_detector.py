"""Extracts bounding boxes from a binary mask via contour analysis."""

import cv2


class Detection:
    """A single raw detection for one frame (before tracking/ID assignment)."""

    __slots__ = ("x", "y", "width", "height", "center_x", "center_y", "area", "confidence")

    def __init__(self, x, y, w, h, confidence=1.0):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.center_x = x + w // 2
        self.center_y = y + h // 2
        self.area = w * h
        self.confidence = confidence  # overwritten by TemplateMatcher.confirm() when template matching is on


class ContourDetector:
    def __init__(self, min_area=100, max_area=50000):
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(x, y, w, h))
        return detections
