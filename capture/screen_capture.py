"""Real-time screen capture using mss.

Captures either a full monitor or a sub-region of it (capture_region),
and hands back a BGR numpy array ready for OpenCV.
"""

import cv2
import mss
import numpy as np


class ScreenCapturer:
    def __init__(self, monitor_index=1, region=None):
        """
        monitor_index: index into mss.mss().monitors (0 = all monitors combined,
                        1..N = individual physical monitors)
        region: optional dict {x, y, width, height} relative to the chosen
                monitor's top-left corner. None = capture the full monitor.
        """
        self._sct = mss.mss()
        self.monitor_index = monitor_index
        self.region = region
        self._monitor = self._resolve_monitor()

    def _resolve_monitor(self):
        monitors = self._sct.monitors
        if self.monitor_index < 0 or self.monitor_index >= len(monitors):
            raise ValueError(
                f"Invalid monitor_index {self.monitor_index}. "
                f"Available range: 0..{len(monitors) - 1}"
            )
        base = monitors[self.monitor_index]
        if self.region:
            return {
                "left": base["left"] + self.region["x"],
                "top": base["top"] + self.region["y"],
                "width": self.region["width"],
                "height": self.region["height"],
            }
        return base

    def set_monitor(self, monitor_index):
        self.monitor_index = monitor_index
        self._monitor = self._resolve_monitor()

    def set_region(self, region):
        self.region = region
        self._monitor = self._resolve_monitor()

    def list_monitors(self):
        return self._sct.monitors

    def grab(self):
        """Returns the captured frame as a BGR numpy array."""
        shot = self._sct.grab(self._monitor)
        frame = np.asarray(shot)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self._sct.close()
