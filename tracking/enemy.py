"""Enemy state: position, size, velocity, and short position history."""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Enemy:
    id: int
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float = 1.0
    velocity_x: float = 0.0          # pixels per frame (last update)
    velocity_y: float = 0.0
    pixels_per_second_x: float = 0.0
    pixels_per_second_y: float = 0.0
    missing_frames: int = 0
    predicted: bool = False          # True if position this frame is a prediction, not a real detection
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    def update_position(self, x, y, w, h, dt, confidence=1.0):
        """Called when a matching detection was found this frame."""
        center_x = x + w // 2
        center_y = y + h // 2

        self.velocity_x = center_x - self.center_x
        self.velocity_y = center_y - self.center_y
        if dt > 0:
            self.pixels_per_second_x = self.velocity_x / dt
            self.pixels_per_second_y = self.velocity_y / dt

        self.x, self.y, self.width, self.height = x, y, w, h
        self.center_x, self.center_y = center_x, center_y
        self.confidence = confidence
        self.missing_frames = 0
        self.predicted = False
        self.last_seen_at = time.time()
        self.history.append((center_x, center_y))

    def predict_next_position(self):
        """Extrapolate the next center position from the last known velocity."""
        return (
            int(self.center_x + self.velocity_x),
            int(self.center_y + self.velocity_y),
        )

    VELOCITY_DECAY = 0.75  # shrink the coasting velocity each missed frame so a long gap
    #                        can't drag the predicted position far from where the target
    #                        actually is -- otherwise the eventual real re-match jumps back
    #                        from that stale prediction, producing a huge, bogus velocity spike.

    def apply_prediction(self):
        """Called when no matching detection was found this frame."""
        pred_x, pred_y = self.predict_next_position()
        self.x += pred_x - self.center_x
        self.y += pred_y - self.center_y
        self.center_x, self.center_y = pred_x, pred_y
        self.velocity_x *= self.VELOCITY_DECAY
        self.velocity_y *= self.VELOCITY_DECAY
        self.missing_frames += 1
        self.predicted = True
