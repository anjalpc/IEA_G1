"""Fail-closed cache and conversion for ROS RGB images."""

import math
import threading

import numpy as np


EXPECTED_COLOR_FRAME = "d435_color_optical_frame"


class ColorFrameCache:
    """Retain only the latest valid and fresh RGB image."""

    def __init__(self, stale_after_s):
        """Configure the maximum accepted frame age in seconds."""
        self._stale_after_s = float(stale_after_s)
        if (
            not math.isfinite(self._stale_after_s)
            or self._stale_after_s <= 0.0
        ):
            raise ValueError("stale_after_s must be positive and finite")
        self._lock = threading.Lock()
        self._message = None
        self._received_at = None

    @staticmethod
    def _valid(message):
        try:
            height = int(message.height)
            width = int(message.width)
            step = int(message.step)
            encoding = str(message.encoding)
            frame_id = str(message.header.frame_id)
            data_size = len(message.data)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            encoding == "rgb8"
            and frame_id == EXPECTED_COLOR_FRAME
            and height > 0
            and width > 0
            and step >= width * 3
            and data_size == height * step
        )

    def offer(self, message, received_at):
        """Store a valid image, clearing the cache on invalid input."""
        try:
            received_at = float(received_at)
        except (TypeError, ValueError):
            received_at = math.nan
        valid = math.isfinite(received_at) and self._valid(message)
        with self._lock:
            if valid:
                self._message = message
                self._received_at = received_at
            else:
                self._message = None
                self._received_at = None
        return valid

    def latest(self, now):
        """Return the cached image only while its receipt time is fresh."""
        try:
            now = float(now)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(now):
            return None
        with self._lock:
            if self._message is None or self._received_at is None:
                return None
            age = now - self._received_at
            if age < 0.0 or age > self._stale_after_s:
                self._message = None
                self._received_at = None
                return None
            return self._message


def color_image_to_bgr(message):
    """Convert one validated rgb8 image to contiguous BGR pixels."""
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        int(message.height),
        int(message.step),
    )
    rgb = rows[:, : int(message.width) * 3].reshape(
        int(message.height),
        int(message.width),
        3,
    )
    return np.ascontiguousarray(rgb[:, :, ::-1])
