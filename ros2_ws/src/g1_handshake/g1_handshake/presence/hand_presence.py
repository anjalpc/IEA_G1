"""Pure stability-gating logic for the handshake hand-presence trigger.

No ROS, camera, MediaPipe, RealSense, or Unitree dependency lives here.

Instead of requiring perfectly consecutive detections, this gate tolerates
brief missed detections by evaluating a sliding window.
"""

from collections import deque


class HandPresenceDetector:
    """Trigger when enough hits occur within a recent observation window."""

    def __init__(
        self,
        required_stable_frames,
        window_size=None,
    ):
        required_stable_frames = int(
            required_stable_frames
        )

        if required_stable_frames < 1:
            raise ValueError(
                "required_stable_frames must be at least 1"
            )

        if window_size is None:
            window_size = required_stable_frames

        window_size = int(window_size)

        if window_size < required_stable_frames:
            raise ValueError(
                "window_size must be >= required_stable_frames"
            )

        self.required_stable_frames = (
            required_stable_frames
        )
        self.window_size = window_size

        self._observations = deque(
            maxlen=self.window_size
        )

    def reset(self):
        """Clear accumulated observations."""
        self._observations.clear()

    def observe(self, hit):
        """Feed one hit/miss observation.

        Return True when at least required_stable_frames positive
        observations exist inside the current sliding window.

        After firing, the window is cleared. The caller should still latch
        the event if only one trigger per presentation is allowed.
        """
        self._observations.append(
            bool(hit)
        )

        if (
            len(self._observations)
            < self.required_stable_frames
        ):
            return False

        hit_count = sum(
            self._observations
        )

        if (
            hit_count
            >= self.required_stable_frames
        ):
            self._observations.clear()
            return True

        return False
