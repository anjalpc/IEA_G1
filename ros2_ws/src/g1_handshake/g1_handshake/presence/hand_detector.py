"""MediaPipe-based human-hand detector.

The detector is independent of ROS and camera acquisition. It receives an
OpenCV BGR frame and returns the detected hand center in pixel coordinates
of the ORIGINAL input frame.

Inference is optionally performed on a smaller image for efficiency.
"""


class DetectedHand:
    """One detected hand in original-frame pixel coordinates."""

    def __init__(
        self,
        center_x,
        center_y,
        confidence,
        bbox=None,
        landmarks=None,
    ):
        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.confidence = float(confidence)
        self.bbox = bbox
        self.landmarks = tuple(landmarks or ())


class HandDetector:
    """Detect one human hand using MediaPipe HandLandmarker."""

    def __init__(
        self,
        min_detection_confidence=0.5,
        model_path=None,
        max_width=960,
    ):
        self.min_detection_confidence = float(
            min_detection_confidence
        )
        self.model_path = model_path
        self.max_width = int(max_width)

        if self.max_width < 1:
            raise ValueError(
                "max_width must be positive"
            )

        self._landmarker = None

    def start(self):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        if self._landmarker is not None:
            raise RuntimeError(
                "hand detector is already started"
            )

        model_path = (
            self.model_path
            or _default_model_path()
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=(
                self.min_detection_confidence
            ),
        )

        self._landmarker = (
            HandLandmarker.create_from_options(
                options
            )
        )

    def detect_all(self, color_frame_bgr):
        """Return up to two detected hands in original-frame coordinates."""
        import cv2
        import mediapipe as mp
        import numpy as np

        if self._landmarker is None:
            raise RuntimeError(
                "hand detector is not started"
            )

        frame = np.asarray(color_frame_bgr)

        if (
            frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError(
                "expected BGR frame with shape HxWx3"
            )

        original_height, original_width = (
            frame.shape[:2]
        )

        detection_frame = frame

        if original_width > self.max_width:
            scale = (
                self.max_width
                / float(original_width)
            )

            detection_height = max(
                1,
                int(round(
                    original_height * scale
                )),
            )

            detection_frame = cv2.resize(
                frame,
                (
                    self.max_width,
                    detection_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(
            detection_frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb = np.ascontiguousarray(rgb)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self._landmarker.detect(image)

        if not result.hand_landmarks:
            return []

        detected = []
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            # Wrist and finger MCP joints define a stable palm centre.
            palm = [landmarks[index] for index in (0, 5, 9, 13, 17)]
            center_x = sum(point.x for point in palm) / len(palm)
            center_y = sum(point.y for point in palm) / len(palm)
            x_values = [point.x for point in landmarks]
            y_values = [point.y for point in landmarks]
            confidence = (
                result.handedness[hand_index][0].score
                if len(result.handedness) > hand_index
                and result.handedness[hand_index]
                else 1.0
            )
            detected.append(DetectedHand(
                center_x=center_x * original_width,
                center_y=center_y * original_height,
                confidence=confidence,
                bbox=(
                    int(min(x_values) * original_width),
                    int(min(y_values) * original_height),
                    int(max(x_values) * original_width),
                    int(max(y_values) * original_height),
                ),
                landmarks=tuple(
                    (
                        int(point.x * original_width),
                        int(point.y * original_height),
                    )
                    for point in landmarks
                ),
            ))
        return detected

    def detect(self, color_frame_bgr):
        """Return the first detected hand, or None, for compatibility."""
        hands = self.detect_all(color_frame_bgr)
        return hands[0] if hands else None

    def stop(self):
        if self._landmarker is not None:
            self._landmarker.close()

        self._landmarker = None


def _default_model_path():
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "hand_landmarker.task",
    )

    # Compatibility with the model's previous location before the
    # presence/ subpackage restructure.
    if not os.path.isfile(path):
        path = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            "models",
            "hand_landmarker.task",
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "hand_landmarker.task model not found; "
            "place it in g1_handshake/models/ or "
            "g1_handshake/presence/models/, or pass "
            "model_path explicitly"
        )

    return path
