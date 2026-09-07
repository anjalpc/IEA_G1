"""Real-image tests for HandDetector.

A trained model needs real-looking input to mean anything, so this uses an
actual photo (test/fixtures/hand_sample.jpg) rather than synthetic arrays —
unlike hand_presence.py's pure logic, which is genuinely hardware-agnostic.

Skips cleanly wherever mediapipe or the downloaded model file aren't
available (most colcon/CI environments), the same way this repo already
keeps unitree_sdk2py-dependent code import-safe without the SDK installed.
"""

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from g1_handshake.presence.hand_detector import HandDetector  # noqa: E402


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "hand_sample.jpg")


def detector():
    instance = HandDetector(min_detection_confidence=0.5)
    try:
        instance.start()
    except FileNotFoundError:
        pytest.skip("hand_landmarker.task model not downloaded locally")
    return instance


def test_finds_a_hand_in_a_real_photo():
    frame = cv2.imread(FIXTURE)
    assert frame is not None, f"could not load fixture at {FIXTURE}"
    instance = detector()
    try:
        hand = instance.detect(frame)
    finally:
        instance.stop()
    assert hand is not None
    assert 0 <= hand.center_x < frame.shape[1]
    assert 0 <= hand.center_y < frame.shape[0]
    assert 0.0 < hand.confidence <= 1.0


def test_blank_frame_is_not_a_false_positive():
    frame = cv2.imread(FIXTURE)
    blank = np.zeros_like(frame)
    instance = detector()
    try:
        hand = instance.detect(blank)
    finally:
        instance.stop()
    assert hand is None


def test_detect_before_start_raises():
    instance = HandDetector()
    with pytest.raises(RuntimeError, match="not started"):
        instance.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_double_start_raises():
    instance = detector()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            instance.start()
    finally:
        instance.stop()
