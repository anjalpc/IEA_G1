"""Tests for fail-closed ROS color handling."""

import numpy as np
from sensor_msgs.msg import Image

from g1_hardware_bridge.camera.ros_color import (
    ColorFrameCache,
    color_image_to_bgr,
)


def color_message():
    """Return a valid padded rgb8 image."""
    message = Image()
    message.header.frame_id = "d435_color_optical_frame"
    message.height = 1
    message.width = 2
    message.encoding = "rgb8"
    message.step = 8
    message.data = bytes((255, 0, 0, 0, 255, 0, 9, 9))
    return message


def test_accepts_fresh_driver_color_and_converts_to_bgr():
    cache = ColorFrameCache(stale_after_s=0.25)
    message = color_message()

    assert cache.offer(message, received_at=10.0)
    assert cache.latest(now=10.2) is message

    bgr = color_image_to_bgr(message)
    assert bgr.shape == (1, 2, 3)
    assert bgr.dtype == np.uint8
    assert bgr.tolist() == [[[0, 0, 255], [0, 255, 0]]]


def test_rejects_stale_or_structurally_invalid_color():
    cache = ColorFrameCache(stale_after_s=0.25)
    message = color_message()

    assert cache.offer(message, received_at=20.0)
    assert cache.latest(now=20.3) is None

    message.header.frame_id = "wrong_frame"
    assert not cache.offer(message, received_at=21.0)
    assert cache.latest(now=21.0) is None
