"""Tests for fail-closed ROS depth-frame consumption."""

from sensor_msgs.msg import Image

from g1_hardware_bridge.camera.ros_depth import DepthFrameCache


def depth_message():
    """Return one valid two-pixel RealSense depth image."""
    message = Image()
    message.header.frame_id = "d435_depth_optical_frame"
    message.height = 1
    message.width = 2
    message.encoding = "16UC1"
    message.is_bigendian = False
    message.step = 4
    message.data = b"\x64\x00\xc8\x00"
    return message


def test_returns_original_fresh_depth_message():
    cache = DepthFrameCache(stale_after_s=0.5)
    message = depth_message()

    assert cache.offer(message, received_at=10.0) is True
    assert cache.latest(now=10.49) is message


def test_rejects_stale_depth_message():
    cache = DepthFrameCache(stale_after_s=0.5)

    cache.offer(depth_message(), received_at=10.0)

    assert cache.latest(now=10.51) is None


def test_invalid_message_clears_previous_depth():
    cache = DepthFrameCache(stale_after_s=0.5)
    cache.offer(depth_message(), received_at=10.0)

    invalid = depth_message()
    invalid.encoding = "32FC1"

    assert cache.offer(invalid, received_at=10.1) is False
    assert cache.latest(now=10.2) is None


def test_rejects_unexpected_depth_frame():
    cache = DepthFrameCache(stale_after_s=0.5)
    invalid = depth_message()
    invalid.header.frame_id = "unexpected_depth_frame"

    assert cache.offer(invalid, received_at=10.0) is False
    assert cache.latest(now=10.0) is None


def test_rejects_inconsistent_depth_storage():
    cache = DepthFrameCache(stale_after_s=0.5)
    invalid = depth_message()
    invalid.data = b"\x64\x00"

    assert cache.offer(invalid, received_at=10.0) is False
    assert cache.latest(now=10.0) is None
