"""Tests for camera publication without opening physical devices."""

import numpy as np

from builtin_interfaces.msg import Time

from g1_hardware_bridge.camera.camera_bridge_node import depth_image_message


def test_depth_image_is_16uc1_in_millimetres():
    depth = np.array([[0.0, 425.8], [70000.0, 1480.0]], dtype=np.float32)
    message = depth_image_message(depth, Time(sec=12, nanosec=34))

    assert message.height == 2
    assert message.width == 2
    assert message.encoding == "16UC1"
    assert message.step == 4
    assert message.header.frame_id == "g1_depth_camera"
    assert np.frombuffer(message.data, dtype=np.uint16).tolist() == [
        0, 425, 65535, 1480,
    ]
