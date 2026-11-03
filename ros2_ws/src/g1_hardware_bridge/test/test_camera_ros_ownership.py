"""ROS ownership tests for the physical camera bridge."""

import rclpy
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image

from g1_hardware_bridge.camera import node as camera_node


class PublisherRecorder:
    """Record published messages."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def depth_message():
    """Return one valid driver-owned depth message."""
    message = Image()
    message.header.frame_id = "d435_depth_optical_frame"
    message.height = 1
    message.width = 2
    message.encoding = "16UC1"
    message.step = 4
    message.data = b"\x64\x00\xc8\x00"
    return message


def color_message():
    """Return one valid driver-owned RGB message."""
    message = Image()
    message.header.frame_id = "d435_color_optical_frame"
    message.height = 1
    message.width = 2
    message.encoding = "rgb8"
    message.step = 6
    message.data = bytes((255, 0, 0, 0, 255, 0))
    return message


def test_bridge_subscribes_to_driver_depth_and_preserves_message():
    now = [10.0]
    rclpy.init()
    node = None
    try:
        node = camera_node.CameraBridgeNode(
            monotonic=lambda: now[0],
        )
        assert node._depth_subscription.topic_name == (
            "/d435/depth/image_rect_raw"
        )
        assert node.depth_publisher.topic_name == (
            "/g1/camera/depth/image_raw"
        )

        recorder = PublisherRecorder()
        node.depth_publisher = recorder

        message = depth_message()
        node._depth_callback(message)
        now[0] = 10.1
        node._publish_frames()

        assert recorder.messages == [message]
        assert recorder.messages[0].header.frame_id == (
            "d435_depth_optical_frame"
        )
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_bridge_does_not_republish_stale_depth():
    now = [20.0]

    rclpy.init()
    node = None
    try:
        node = camera_node.CameraBridgeNode(
            monotonic=lambda: now[0],
        )
        recorder = PublisherRecorder()
        node.depth_publisher = recorder

        node._depth_callback(depth_message())
        now[0] = 21.0
        node._publish_frames()

        assert recorder.messages == []
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_bridge_subscribes_to_driver_color_and_publishes_jpeg():
    """Preserve the compatibility RGB topic from driver-owned color."""
    now = [30.0]

    rclpy.init()
    node = None
    try:
        node = camera_node.CameraBridgeNode(
            monotonic=lambda: now[0],
        )
        assert node._color_subscription.topic_name == (
            "/d435/color/image_raw"
        )
        assert node.rgb_publisher.topic_name == (
            "/g1/camera/rgb/compressed"
        )

        recorder = PublisherRecorder()
        node.rgb_publisher = recorder
        node._color_callback(color_message())

        assert len(recorder.messages) == 1
        assert recorder.messages[0].format == "jpeg"
        assert recorder.messages[0].header.frame_id == (
            "d435_color_optical_frame"
        )
        assert bytes(recorder.messages[0].data).startswith(b"\xff\xd8")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_driver_input_subscriptions_match_reliable_d435_publishers():
    """Prevent sustained D435 delivery loss on Foxy CycloneDDS."""
    rclpy.init()
    node = None
    try:
        node = camera_node.CameraBridgeNode()

        assert node._color_subscription.qos_profile.reliability == (
            QoSReliabilityPolicy.RELIABLE
        )
        assert node._depth_subscription.qos_profile.reliability == (
            QoSReliabilityPolicy.RELIABLE
        )
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_bridge_source_does_not_own_realsense_device():
    source = camera_node.__file__
    text = open(source, encoding="utf-8").read()

    assert "DepthCamera" not in text
    assert "pyrealsense2" not in text


def test_bridge_separates_depth_reception_from_frame_publishing():
    """Keep high-rate depth reception independent from publishing work."""
    rclpy.init()
    node = None
    executor = None
    try:
        node = camera_node.CameraBridgeNode()

        callback_groups = {
            node._color_subscription.callback_group,
            node._depth_subscription.callback_group,
            node._publish_timer.callback_group,
        }
        assert len(callback_groups) == 3

        executor = camera_node.create_executor(node)
        assert executor._executor._max_workers == 3
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
