"""ROS contract tests for staged physical-body odometry."""

import math
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu

from g1_state_estimation.physical_body_node import PhysicalBodyNode


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class PublisherRecorder:
    """Collect staged physical-body odometry."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def stamp(message, seconds):
    """Set one ROS timestamp from floating-point seconds."""

    message.header.stamp.sec = int(seconds)
    message.header.stamp.nanosec = int(
        round((seconds - int(seconds)) * 1e9)
    )
    return message


def imu(seconds):
    """Build one plausible physical pelvis-IMU sample."""

    message = stamp(Imu(), seconds)
    message.header.frame_id = "imu_in_pelvis"
    message.orientation.w = 1.0
    message.linear_acceleration.z = 9.81
    return message


def odometry(seconds, *, x=0.0, yaw=0.0):
    """Build one qualified planar GLIM sample."""

    message = stamp(Odometry(), seconds)
    message.header.frame_id = "odom"
    message.child_frame_id = "base_footprint"
    message.pose.pose.position.x = x
    message.pose.pose.orientation.z = math.sin(yaw * 0.5)
    message.pose.pose.orientation.w = math.cos(yaw * 0.5)
    return message


def make_node():
    if not rclpy.ok():
        rclpy.init()

    clock = FakeClock()
    node = PhysicalBodyNode(
        parameter_overrides=[
            Parameter("imu_warmup_samples", value=2),
        ],
        monotonic=clock,
    )
    return node, clock


def destroy(node):
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_uses_only_staged_inputs_and_output():
    node, _clock = make_node()
    try:
        assert node.odometry_subscription.topic_name == (
            "/g1/navigation/glim/odometry_planar"
        )
        assert node.imu_subscription.topic_name == (
            "/g1/state/imu_raw"
        )
        assert node.odometry_publisher.topic_name == (
            "/g1/navigation/physical/odometry_staged"
        )

        publisher_topics = {
            publisher.topic_name
            for publisher in node.publishers
        }
        assert "/g1/state/odometry" not in publisher_topics
        assert "/tf" not in publisher_topics
    finally:
        destroy(node)


def test_releases_only_after_pelvis_imu_warmup():
    node, _clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.imu_callback(imu(100.00))
        node.odometry_callback(odometry(100.00))
        assert recorder.messages == []

        node.imu_callback(imu(100.01))
        node.odometry_callback(
            odometry(100.02, x=0.4, yaw=0.2)
        )

        assert len(recorder.messages) == 1
        result = recorder.messages[0]
        assert result.header.frame_id == "odom"
        assert result.child_frame_id == "pelvis"
        assert result.pose.pose.position.x == 0.4
        assert result.pose.pose.position.z == 0.785
    finally:
        destroy(node)


def test_retries_pending_odometry_when_matching_imu_arrives():
    node, _clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.imu_callback(imu(100.00))
        node.imu_callback(imu(100.01))

        node.odometry_callback(
            odometry(100.20, x=0.4, yaw=0.2)
        )
        assert recorder.messages == []

        node.imu_callback(imu(100.19))

        assert len(recorder.messages) == 1
        result = recorder.messages[0]
        assert result.header.frame_id == "odom"
        assert result.child_frame_id == "pelvis"
        assert result.pose.pose.position.x == 0.4
        assert result.pose.pose.position.z == 0.785
    finally:
        destroy(node)


def test_uses_nearest_buffered_imu_when_latest_is_too_new():
    node, _clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.imu_callback(imu(100.00))
        node.imu_callback(imu(100.01))
        node.imu_callback(imu(100.20))

        node.odometry_callback(
            odometry(100.02, x=0.5, yaw=0.1)
        )

        assert len(recorder.messages) == 1
        result = recorder.messages[0]
        assert result.pose.pose.position.x == 0.5
        assert result.pose.pose.position.z == 0.785
    finally:
        destroy(node)


def test_rejects_stale_pelvis_imu():
    node, clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.imu_callback(imu(100.00))
        node.imu_callback(imu(100.01))
        clock.now += 0.11

        node.odometry_callback(odometry(100.02))

        assert recorder.messages == []
    finally:
        destroy(node)


def test_rejects_unsynchronized_inputs():
    node, _clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.imu_callback(imu(100.00))
        node.imu_callback(imu(100.01))

        node.odometry_callback(odometry(100.20))

        assert recorder.messages == []
    finally:
        destroy(node)


def test_package_installs_entrypoint_and_calibration():
    package_root = Path(__file__).parents[1]
    setup_text = (
        package_root / "setup.py"
    ).read_text(encoding="utf-8")
    configuration = (
        package_root / "config/physical_body.yaml"
    )

    assert configuration.is_file()
    assert "physical_body_node = " in setup_text

    configuration_text = configuration.read_text(
        encoding="utf-8"
    )
    assert "pelvis_height_m: 0.785" in configuration_text
    assert "imu_stale_after_seconds: 0.10" in configuration_text
    assert "imu_warmup_samples: 20" in configuration_text
    assert "maximum_sync_offset_seconds: 0.05" in (
        configuration_text
    )
