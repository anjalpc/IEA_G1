"""ROS contract tests for the isolated GLIM health adapter."""

import rclpy

from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu, PointCloud2

from g1_state_estimation.glim_node import GlimHealthNode


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class PublisherRecorder:
    """Collect messages released by the adapter."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def stamp(message, seconds):
    message.header.stamp.sec = int(seconds)
    message.header.stamp.nanosec = int((seconds % 1.0) * 1e9)
    return message


def cloud(seconds=100.0):
    message = stamp(PointCloud2(), seconds)
    message.header.frame_id = "mid360_link"
    return message


def imu(seconds=100.0):
    message = stamp(Imu(), seconds)
    message.header.frame_id = "livox_frame"
    return message


def odometry(seconds=100.0):
    message = stamp(Odometry(), seconds)
    message.header.frame_id = "glim_odom"
    message.child_frame_id = "livox_frame"
    message.pose.pose.orientation.w = 1.0
    return message


def make_node():
    if not rclpy.ok():
        rclpy.init()

    clock = FakeClock()
    node = GlimHealthNode(
        parameter_overrides=[
            Parameter("warmup_samples", value=2),
        ],
        monotonic=clock,
    )
    return node, clock


def destroy(node):
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_uses_isolated_inputs_and_intermediate_output_only():
    node, _clock = make_node()
    try:
        assert node.cloud_subscription.topic_name == (
            "/g1/sensors/lidar/points"
        )
        assert node.imu_subscription.topic_name == (
            "/g1/sensors/lidar/imu_raw"
        )
        assert node.odometry_subscription.topic_name == (
            "/glim_ros/odom_corrected"
        )
        assert node.odometry_publisher.topic_name == (
            "/g1/navigation/glim/odometry_validated"
        )

        subscription_topics = {
            subscription.topic_name
            for subscription in node.subscriptions
        }
        publisher_topics = {
            publisher.topic_name
            for publisher in node.publishers
        }

        assert subscription_topics == {
            "/g1/sensors/lidar/points",
            "/g1/sensors/lidar/imu_raw",
            "/glim_ros/odom_corrected",
        }
        assert "/g1/state/odometry" not in publisher_topics
        assert "/tf" not in publisher_topics
    finally:
        destroy(node)


def test_releases_only_after_all_inputs_and_warmup():
    node, _clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.cloud_callback(cloud())
        node.imu_callback(imu())
        first = odometry(100.0)
        second = odometry(100.1)

        node.odometry_callback(first)
        assert recorder.messages == []

        node.odometry_callback(second)
        assert recorder.messages == [second]
    finally:
        destroy(node)


def test_zero_glim_covariance_does_not_bypass_health_checks():
    node, _clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.cloud_callback(cloud())
        node.imu_callback(imu())
        invalid = odometry(100.0)
        invalid.pose.pose.position.x = float("nan")
        node.odometry_callback(invalid)

        assert recorder.messages == []
    finally:
        destroy(node)


def test_does_not_modify_released_glim_message():
    node, _clock = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.cloud_callback(cloud())
        node.imu_callback(imu())
        node.odometry_callback(odometry(100.0))
        released = odometry(100.1)
        node.odometry_callback(released)

        assert recorder.messages == [released]
        assert list(released.pose.covariance) == [0.0] * 36
        assert list(released.twist.covariance) == [0.0] * 36
    finally:
        destroy(node)
