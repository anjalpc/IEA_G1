"""ROS-facing tests for validated state publication."""

import pytest
import rclpy

from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu

from g1_state_estimation.node import StateEstimationNode


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class PublisherRecorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class TransformRecorder:
    def __init__(self):
        self.transforms = []

    def sendTransform(self, transform):
        self.transforms.append(transform)


def populated_covariance(size):
    covariance = [0.0] * size
    width = 3 if size == 9 else 6
    for index in range(width):
        covariance[index * width + index] = 0.01
    return covariance


def valid_imu():
    message = Imu()
    message.header.stamp.sec = 123
    message.header.frame_id = "pelvis"
    message.orientation.w = 1.0
    message.orientation_covariance = populated_covariance(9)
    message.angular_velocity_covariance = (
        populated_covariance(9)
    )
    message.linear_acceleration_covariance = (
        populated_covariance(9)
    )
    return message


def valid_odometry():
    message = Odometry()
    message.header.stamp.sec = 123
    message.header.frame_id = "odom"
    message.child_frame_id = "pelvis"
    message.pose.pose.orientation.w = 1.0
    message.pose.covariance = populated_covariance(36)
    message.twist.covariance = populated_covariance(36)
    return message


@pytest.fixture
def node():
    if not rclpy.ok():
        rclpy.init()

    clock = FakeClock()
    instance = StateEstimationNode(
        parameter_overrides=[
            Parameter(
                "stale_after_seconds",
                value=0.5,
            ),
        ],
        monotonic=clock,
    )

    yield instance, clock

    instance.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_creates_raw_inputs_and_validated_output(node):
    instance, _clock = node
    topics = {
        name
        for name, _types
        in instance.get_topic_names_and_types()
    }

    assert "/g1/state/imu_raw" in topics
    assert "/g1/state/odometry_raw" in topics
    assert "/g1/state/odometry" in topics
    assert "/tf" in topics


def test_uses_bounded_volatile_state_stream_qos(node):
    instance, _clock = node

    assert instance.input_qos.depth == 1
    assert (
        instance.input_qos.reliability
        == ReliabilityPolicy.BEST_EFFORT
    )
    assert (
        instance.input_qos.durability
        == DurabilityPolicy.VOLATILE
    )

    assert instance.output_qos.depth == 1
    assert (
        instance.output_qos.reliability
        == ReliabilityPolicy.RELIABLE
    )
    assert (
        instance.output_qos.durability
        == DurabilityPolicy.VOLATILE
    )


def test_publishes_only_after_both_streams_are_valid(node):
    instance, _clock = node
    recorder = PublisherRecorder()
    instance.odometry_publisher = recorder

    first_odometry = valid_odometry()
    instance.odometry_callback(first_odometry)

    assert recorder.messages == []

    instance.imu_callback(valid_imu())
    released_odometry = valid_odometry()
    instance.odometry_callback(released_odometry)

    assert recorder.messages == [released_odometry]


def test_invalid_imu_blocks_odometry_and_tf_publication(node):
    instance, _clock = node
    odometry_recorder = PublisherRecorder()
    transform_recorder = TransformRecorder()
    instance.odometry_publisher = odometry_recorder
    instance.transform_broadcaster = transform_recorder

    imu = valid_imu()
    imu.orientation.w = 2.0

    instance.imu_callback(imu)
    instance.odometry_callback(valid_odometry())

    assert odometry_recorder.messages == []
    assert transform_recorder.transforms == []


def test_stale_imu_blocks_fresh_odometry_and_tf(node):
    instance, clock = node
    odometry_recorder = PublisherRecorder()
    transform_recorder = TransformRecorder()
    instance.odometry_publisher = odometry_recorder
    instance.transform_broadcaster = transform_recorder

    instance.imu_callback(valid_imu())
    clock.now += 0.501
    instance.odometry_callback(valid_odometry())

    assert odometry_recorder.messages == []
    assert transform_recorder.transforms == []


def test_broadcasts_planar_tf_only_for_released_odometry(node):
    instance, _clock = node
    odometry_recorder = PublisherRecorder()
    transform_recorder = TransformRecorder()
    instance.odometry_publisher = odometry_recorder
    instance.transform_broadcaster = transform_recorder

    blocked_odometry = valid_odometry()
    instance.odometry_callback(blocked_odometry)

    assert odometry_recorder.messages == []
    assert transform_recorder.transforms == []

    instance.imu_callback(valid_imu())
    released_odometry = valid_odometry()
    released_odometry.pose.pose.position.x = 1.25
    released_odometry.pose.pose.position.y = -0.75
    released_odometry.pose.pose.position.z = 1.24
    instance.odometry_callback(released_odometry)

    assert odometry_recorder.messages == [released_odometry]
    assert len(transform_recorder.transforms) == 1

    transform = transform_recorder.transforms[0]
    assert transform.header.stamp == released_odometry.header.stamp
    assert transform.header.frame_id == "odom"
    assert transform.child_frame_id == "base_footprint"
    assert transform.transform.translation.x == 1.25
    assert transform.transform.translation.y == -0.75
    assert transform.transform.translation.z == 0.0
