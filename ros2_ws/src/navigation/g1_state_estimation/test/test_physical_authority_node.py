"""ROS contract tests for physical odometry authority."""

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    ReliabilityPolicy,
)

from g1_state_estimation.physical_authority_node import (
    ODOMETRY_TOPIC,
    STAGED_ODOMETRY_TOPIC,
    PhysicalAuthorityNode,
)


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class PublisherRecorder:
    """Record canonical odometry."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class TransformRecorder:
    """Record authoritative transforms."""

    def __init__(self):
        self.transforms = []

    def sendTransform(self, transforms):
        self.transforms.extend(transforms)


def odometry(seconds, *, x=0.0, yaw=0.0):
    """Build one staged physical-body sample."""

    message = Odometry()
    message.header.stamp.sec = int(seconds)
    message.header.stamp.nanosec = int(
        round((seconds - int(seconds)) * 1e9)
    )
    message.header.frame_id = "odom"
    message.child_frame_id = "pelvis"
    message.pose.pose.position.x = x
    message.pose.pose.position.z = 0.785
    message.pose.pose.orientation.z = math.sin(yaw * 0.5)
    message.pose.pose.orientation.w = math.cos(yaw * 0.5)
    return message


def make_node():
    """Create an authority node with short deterministic warmup."""

    if not rclpy.ok():
        rclpy.init()

    clock = FakeClock()
    node = PhysicalAuthorityNode(
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


def test_uses_only_staged_input_canonical_output_and_tf():
    node, _clock = make_node()
    try:
        assert node.odometry_subscription.topic_name == (
            STAGED_ODOMETRY_TOPIC
        )
        assert node.odometry_publisher.topic_name == ODOMETRY_TOPIC
        assert node.input_qos.depth == 1
        assert (
            node.input_qos.reliability
            == ReliabilityPolicy.RELIABLE
        )
        assert (
            node.input_qos.durability
            == DurabilityPolicy.VOLATILE
        )
        assert node.output_qos.depth == 1
        assert (
            node.output_qos.reliability
            == ReliabilityPolicy.RELIABLE
        )
    finally:
        destroy(node)


def test_releases_canonical_odometry_and_two_tf_after_warmup():
    node, _clock = make_node()
    publisher = PublisherRecorder()
    transforms = TransformRecorder()
    node.odometry_publisher = publisher
    node.transform_broadcaster = transforms
    try:
        node.odometry_callback(odometry(100.0))
        assert publisher.messages == []
        assert transforms.transforms == []

        source = odometry(100.1, x=0.01, yaw=0.02)
        node.odometry_callback(source)

        assert len(publisher.messages) == 1
        canonical = publisher.messages[0]
        assert canonical is not source
        assert canonical.header.frame_id == "odom"
        assert canonical.child_frame_id == "base_footprint"
        assert canonical.pose.pose.position.x == 0.01
        assert canonical.pose.pose.position.y == 0.0
        assert canonical.pose.pose.position.z == 0.0
        assert canonical.pose.pose.orientation.x == 0.0
        assert canonical.pose.pose.orientation.y == 0.0
        assert canonical.pose.pose.orientation.z == (
            math.sin(0.01)
        )
        assert canonical.pose.pose.orientation.w == (
            math.cos(0.01)
        )
        assert list(canonical.pose.covariance) == [0.0] * 36

        assert source.child_frame_id == "pelvis"
        assert source.pose.pose.position.z == 0.785
        assert list(source.pose.covariance) == [0.0] * 36
        assert len(transforms.transforms) == 2

        base = transforms.transforms[0]
        assert base.header.frame_id == "odom"
        assert base.child_frame_id == "base_footprint"

        pelvis = transforms.transforms[1]
        assert pelvis.header.frame_id == "base_footprint"
        assert pelvis.child_frame_id == "pelvis"
        assert pelvis.transform.translation.z == 0.785
    finally:
        destroy(node)


def test_invalid_step_publishes_neither_odometry_nor_tf():
    node, _clock = make_node()
    publisher = PublisherRecorder()
    transforms = TransformRecorder()
    node.odometry_publisher = publisher
    node.transform_broadcaster = transforms
    try:
        node.odometry_callback(odometry(100.0))
        node.odometry_callback(odometry(100.1, x=0.01))
        assert len(publisher.messages) == 1
        assert len(transforms.transforms) == 2

        node.odometry_callback(odometry(100.2, x=0.30))

        assert len(publisher.messages) == 1
        assert len(transforms.transforms) == 2
    finally:
        destroy(node)


def test_wires_rate_and_gap_parameters_into_boundary():
    if not rclpy.ok():
        rclpy.init()

    node = PhysicalAuthorityNode(
        parameter_overrides=[
            Parameter("maximum_translation_rate", value=0.4),
            Parameter(
                "maximum_rotation_rate_degrees",
                value=50.0,
            ),
            Parameter(
                "maximum_motion_gap_seconds",
                value=1.5,
            ),
        ],
        monotonic=FakeClock(),
    )
    try:
        limits = node.boundary.limits
        assert limits.maximum_translation_rate == 0.4
        assert limits.maximum_rotation_rate == math.radians(50.0)
        assert limits.maximum_motion_gap_seconds == 1.5
    finally:
        destroy(node)
