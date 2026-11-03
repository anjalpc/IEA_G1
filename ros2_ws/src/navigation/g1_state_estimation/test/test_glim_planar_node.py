"""ROS contract tests for staged planar GLIM odometry."""

import rclpy
from nav_msgs.msg import Odometry

from g1_state_estimation.glim_planar_node import GlimPlanarNode


class PublisherRecorder:
    """Collect staged planar odometry."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def odometry(x=0.0):
    """Build one health-qualified GLIM sample."""

    message = Odometry()
    message.header.stamp.sec = 123
    message.header.frame_id = "glim_odom"
    message.child_frame_id = "livox_frame"
    message.pose.pose.position.x = x
    message.pose.pose.orientation.w = 1.0
    return message


def make_node():
    if not rclpy.ok():
        rclpy.init()
    return GlimPlanarNode()


def destroy(node):
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_uses_only_validated_input_and_staged_output():
    node = make_node()
    try:
        assert node.odometry_subscription.topic_name == (
            "/g1/navigation/glim/odometry_validated"
        )
        assert node.odometry_publisher.topic_name == (
            "/g1/navigation/glim/odometry_planar"
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
            "/g1/navigation/glim/odometry_validated"
        }
        assert "/g1/navigation/glim/odometry_planar" in publisher_topics
        assert "/g1/state/odometry" not in publisher_topics
        assert "/tf" not in publisher_topics
    finally:
        destroy(node)


def test_publishes_start_relative_planar_samples():
    node = make_node()
    recorder = PublisherRecorder()
    node.odometry_publisher = recorder
    try:
        node.odometry_callback(odometry())
        node.odometry_callback(odometry(x=0.5))

        assert len(recorder.messages) == 2
        first, second = recorder.messages
        assert first.header.frame_id == "odom"
        assert first.child_frame_id == "base_footprint"
        assert first.pose.pose.position.x == 0.0
        assert second.pose.pose.position.x == 0.5
    finally:
        destroy(node)
