"""Contract tests for guarded GLIM planar-frame conversion."""

import math

import pytest
from nav_msgs.msg import Odometry

from g1_state_estimation.glim_planar import GlimPlanarProjector


def odometry(*, x=0.0, y=0.0, z=0.0, yaw=0.0):
    """Build one isolated GLIM odometry sample."""

    message = Odometry()
    message.header.stamp.sec = 123
    message.header.frame_id = "glim_odom"
    message.child_frame_id = "livox_frame"
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.position.z = z
    message.pose.pose.orientation.z = math.sin(yaw * 0.5)
    message.pose.pose.orientation.w = math.cos(yaw * 0.5)
    return message


def yaw_from(message):
    """Return planar yaw from one output quaternion."""

    orientation = message.pose.pose.orientation
    return math.atan2(
        2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        ),
        1.0 - 2.0 * (
            orientation.y * orientation.y
            + orientation.z * orientation.z
        ),
    )


def test_first_sample_defines_zero_planar_origin():
    projector = GlimPlanarProjector()

    result = projector.project(
        odometry(x=4.0, y=-2.0, z=1.0, yaw=0.7)
    )

    assert result.header.frame_id == "odom"
    assert result.child_frame_id == "base_footprint"
    assert result.pose.pose.position.x == pytest.approx(0.0)
    assert result.pose.pose.position.y == pytest.approx(0.0)
    assert result.pose.pose.position.z == pytest.approx(0.0)
    assert yaw_from(result) == pytest.approx(0.0)


def test_robot_forward_remains_positive_x_without_scale_correction():
    projector = GlimPlanarProjector()
    projector.project(odometry())

    result = projector.project(odometry(x=0.5))

    assert result.pose.pose.position.x == pytest.approx(0.5)
    assert result.pose.pose.position.y == pytest.approx(0.0)


def test_inverted_sensor_left_turn_becomes_positive_robot_yaw():
    projector = GlimPlanarProjector()
    projector.project(odometry())

    result = projector.project(
        odometry(yaw=-math.pi / 2.0)
    )

    assert yaw_from(result) == pytest.approx(math.pi / 2.0)


def test_inverted_sensor_lateral_axis_is_corrected():
    projector = GlimPlanarProjector()
    projector.project(odometry())

    result = projector.project(odometry(y=-0.25))

    assert result.pose.pose.position.x == pytest.approx(0.0)
    assert result.pose.pose.position.y == pytest.approx(0.25)


def test_projection_preserves_timestamp_and_unknown_covariance():
    projector = GlimPlanarProjector()
    source = odometry()

    result = projector.project(source)

    assert result.header.stamp == source.header.stamp
    assert list(result.pose.covariance) == [0.0] * 36
    assert list(result.twist.covariance) == [0.0] * 36


def test_projection_does_not_modify_validated_input():
    projector = GlimPlanarProjector()
    source = odometry(x=1.0, y=2.0, z=3.0, yaw=-0.4)

    projector.project(source)

    assert source.header.frame_id == "glim_odom"
    assert source.child_frame_id == "livox_frame"
    assert source.pose.pose.position.x == pytest.approx(1.0)
    assert source.pose.pose.position.y == pytest.approx(2.0)
    assert source.pose.pose.position.z == pytest.approx(3.0)


def test_initial_heading_is_removed_from_forward_translation():
    projector = GlimPlanarProjector()
    projector.project(
        odometry(yaw=math.pi / 2.0)
    )

    result = projector.project(
        odometry(y=0.5, yaw=math.pi / 2.0)
    )

    assert result.pose.pose.position.x == pytest.approx(0.5)
    assert result.pose.pose.position.y == pytest.approx(0.0)
    assert yaw_from(result) == pytest.approx(0.0)


def test_twist_uses_corrected_robot_axes():
    projector = GlimPlanarProjector()
    source = odometry()
    source.twist.twist.linear.x = 0.4
    source.twist.twist.linear.y = -0.2
    source.twist.twist.linear.z = 0.3
    source.twist.twist.angular.x = 0.1
    source.twist.twist.angular.y = -0.1
    source.twist.twist.angular.z = -0.6

    result = projector.project(source)

    assert result.twist.twist.linear.x == pytest.approx(0.4)
    assert result.twist.twist.linear.y == pytest.approx(0.2)
    assert result.twist.twist.linear.z == pytest.approx(0.0)
    assert result.twist.twist.angular.x == pytest.approx(0.0)
    assert result.twist.twist.angular.y == pytest.approx(0.0)
    assert result.twist.twist.angular.z == pytest.approx(0.6)
