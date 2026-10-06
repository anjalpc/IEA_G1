"""Contract tests for raw navigation-state ROS messages."""

import pytest

from g1_simulation_bridge.simulation_state import (
    SimulationStateSample,
)
from g1_simulation_bridge.simulation_timing import (
    SimulationTimestamp,
)
from g1_simulation_bridge.state_messages import (
    build_clock_message,
    build_imu_message,
    build_odometry_message,
)


TIMESTAMP = SimulationTimestamp(sec=12, nanosec=345678901)
COVARIANCE_VARIANCE = 1e-6


def sample():
    return SimulationStateSample(
        position_xyz=(1.0, 2.0, 3.0),
        orientation_xyzw=(0.1, 0.2, 0.3, 0.9),
        linear_velocity_xyz=(0.4, 0.5, 0.6),
        angular_velocity_xyz=(0.7, 0.8, 0.9),
        imu_orientation_xyzw=(0.11, 0.22, 0.33, 0.88),
        imu_angular_velocity_xyz=(1.1, 1.2, 1.3),
        imu_linear_acceleration_xyz=(2.1, 2.2, 2.3),
    )


def diagonal(width):
    values = [0.0] * (width * width)
    for index in range(width):
        values[index * width + index] = (
            COVARIANCE_VARIANCE
        )
    return values


def test_builds_clock_from_simulation_timestamp():
    message = build_clock_message(TIMESTAMP)

    assert message.clock.sec == 12
    assert message.clock.nanosec == 345678901


def test_builds_pelvis_frame_imu_message():
    message = build_imu_message(
        sample(),
        TIMESTAMP,
        frame_id="pelvis",
        covariance_variance=COVARIANCE_VARIANCE,
    )

    assert message.header.stamp.sec == 12
    assert message.header.stamp.nanosec == 345678901
    assert message.header.frame_id == "pelvis"

    assert (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) == pytest.approx((0.11, 0.22, 0.33, 0.88))

    assert (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    ) == pytest.approx((1.1, 1.2, 1.3))

    assert (
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
    ) == pytest.approx((2.1, 2.2, 2.3))

    assert list(message.orientation_covariance) == diagonal(3)
    assert list(message.angular_velocity_covariance) == diagonal(3)
    assert list(message.linear_acceleration_covariance) == diagonal(3)


def test_builds_odom_to_pelvis_odometry_message():
    message = build_odometry_message(
        sample(),
        TIMESTAMP,
        odom_frame_id="odom",
        child_frame_id="pelvis",
        covariance_variance=COVARIANCE_VARIANCE,
    )

    assert message.header.stamp.sec == 12
    assert message.header.stamp.nanosec == 345678901
    assert message.header.frame_id == "odom"
    assert message.child_frame_id == "pelvis"

    assert (
        message.pose.pose.position.x,
        message.pose.pose.position.y,
        message.pose.pose.position.z,
    ) == pytest.approx((1.0, 2.0, 3.0))

    assert (
        message.pose.pose.orientation.x,
        message.pose.pose.orientation.y,
        message.pose.pose.orientation.z,
        message.pose.pose.orientation.w,
    ) == pytest.approx((0.1, 0.2, 0.3, 0.9))

    assert list(message.pose.covariance) == diagonal(6)


def test_places_body_frame_velocities_in_odometry_twist():
    message = build_odometry_message(
        sample(),
        TIMESTAMP,
        odom_frame_id="odom",
        child_frame_id="pelvis",
        covariance_variance=COVARIANCE_VARIANCE,
    )

    assert (
        message.twist.twist.linear.x,
        message.twist.twist.linear.y,
        message.twist.twist.linear.z,
    ) == pytest.approx((0.4, 0.5, 0.6))

    assert (
        message.twist.twist.angular.x,
        message.twist.twist.angular.y,
        message.twist.twist.angular.z,
    ) == pytest.approx((0.7, 0.8, 0.9))

    assert list(message.twist.covariance) == diagonal(6)
