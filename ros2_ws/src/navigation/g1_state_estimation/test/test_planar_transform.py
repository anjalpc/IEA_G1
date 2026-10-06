"""Contract tests for planar Nav2 transform projection."""

import math

import pytest
from nav_msgs.msg import Odometry

from g1_state_estimation.planar_transform import (
    odometry_to_base_footprint_transform,
    odometry_to_pelvis_transform,
)


def quaternion_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def odometry_with_pose(*, roll, pitch, yaw):
    message = Odometry()
    message.header.stamp.sec = 123
    message.header.stamp.nanosec = 456
    message.header.frame_id = "odom"
    message.child_frame_id = "pelvis"

    message.pose.pose.position.x = 1.25
    message.pose.pose.position.y = -0.75
    message.pose.pose.position.z = 1.24

    x, y, z, w = quaternion_from_rpy(
        roll,
        pitch,
        yaw,
    )
    message.pose.pose.orientation.x = x
    message.pose.pose.orientation.y = y
    message.pose.pose.orientation.z = z
    message.pose.pose.orientation.w = w
    return message


def test_projects_pelvis_position_onto_navigation_plane():
    message = odometry_with_pose(
        roll=0.1,
        pitch=-0.2,
        yaw=0.3,
    )

    transform = odometry_to_base_footprint_transform(
        message
    )

    assert transform.header.stamp == message.header.stamp
    assert transform.header.frame_id == "odom"
    assert transform.child_frame_id == "base_footprint"
    assert transform.transform.translation.x == pytest.approx(
        1.25
    )
    assert transform.transform.translation.y == pytest.approx(
        -0.75
    )
    assert transform.transform.translation.z == 0.0


def test_removes_roll_and_pitch_but_preserves_yaw():
    yaw = 0.7
    message = odometry_with_pose(
        roll=0.25,
        pitch=-0.35,
        yaw=yaw,
    )

    transform = odometry_to_base_footprint_transform(
        message
    )
    rotation = transform.transform.rotation

    assert rotation.x == pytest.approx(0.0, abs=1e-12)
    assert rotation.y == pytest.approx(0.0, abs=1e-12)
    assert rotation.z == pytest.approx(
        math.sin(yaw * 0.5),
        abs=1e-12,
    )
    assert rotation.w == pytest.approx(
        math.cos(yaw * 0.5),
        abs=1e-12,
    )


def test_projection_does_not_mutate_validated_odometry():
    message = odometry_with_pose(
        roll=0.1,
        pitch=0.2,
        yaw=0.3,
    )
    original_z = message.pose.pose.position.z
    original_orientation_x = (
        message.pose.pose.orientation.x
    )

    odometry_to_base_footprint_transform(message)

    assert message.pose.pose.position.z == original_z
    assert (
        message.pose.pose.orientation.x
        == original_orientation_x
    )


def test_pelvis_transform_preserves_vertical_body_motion():
    message = odometry_with_pose(
        roll=0.25,
        pitch=-0.35,
        yaw=0.0,
    )

    transform = odometry_to_pelvis_transform(message)

    assert transform.header.stamp == message.header.stamp
    assert transform.header.frame_id == "base_footprint"
    assert transform.child_frame_id == "pelvis"
    assert transform.transform.translation.x == 0.0
    assert transform.transform.translation.y == 0.0
    assert transform.transform.translation.z == pytest.approx(
        1.24
    )

    source_rotation = message.pose.pose.orientation
    result_rotation = transform.transform.rotation
    assert result_rotation.x == pytest.approx(
        source_rotation.x
    )
    assert result_rotation.y == pytest.approx(
        source_rotation.y
    )
    assert result_rotation.z == pytest.approx(
        source_rotation.z
    )
    assert result_rotation.w == pytest.approx(
        source_rotation.w
    )


def test_pelvis_transform_removes_navigation_yaw():
    yaw = 0.7
    message = odometry_with_pose(
        roll=0.0,
        pitch=0.0,
        yaw=yaw,
    )

    transform = odometry_to_pelvis_transform(message)
    rotation = transform.transform.rotation

    assert rotation.x == pytest.approx(0.0, abs=1e-12)
    assert rotation.y == pytest.approx(0.0, abs=1e-12)
    assert rotation.z == pytest.approx(0.0, abs=1e-12)
    assert rotation.w == pytest.approx(1.0, abs=1e-12)
