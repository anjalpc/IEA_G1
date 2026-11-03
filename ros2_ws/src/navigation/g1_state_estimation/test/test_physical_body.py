"""Contract tests for calibrated physical-body staging."""

import math

import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from g1_state_estimation.physical_body import PhysicalBodyProjector


def quaternion_from_rpy(roll, pitch, yaw):
    """Return one ROS-order quaternion."""

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


def planar_odometry(*, x=0.0, y=0.0, yaw=0.0):
    """Build one qualified planar GLIM sample."""

    message = Odometry()
    message.header.stamp.sec = 123
    message.header.stamp.nanosec = 456
    message.header.frame_id = "odom"
    message.child_frame_id = "base_footprint"
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.z = math.sin(yaw * 0.5)
    message.pose.pose.orientation.w = math.cos(yaw * 0.5)
    return message


def pelvis_imu(*, roll=0.0, pitch=0.0, yaw=0.0):
    """Build one measured pelvis-IMU sample."""

    message = Imu()
    message.header.frame_id = "imu_in_pelvis"
    x, y, z, w = quaternion_from_rpy(roll, pitch, yaw)
    message.orientation.x = x
    message.orientation.y = y
    message.orientation.z = z
    message.orientation.w = w
    return message


def rpy_from_orientation(orientation):
    """Return roll, pitch, and yaw from one quaternion."""

    roll = math.atan2(
        2.0 * (
            orientation.w * orientation.x
            + orientation.y * orientation.z
        ),
        1.0 - 2.0 * (
            orientation.x * orientation.x
            + orientation.y * orientation.y
        ),
    )
    pitch = math.asin(
        max(
            -1.0,
            min(
                1.0,
                2.0 * (
                    orientation.w * orientation.y
                    - orientation.z * orientation.x
                ),
            ),
        )
    )
    yaw = math.atan2(
        2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        ),
        1.0 - 2.0 * (
            orientation.y * orientation.y
            + orientation.z * orientation.z
        ),
    )
    return roll, pitch, yaw


def test_uses_measured_planar_pose_and_calibrated_pelvis_height():
    projector = PhysicalBodyProjector(
        pelvis_height_m=0.785,
    )

    result = projector.project(
        planar_odometry(x=1.2, y=-0.4, yaw=0.3),
        pelvis_imu(),
    )

    assert result.header.frame_id == "odom"
    assert result.child_frame_id == "pelvis"
    assert result.header.stamp.sec == 123
    assert result.header.stamp.nanosec == 456
    assert result.pose.pose.position.x == pytest.approx(1.2)
    assert result.pose.pose.position.y == pytest.approx(-0.4)
    assert result.pose.pose.position.z == pytest.approx(0.785)


def test_uses_glim_yaw_and_pelvis_imu_tilt_only():
    projector = PhysicalBodyProjector(
        pelvis_height_m=0.785,
    )

    result = projector.project(
        planar_odometry(yaw=0.6),
        pelvis_imu(roll=0.12, pitch=-0.08, yaw=-1.1),
    )

    roll, pitch, yaw = rpy_from_orientation(
        result.pose.pose.orientation
    )
    assert roll == pytest.approx(0.12)
    assert pitch == pytest.approx(-0.08)
    assert yaw == pytest.approx(0.6)


def test_preserves_unknown_covariance_without_invention():
    projector = PhysicalBodyProjector(
        pelvis_height_m=0.785,
    )

    result = projector.project(
        planar_odometry(),
        pelvis_imu(),
    )

    assert list(result.pose.covariance) == [0.0] * 36
    assert list(result.twist.covariance) == [0.0] * 36


@pytest.mark.parametrize(
    ("odometry_frame", "odometry_child", "imu_frame"),
    [
        ("map", "base_footprint", "imu_in_pelvis"),
        ("odom", "pelvis", "imu_in_pelvis"),
        ("odom", "base_footprint", "wrong_imu"),
    ],
)
def test_rejects_unexpected_input_frames(
    odometry_frame,
    odometry_child,
    imu_frame,
):
    projector = PhysicalBodyProjector(
        pelvis_height_m=0.785,
    )
    odometry = planar_odometry()
    odometry.header.frame_id = odometry_frame
    odometry.child_frame_id = odometry_child
    imu = pelvis_imu()
    imu.header.frame_id = imu_frame

    with pytest.raises(ValueError):
        projector.project(odometry, imu)
