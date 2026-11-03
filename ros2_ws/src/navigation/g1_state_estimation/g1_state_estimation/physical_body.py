"""Calibrated physical-body projection without runtime authority."""

import math

from nav_msgs.msg import Odometry


ODOM_FRAME = "odom"
BASE_FOOTPRINT_FRAME = "base_footprint"
PELVIS_FRAME = "pelvis"
PELVIS_IMU_FRAME = "imu_in_pelvis"


def _rpy(orientation):
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


def _quaternion_from_rpy(roll, pitch, yaw):
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


class PhysicalBodyProjector:
    """Combine planar GLIM pose with calibrated pelvis tilt and height."""

    def __init__(self, *, pelvis_height_m):
        if not math.isfinite(pelvis_height_m):
            raise ValueError("pelvis height must be finite")
        if pelvis_height_m <= 0.0:
            raise ValueError("pelvis height must be positive")

        self.pelvis_height_m = pelvis_height_m

    def project(self, planar_odometry, pelvis_imu):
        """Create one staged odom-to-pelvis sample."""

        if planar_odometry.header.frame_id != ODOM_FRAME:
            raise ValueError("planar odometry frame must be odom")
        if (
            planar_odometry.child_frame_id
            != BASE_FOOTPRINT_FRAME
        ):
            raise ValueError(
                "planar odometry child must be base_footprint"
            )
        if pelvis_imu.header.frame_id != PELVIS_IMU_FRAME:
            raise ValueError(
                "pelvis IMU frame must be imu_in_pelvis"
            )

        imu_roll, imu_pitch, _imu_yaw = _rpy(
            pelvis_imu.orientation
        )
        _planar_roll, _planar_pitch, planar_yaw = _rpy(
            planar_odometry.pose.pose.orientation
        )
        x, y, z, w = _quaternion_from_rpy(
            imu_roll,
            imu_pitch,
            planar_yaw,
        )

        output = Odometry()
        output.header.stamp = planar_odometry.header.stamp
        output.header.frame_id = ODOM_FRAME
        output.child_frame_id = PELVIS_FRAME
        output.pose.pose.position.x = (
            planar_odometry.pose.pose.position.x
        )
        output.pose.pose.position.y = (
            planar_odometry.pose.pose.position.y
        )
        output.pose.pose.position.z = self.pelvis_height_m
        output.pose.pose.orientation.x = x
        output.pose.pose.orientation.y = y
        output.pose.pose.orientation.z = z
        output.pose.pose.orientation.w = w
        output.twist = planar_odometry.twist

        return output
