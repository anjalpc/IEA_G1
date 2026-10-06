"""Planar transform projection for Nav2."""

import math

from geometry_msgs.msg import TransformStamped


BASE_FOOTPRINT_FRAME = "base_footprint"


def odometry_to_base_footprint_transform(odometry):
    """Project validated pelvis odometry onto the navigation plane."""

    orientation = odometry.pose.pose.orientation
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

    transform = TransformStamped()
    transform.header.stamp.sec = odometry.header.stamp.sec
    transform.header.stamp.nanosec = (
        odometry.header.stamp.nanosec
    )
    transform.header.frame_id = odometry.header.frame_id
    transform.child_frame_id = BASE_FOOTPRINT_FRAME

    transform.transform.translation.x = (
        odometry.pose.pose.position.x
    )
    transform.transform.translation.y = (
        odometry.pose.pose.position.y
    )
    transform.transform.translation.z = 0.0

    transform.transform.rotation.x = 0.0
    transform.transform.rotation.y = 0.0
    transform.transform.rotation.z = math.sin(yaw * 0.5)
    transform.transform.rotation.w = math.cos(yaw * 0.5)

    return transform
