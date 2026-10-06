"""Transform projection from validated pelvis odometry."""

import math

from geometry_msgs.msg import TransformStamped


BASE_FOOTPRINT_FRAME = "base_footprint"


def _yaw_from_orientation(orientation):
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


def _new_transform(odometry, parent_frame, child_frame):
    transform = TransformStamped()
    transform.header.stamp.sec = odometry.header.stamp.sec
    transform.header.stamp.nanosec = (
        odometry.header.stamp.nanosec
    )
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    return transform


def odometry_to_base_footprint_transform(odometry):
    """Project validated pelvis odometry onto the navigation plane."""

    yaw = _yaw_from_orientation(
        odometry.pose.pose.orientation
    )
    transform = _new_transform(
        odometry,
        odometry.header.frame_id,
        BASE_FOOTPRINT_FRAME,
    )

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


def odometry_to_pelvis_transform(odometry):
    """Express the validated pelvis pose relative to base_footprint."""

    orientation = odometry.pose.pose.orientation
    yaw = _yaw_from_orientation(orientation)

    inverse_base_z = -math.sin(yaw * 0.5)
    inverse_base_w = math.cos(yaw * 0.5)

    relative_x = (
        inverse_base_w * orientation.x
        - inverse_base_z * orientation.y
    )
    relative_y = (
        inverse_base_w * orientation.y
        + inverse_base_z * orientation.x
    )
    relative_z = (
        inverse_base_w * orientation.z
        + inverse_base_z * orientation.w
    )
    relative_w = (
        inverse_base_w * orientation.w
        - inverse_base_z * orientation.z
    )

    transform = _new_transform(
        odometry,
        BASE_FOOTPRINT_FRAME,
        odometry.child_frame_id,
    )
    transform.transform.translation.x = 0.0
    transform.transform.translation.y = 0.0
    transform.transform.translation.z = (
        odometry.pose.pose.position.z
    )
    transform.transform.rotation.x = relative_x
    transform.transform.rotation.y = relative_y
    transform.transform.rotation.z = relative_z
    transform.transform.rotation.w = relative_w

    return transform
