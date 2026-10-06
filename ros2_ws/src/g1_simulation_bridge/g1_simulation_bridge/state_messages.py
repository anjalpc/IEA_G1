"""Build raw navigation ROS messages from simulation state."""

from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu

from g1_simulation_bridge.simulation_state import (
    diagonal_covariance,
)


def _set_stamp(target, timestamp):
    target.sec = int(timestamp.sec)
    target.nanosec = int(timestamp.nanosec)


def _set_vector(target, values):
    target.x, target.y, target.z = values


def _set_quaternion(target, values):
    target.x, target.y, target.z, target.w = values


def build_clock_message(timestamp):
    """Build one `/clock` message from simulation time."""

    message = Clock()
    _set_stamp(message.clock, timestamp)
    return message


def build_imu_message(
    sample,
    timestamp,
    *,
    frame_id,
    covariance_variance,
):
    """Build one raw IMU message in the configured body frame."""

    covariance = diagonal_covariance(
        width=3,
        variance=covariance_variance,
    )

    message = Imu()
    _set_stamp(message.header.stamp, timestamp)
    message.header.frame_id = frame_id

    _set_quaternion(
        message.orientation,
        sample.imu_orientation_xyzw,
    )
    _set_vector(
        message.angular_velocity,
        sample.imu_angular_velocity_xyz,
    )
    _set_vector(
        message.linear_acceleration,
        sample.imu_linear_acceleration_xyz,
    )

    message.orientation_covariance = covariance
    message.angular_velocity_covariance = covariance
    message.linear_acceleration_covariance = covariance

    return message


def build_odometry_message(
    sample,
    timestamp,
    *,
    odom_frame_id,
    child_frame_id,
    covariance_variance,
):
    """Build raw odometry with body-frame twist values."""

    covariance = diagonal_covariance(
        width=6,
        variance=covariance_variance,
    )

    message = Odometry()
    _set_stamp(message.header.stamp, timestamp)
    message.header.frame_id = odom_frame_id
    message.child_frame_id = child_frame_id

    _set_vector(
        message.pose.pose.position,
        sample.position_xyz,
    )
    _set_quaternion(
        message.pose.pose.orientation,
        sample.orientation_xyzw,
    )
    _set_vector(
        message.twist.twist.linear,
        sample.linear_velocity_xyz,
    )
    _set_vector(
        message.twist.twist.angular,
        sample.angular_velocity_xyz,
    )

    message.pose.covariance = covariance
    message.twist.covariance = covariance

    return message
