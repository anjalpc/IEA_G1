"""Convert qualified inverted MID-360 odometry to a planar robot frame."""

import math

from nav_msgs.msg import Odometry


ODOM_FRAME = "odom"
BASE_FOOTPRINT_FRAME = "base_footprint"

# The physical MID-360 is mounted upside down. Its X axis remains forward,
# while Y and Z are reversed relative to the robot navigation frame.
ROBOT_FROM_SENSOR = (1.0, 0.0, 0.0, 0.0)


def _multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _conjugate(quaternion):
    return (
        -quaternion[0],
        -quaternion[1],
        -quaternion[2],
        quaternion[3],
    )


def _normalize(quaternion):
    norm = math.sqrt(
        sum(component * component for component in quaternion)
    )
    return tuple(component / norm for component in quaternion)


def _rotate(quaternion, vector):
    rotated = _multiply(
        _multiply(
            quaternion,
            (vector[0], vector[1], vector[2], 0.0),
        ),
        _conjugate(quaternion),
    )
    return rotated[:3]


def _position(message):
    value = message.pose.pose.position
    return (value.x, value.y, value.z)


def _orientation(message):
    value = message.pose.pose.orientation
    return _normalize((value.x, value.y, value.z, value.w))


def _yaw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class GlimPlanarProjector:
    """Create start-relative planar odometry from qualified GLIM poses."""

    def __init__(self):
        self._origin_position = None
        self._origin_orientation = None

    def project(self, source):
        """Project one qualified GLIM sample without modifying it."""

        position = _position(source)
        orientation = _orientation(source)

        if self._origin_position is None:
            self._origin_position = position
            self._origin_orientation = orientation

        origin_inverse = _conjugate(
            self._origin_orientation
        )
        world_delta = tuple(
            current - origin
            for current, origin in zip(
                position,
                self._origin_position,
            )
        )
        sensor_translation = _rotate(
            origin_inverse,
            world_delta,
        )
        robot_translation = _rotate(
            ROBOT_FROM_SENSOR,
            sensor_translation,
        )

        sensor_rotation = _multiply(
            origin_inverse,
            orientation,
        )
        robot_rotation = _multiply(
            _multiply(
                ROBOT_FROM_SENSOR,
                sensor_rotation,
            ),
            _conjugate(ROBOT_FROM_SENSOR),
        )
        yaw = _yaw(robot_rotation)

        output = Odometry()
        output.header.stamp = source.header.stamp
        output.header.frame_id = ODOM_FRAME
        output.child_frame_id = BASE_FOOTPRINT_FRAME
        output.pose.pose.position.x = robot_translation[0]
        output.pose.pose.position.y = robot_translation[1]
        output.pose.pose.position.z = 0.0
        output.pose.pose.orientation.z = math.sin(yaw * 0.5)
        output.pose.pose.orientation.w = math.cos(yaw * 0.5)

        output.twist.twist.linear.x = (
            source.twist.twist.linear.x
        )
        output.twist.twist.linear.y = (
            -source.twist.twist.linear.y
        )
        output.twist.twist.angular.z = (
            -source.twist.twist.angular.z
        )

        return output
