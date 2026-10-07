"""Build strict simulated MID-360 PointCloud2 messages."""

import math
import struct

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField


_POINT_STEP_BYTES = 12
_FLOAT32_MAX = 3.4028234663852886e38


def lidar_qos():
    """Return the navigation lidar sensor-data QoS contract."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _validated_frame_id(frame_id):
    if (
        not isinstance(frame_id, str)
        or not frame_id.strip()
    ):
        raise ValueError(
            "lidar frame_id must be a non-empty string"
        )
    return frame_id


def _validated_points(points_sensor):
    try:
        raw_points = tuple(points_sensor)
    except TypeError:
        raise TypeError(
            "lidar points must be iterable"
        ) from None

    points = []
    for point in raw_points:
        try:
            values = tuple(point)
        except TypeError:
            raise TypeError(
                "each lidar point must be iterable"
            ) from None

        if len(values) != 3:
            raise ValueError(
                "each lidar point must contain x, y, z"
            )

        try:
            numeric = tuple(
                float(value)
                for value in values
            )
        except (TypeError, ValueError):
            raise TypeError(
                "lidar point values must be numeric"
            ) from None

        if not all(
            math.isfinite(value)
            for value in numeric
        ):
            raise ValueError(
                "lidar point values must be finite"
            )
        if any(
            abs(value) > _FLOAT32_MAX
            for value in numeric
        ):
            raise ValueError(
                "lidar point values must fit float32"
            )
        points.append(numeric)

    return tuple(points)


def _xyz_fields():
    return [
        PointField(
            name="x",
            offset=0,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="y",
            offset=4,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="z",
            offset=8,
            datatype=PointField.FLOAT32,
            count=1,
        ),
    ]


def build_lidar_point_cloud(
    *,
    points_sensor,
    timestamp,
    frame_id,
):
    """Build one finite XYZ cloud in the MID-360 sensor frame."""
    frame_id = _validated_frame_id(
        frame_id
    )
    points = _validated_points(
        points_sensor
    )

    payload = b"".join(
        struct.pack(
            "<fff",
            *point,
        )
        for point in points
    )

    message = PointCloud2()
    message.header.stamp.sec = int(
        timestamp.sec
    )
    message.header.stamp.nanosec = int(
        timestamp.nanosec
    )
    message.header.frame_id = frame_id
    message.height = 1
    message.width = len(points)
    message.fields = _xyz_fields()
    message.is_bigendian = False
    message.point_step = _POINT_STEP_BYTES
    message.row_step = (
        _POINT_STEP_BYTES * len(points)
    )
    message.data = payload
    message.is_dense = True
    return message
