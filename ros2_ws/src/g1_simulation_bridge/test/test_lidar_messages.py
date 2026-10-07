"""Contract tests for simulated MID-360 PointCloud2 output."""

import math
import struct

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointField
import pytest

from g1_simulation_bridge.lidar_messages import (
    build_lidar_point_cloud,
    lidar_qos,
)
from g1_simulation_bridge.simulation_timing import (
    SimulationTimestamp,
)


TIMESTAMP = SimulationTimestamp(
    sec=17,
    nanosec=250000000,
)


def test_builds_strict_xyz_float32_point_cloud():
    message = build_lidar_point_cloud(
        points_sensor=(
            (1.0, 2.0, 3.0),
            (-4.5, 5.25, 6.75),
        ),
        timestamp=TIMESTAMP,
        frame_id="mid360_link",
    )

    assert message.header.stamp.sec == 17
    assert message.header.stamp.nanosec == 250000000
    assert message.header.frame_id == "mid360_link"

    assert message.height == 1
    assert message.width == 2
    assert message.is_bigendian is False
    assert message.point_step == 12
    assert message.row_step == 24
    assert message.is_dense is True

    assert [
        (
            field.name,
            field.offset,
            field.datatype,
            field.count,
        )
        for field in message.fields
    ] == [
        ("x", 0, PointField.FLOAT32, 1),
        ("y", 4, PointField.FLOAT32, 1),
        ("z", 8, PointField.FLOAT32, 1),
    ]

    assert struct.unpack(
        "<ffffff",
        bytes(message.data),
    ) == pytest.approx(
        (
            1.0,
            2.0,
            3.0,
            -4.5,
            5.25,
            6.75,
        )
    )


def test_builds_empty_cloud_without_fabricated_points():
    message = build_lidar_point_cloud(
        points_sensor=(),
        timestamp=TIMESTAMP,
        frame_id="mid360_link",
    )

    assert message.height == 1
    assert message.width == 0
    assert message.point_step == 12
    assert message.row_step == 0
    assert bytes(message.data) == b""
    assert message.is_dense is True


@pytest.mark.parametrize(
    "points",
    [
        ((1.0, 2.0),),
        ((1.0, 2.0, 3.0, 4.0),),
        ((math.nan, 0.0, 0.0),),
        ((math.inf, 0.0, 0.0),),
        ((1.0e300, 0.0, 0.0),),
        (("private", 0.0, 0.0),),
    ],
)
def test_rejects_invalid_point_values(points):
    with pytest.raises(
        (TypeError, ValueError),
    ):
        build_lidar_point_cloud(
            points_sensor=points,
            timestamp=TIMESTAMP,
            frame_id="mid360_link",
        )


@pytest.mark.parametrize(
    "frame_id",
    [
        "",
        " ",
        123,
    ],
)
def test_rejects_invalid_frame_id(frame_id):
    with pytest.raises(
        (TypeError, ValueError),
    ):
        build_lidar_point_cloud(
            points_sensor=(),
            timestamp=TIMESTAMP,
            frame_id=frame_id,
        )


def test_uses_best_effort_volatile_depth_one_qos():
    qos = lidar_qos()

    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 1
    assert qos.reliability == (
        ReliabilityPolicy.BEST_EFFORT
    )
    assert qos.durability == (
        DurabilityPolicy.VOLATILE
    )
