"""Contract tests for scheduled simulated lidar publication."""

import math
from types import SimpleNamespace

from rclpy.qos import (
    DurabilityPolicy,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2

from g1_simulation_bridge.lidar_publisher import (
    LIDAR_POINTS_TOPIC,
    SimulationLidarPublisher,
)
from g1_simulation_bridge.simulation_timing import (
    LidarPublicationSchedule,
)


class FakeRosPublisher:
    def __init__(self, message_type, topic, qos):
        self.message_type = message_type
        self.topic = topic
        self.qos = qos
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self):
        self.publishers = {}

    def create_publisher(
        self,
        message_type,
        topic,
        qos,
    ):
        publisher = FakeRosPublisher(
            message_type,
            topic,
            qos,
        )
        self.publishers[topic] = publisher
        return publisher


class FakeRayAdapter:
    def __init__(self):
        self.torso_pose_calls = 0
        self.cast_calls = []

    def torso_pose(self):
        self.torso_pose_calls += 1
        return SimpleNamespace(
            position_xyz=(1.0, 2.0, 3.0),
            orientation_wxyz=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        )

    def cast(
        self,
        *,
        origin_world,
        directions_world,
    ):
        self.cast_calls.append(
            (
                origin_world,
                directions_world,
            )
        )
        return (
            SimpleNamespace(distance=2.0),
            None,
        )


class FakeGeometry:
    def __init__(self):
        self.distance_inputs = []

    def compose_mid360_pose(
        self,
        *,
        torso_position_xyz,
        torso_orientation_wxyz,
    ):
        assert torso_position_xyz == (
            1.0,
            2.0,
            3.0,
        )
        assert torso_orientation_wxyz == (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        return SimpleNamespace(
            position_xyz=(1.1, 2.0, 3.4),
            orientation_wxyz=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        )

    def sensor_directions_to_world(
        self,
        pose,
        directions_sensor,
    ):
        assert pose.position_xyz == (
            1.1,
            2.0,
            3.4,
        )
        assert directions_sensor == (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        return directions_sensor

    def points_from_ray_distances(
        self,
        *,
        directions_sensor,
        distances,
        range_min,
        range_max,
    ):
        assert directions_sensor == (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        assert range_min == 0.25
        assert range_max == 40.0
        self.distance_inputs.append(distances)
        return (
            (2.0, 0.0, 0.0),
        )


def test_creates_only_the_lidar_point_cloud_publisher():
    node = FakeNode()

    SimulationLidarPublisher(
        node=node,
        ray_adapter=FakeRayAdapter(),
        schedule=LidarPublicationSchedule(
            period_seconds=0.1,
        ),
        geometry=FakeGeometry(),
        directions_sensor=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        frame_id="mid360_link",
        range_min=0.25,
        range_max=40.0,
    )

    assert set(node.publishers) == {
        "/g1/sensors/lidar/points",
    }

    publisher = node.publishers[
        LIDAR_POINTS_TOPIC
    ]
    assert publisher.message_type is PointCloud2
    assert publisher.qos.depth == 1
    assert publisher.qos.reliability == (
        ReliabilityPolicy.BEST_EFFORT
    )
    assert publisher.qos.durability == (
        DurabilityPolicy.VOLATILE
    )


def test_raycast_and_publish_occur_only_when_due():
    node = FakeNode()
    adapter = FakeRayAdapter()
    geometry = FakeGeometry()
    publisher = SimulationLidarPublisher(
        node=node,
        ray_adapter=adapter,
        schedule=LidarPublicationSchedule(
            period_seconds=0.1,
        ),
        geometry=geometry,
        directions_sensor=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        frame_id="mid360_link",
        range_min=0.25,
        range_max=40.0,
    )

    publisher.after_step(
        simulation_time=0.05,
    )

    assert adapter.torso_pose_calls == 0
    assert adapter.cast_calls == []
    assert node.publishers[
        LIDAR_POINTS_TOPIC
    ].messages == []

    publisher.after_step(
        simulation_time=0.10,
    )

    assert adapter.torso_pose_calls == 1
    assert adapter.cast_calls == [
        (
            (1.1, 2.0, 3.4),
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
        )
    ]
    assert geometry.distance_inputs == [
        (
            2.0,
            math.inf,
        )
    ]

    messages = node.publishers[
        LIDAR_POINTS_TOPIC
    ].messages
    assert len(messages) == 1

    message = messages[0]
    assert message.header.stamp.sec == 0
    assert message.header.stamp.nanosec == (
        100000000
    )
    assert message.header.frame_id == (
        "mid360_link"
    )
    assert message.width == 1
