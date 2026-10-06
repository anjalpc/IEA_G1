"""Contract tests for simulation navigation-state publishers."""

import pytest
from rclpy.qos import ReliabilityPolicy

from g1_simulation_bridge.simulation_state import (
    SimulationStateSample,
)
from g1_simulation_bridge.simulation_timing import (
    SimulationPublicationSchedule,
)
from g1_simulation_bridge.state_publisher import (
    SimulationStatePublisher,
)


class FakePublisher:
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

    def create_publisher(self, message_type, topic, qos):
        publisher = FakePublisher(
            message_type,
            topic,
            qos,
        )
        self.publishers[topic] = publisher
        return publisher


class FakeReader:
    def __init__(self):
        self.data_seen = []

    def sample(self, data):
        self.data_seen.append(data)
        return SimulationStateSample(
            position_xyz=(1.0, 2.0, 3.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            linear_velocity_xyz=(0.4, 0.5, 0.6),
            angular_velocity_xyz=(0.7, 0.8, 0.9),
            imu_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            imu_angular_velocity_xyz=(1.1, 1.2, 1.3),
            imu_linear_acceleration_xyz=(2.1, 2.2, 2.3),
        )


def publisher(node=None, reader=None):
    return SimulationStatePublisher(
        node=node or FakeNode(),
        state_reader=reader or FakeReader(),
        schedule=SimulationPublicationSchedule(
            state_period_seconds=0.02,
        ),
        imu_frame_id="pelvis",
        odom_frame_id="odom",
        child_frame_id="pelvis",
        covariance_variance=1e-6,
    )


def test_creates_required_best_effort_depth_one_publishers():
    node = FakeNode()

    publisher(node=node)

    assert set(node.publishers) == {
        "/clock",
        "/g1/state/imu_raw",
        "/g1/state/odometry_raw",
    }
    for created in node.publishers.values():
        assert created.qos.depth == 1
        assert (
            created.qos.reliability
            == ReliabilityPolicy.BEST_EFFORT
        )


def test_publishes_clock_after_every_physics_step():
    node = FakeNode()
    state_publisher = publisher(node=node)

    state_publisher.after_step(
        data=object(),
        simulation_time=0.005,
    )
    state_publisher.after_step(
        data=object(),
        simulation_time=0.010,
    )

    messages = node.publishers["/clock"].messages
    assert len(messages) == 2
    assert messages[0].clock.nanosec == 5_000_000
    assert messages[1].clock.nanosec == 10_000_000

    assert (
        node.publishers["/g1/state/imu_raw"].messages
        == []
    )
    assert (
        node.publishers["/g1/state/odometry_raw"].messages
        == []
    )


def test_publishes_raw_state_only_when_schedule_is_due():
    node = FakeNode()
    reader = FakeReader()
    state_publisher = publisher(
        node=node,
        reader=reader,
    )
    data = object()

    for simulation_time in (
        0.005,
        0.010,
        0.015,
        0.020,
    ):
        state_publisher.after_step(
            data=data,
            simulation_time=simulation_time,
        )

    assert reader.data_seen == [data]

    imu_messages = node.publishers[
        "/g1/state/imu_raw"
    ].messages
    odometry_messages = node.publishers[
        "/g1/state/odometry_raw"
    ].messages

    assert len(imu_messages) == 1
    assert len(odometry_messages) == 1

    imu = imu_messages[0]
    odometry = odometry_messages[0]

    assert imu.header.stamp.nanosec == 20_000_000
    assert imu.header.frame_id == "pelvis"

    assert odometry.header.stamp.nanosec == 20_000_000
    assert odometry.header.frame_id == "odom"
    assert odometry.child_frame_id == "pelvis"


def test_does_not_publish_partial_state_when_sampling_fails():
    class FailingReader:
        def sample(self, data):
            del data
            raise ValueError("invalid simulation state")

    node = FakeNode()
    state_publisher = publisher(
        node=node,
        reader=FailingReader(),
    )

    with pytest.raises(
        ValueError,
        match="invalid simulation state",
    ):
        state_publisher.after_step(
            data=object(),
            simulation_time=0.020,
        )

    assert len(node.publishers["/clock"].messages) == 1
    assert (
        node.publishers["/g1/state/imu_raw"].messages
        == []
    )
    assert (
        node.publishers["/g1/state/odometry_raw"].messages
        == []
    )
