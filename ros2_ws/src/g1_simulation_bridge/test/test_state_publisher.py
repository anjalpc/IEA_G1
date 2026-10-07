"""Contract tests for simulation navigation-state publishers."""

import pytest
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

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
            joint_names=(
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ),
            joint_positions=(0.11, -0.22, 0.33),
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


def test_creates_required_publishers_with_compatible_qos():
    node = FakeNode()

    publisher(node=node)

    assert set(node.publishers) == {
        "/clock",
        "/g1/state/imu_raw",
        "/g1/state/odometry_raw",
        "/joint_states",
    }

    for topic in (
        "/clock",
        "/g1/state/imu_raw",
        "/g1/state/odometry_raw",
    ):
        created = node.publishers[topic]
        assert created.qos.depth == 1
        assert (
            created.qos.reliability
            == ReliabilityPolicy.BEST_EFFORT
        )
        assert (
            created.qos.durability
            == DurabilityPolicy.VOLATILE
        )

    joint_states = node.publishers["/joint_states"]
    assert joint_states.qos.depth == 1
    assert (
        joint_states.qos.reliability
        == ReliabilityPolicy.RELIABLE
    )
    assert (
        joint_states.qos.durability
        == DurabilityPolicy.VOLATILE
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
    assert node.publishers["/joint_states"].messages == []


def test_publishes_joint_state_on_shared_state_cadence():
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
    ):
        state_publisher.after_step(
            data=data,
            simulation_time=simulation_time,
        )

    assert node.publishers["/joint_states"].messages == []
    assert reader.data_seen == []

    state_publisher.after_step(
        data=data,
        simulation_time=0.020,
    )

    assert reader.data_seen == [data]

    imu = node.publishers[
        "/g1/state/imu_raw"
    ].messages
    odometry = node.publishers[
        "/g1/state/odometry_raw"
    ].messages
    joint_states = node.publishers[
        "/joint_states"
    ].messages

    assert len(imu) == 1
    assert len(odometry) == 1
    assert len(joint_states) == 1

    joint_state = joint_states[0]
    assert joint_state.header.stamp == imu[0].header.stamp
    assert (
        joint_state.header.stamp
        == odometry[0].header.stamp
    )
    assert joint_state.header.stamp.nanosec == 20_000_000

    assert list(joint_state.name) == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    assert list(joint_state.position) == pytest.approx(
        [0.11, -0.22, 0.33]
    )
    assert list(joint_state.velocity) == []
    assert list(joint_state.effort) == []
