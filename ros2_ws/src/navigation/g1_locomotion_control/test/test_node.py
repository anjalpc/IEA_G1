"""ROS-facing tests for the hardware-free locomotion arbiter."""

import pytest
import rclpy

from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

from g1_locomotion_interfaces.msg import (
    LocomotionCommand,
    LocomotionState,
)
from g1_locomotion_interfaces.srv import (
    AcquireLocomotion,
    ReleaseLocomotion,
)

from g1_locomotion_control.backend import RecordingBackend
from g1_locomotion_control.node import LocomotionArbiterNode


@pytest.fixture
def node():
    if not rclpy.ok():
        rclpy.init()

    backend = RecordingBackend()
    instance = LocomotionArbiterNode(
        backend=backend,
        parameter_overrides=[
            Parameter(
                "max_forward_mps",
                value=0.5,
            ),
            Parameter(
                "max_lateral_mps",
                value=0.25,
            ),
            Parameter(
                "max_yaw_rate_rps",
                value=0.4,
            ),
        ],
        lease_factory=lambda: "node-lease",
    )

    yield instance

    instance.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_creates_contract_services_and_state_publisher(node):
    services = {
        name
        for name, _types in node.get_service_names_and_types()
    }
    topics = {
        name
        for name, _types in node.get_topic_names_and_types()
    }

    assert "/g1/locomotion/acquire" in services
    assert "/g1/locomotion/release" in services
    assert "/g1/locomotion/state" in topics


def test_acquire_service_maps_pure_result(node):
    request = AcquireLocomotion.Request()
    request.requested_owner = request.OWNER_NAVIGATION
    request.requester_id = "nav2-adapter"
    response = AcquireLocomotion.Response()

    result = node.acquire_callback(request, response)

    assert result.granted is True
    assert result.lease_id == "node-lease"
    assert result.active_owner == request.OWNER_NAVIGATION
    assert result.reason == "LEASE_GRANTED"


def test_release_service_stops_and_clears_owner(node):
    acquire_request = AcquireLocomotion.Request()
    acquire_request.requested_owner = (
        acquire_request.OWNER_NAVIGATION
    )
    acquire_request.requester_id = "nav2-adapter"
    acquired = node.acquire_callback(
        acquire_request,
        AcquireLocomotion.Response(),
    )

    release_request = ReleaseLocomotion.Request()
    release_request.owner = acquire_request.OWNER_NAVIGATION
    release_request.lease_id = acquired.lease_id

    released = node.release_callback(
        release_request,
        ReleaseLocomotion.Response(),
    )

    assert released.released is True
    assert released.active_owner == 0
    assert released.reason == "RELEASED"


def test_public_state_maps_snapshot_without_lease(node):
    request = AcquireLocomotion.Request()
    request.requested_owner = request.OWNER_NAVIGATION
    request.requester_id = "nav2-adapter"
    node.acquire_callback(
        request,
        AcquireLocomotion.Response(),
    )

    state = node.state_message()

    assert isinstance(state, LocomotionState)
    assert state.active_owner == state.OWNER_NAVIGATION
    assert state.active_backend == state.BACKEND_SIMULATION
    assert state.ready is True
    assert state.stopped is True
    assert state.emergency_stop is False
    assert state.reason == "LEASE_GRANTED"
    assert not hasattr(state, "lease_id")


def acquire_navigation(node):
    request = AcquireLocomotion.Request()
    request.requested_owner = request.OWNER_NAVIGATION
    request.requester_id = "nav2-adapter"
    return node.acquire_callback(
        request,
        AcquireLocomotion.Response(),
    )


def locomotion_command(lease_id):
    message = LocomotionCommand()
    message.owner = message.OWNER_NAVIGATION
    message.lease_id = lease_id
    message.twist.linear.x = 0.2
    message.twist.linear.y = -0.1
    message.twist.angular.z = 0.3
    return message


def test_creates_best_effort_depth_one_command_subscription(node):
    topics = {
        name
        for name, _types in node.get_topic_names_and_types()
    }

    assert "/g1/locomotion/command" in topics
    assert node.command_qos.depth == 1
    assert (
        node.command_qos.reliability
        == ReliabilityPolicy.BEST_EFFORT
    )
    assert (
        node.command_qos.durability
        == DurabilityPolicy.VOLATILE
    )


def test_command_callback_dispatches_to_recording_backend(node):
    acquired = acquire_navigation(node)

    node.command_callback(
        locomotion_command(acquired.lease_id)
    )

    assert node.backend.commands == [(0.2, -0.1, 0.3)]
    assert node.arbiter.snapshot().reason == "COMMAND_ACCEPTED"


def test_command_callback_does_not_log_or_expose_bad_lease(node):
    acquire_navigation(node)
    message = locomotion_command("expired-secret-token")

    node.command_callback(message)

    assert node.backend.commands == []
    assert node.backend.stop_count == 0
    assert node.arbiter.snapshot().active_owner == (
        LocomotionCommand.OWNER_NAVIGATION
    )


class FakeClock:
    """Deterministic node watchdog clock."""

    def __init__(self):
        self.now = 300.0

    def __call__(self):
        return self.now


@pytest.fixture
def timed_node():
    if not rclpy.ok():
        rclpy.init()

    clock = FakeClock()
    instance = LocomotionArbiterNode(
        backend=RecordingBackend(),
        parameter_overrides=[
            Parameter("max_forward_mps", value=0.5),
            Parameter("max_lateral_mps", value=0.25),
            Parameter("max_yaw_rate_rps", value=0.4),
            Parameter("watchdog_rate_hz", value=50.0),
            Parameter("state_publish_rate_hz", value=10.0),
        ],
        lease_factory=lambda: "timed-lease",
        monotonic=clock,
    )
    yield instance, clock

    instance.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_creates_required_watchdog_and_state_timers(timed_node):
    instance, _clock = timed_node

    assert instance.watchdog_timer.timer_period_ns == 20_000_000
    assert instance.state_timer.timer_period_ns == 100_000_000


def test_watchdog_callback_revokes_expired_lease(timed_node):
    instance, clock = timed_node
    acquire_navigation(instance)
    clock.now += 0.250

    instance.watchdog_callback()

    snapshot = instance.arbiter.snapshot()
    assert snapshot.active_owner == 0
    assert snapshot.reason == "COMMAND_TIMEOUT"
    assert instance.backend.stop_count == 1


def test_rejects_nonrecording_backend_configuration():
    if not rclpy.ok():
        rclpy.init()

    try:
        with pytest.raises(
            ValueError,
            match="only the recording backend",
        ):
            LocomotionArbiterNode(
                parameter_overrides=[
                    Parameter("backend", value="sport"),
                    Parameter(
                        "max_forward_mps",
                        value=0.5,
                    ),
                    Parameter(
                        "max_lateral_mps",
                        value=0.25,
                    ),
                    Parameter(
                        "max_yaw_rate_rps",
                        value=0.4,
                    ),
                ],
            )
    finally:
        if rclpy.ok():
            rclpy.shutdown()
