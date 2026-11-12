"""ROS-facing tests for the hardware-free locomotion arbiter."""

from unittest.mock import Mock

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
from g1_locomotion_control.node import (
    LocomotionArbiterNode,
    backend_state_constant,
)
from g1_locomotion_control.sport_backend import SportBackend
from g1_locomotion_control.simulation_backend import (
    SimulationBackend,
)


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
    assert (
        node.acquire_service.srv_name
        == "/g1/locomotion/acquire"
    )
    assert (
        node.release_service.srv_name
        == "/g1/locomotion/release"
    )
    assert (
        node.state_publisher.topic_name
        == "/g1/locomotion/state"
    )


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
            Parameter(
                "lease_start_timeout_seconds", value=5.0
            ),
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


def test_watchdog_uses_configured_lease_start_deadline(timed_node):
    instance, clock = timed_node
    acquire_navigation(instance)
    clock.now += 0.250

    instance.watchdog_callback()

    snapshot = instance.arbiter.snapshot()
    assert snapshot.active_owner == LocomotionCommand.OWNER_NAVIGATION
    assert snapshot.reason == "LEASE_GRANTED"
    assert instance.backend.stop_count == 0

    clock.now += 4.750
    instance.watchdog_callback()

    snapshot = instance.arbiter.snapshot()
    assert snapshot.active_owner == 0
    assert snapshot.reason == "LEASE_START_TIMEOUT"
    assert instance.backend.stop_count == 1


def test_maps_sport_backend_to_public_identity():
    assert backend_state_constant("sport") == (
        LocomotionState.BACKEND_SPORT
    )


def test_accepts_only_explicit_injected_sport_backend():
    if not rclpy.ok():
        rclpy.init()

    client = Mock()
    client.healthy.return_value = True
    backend = SportBackend(client)
    instance = LocomotionArbiterNode(
        backend=backend,
        parameter_overrides=[
            Parameter("backend", value="sport"),
            Parameter("max_forward_mps", value=0.15),
            Parameter("max_lateral_mps", value=0.10),
            Parameter("max_yaw_rate_rps", value=0.40),
        ],
    )

    try:
        state = instance.state_message()
        assert instance.backend is backend
        assert state.active_backend == state.BACKEND_SPORT
        assert state.stopped is True
        assert client.move.call_count == 0
    finally:
        instance.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_constructs_sport_backend_only_from_explicit_factory():
    if not rclpy.ok():
        rclpy.init()

    client = Mock()
    client.healthy.return_value = True
    backend = SportBackend(client)
    backend_factory = Mock(return_value=backend)
    instance = LocomotionArbiterNode(
        sport_backend_factory=backend_factory,
        parameter_overrides=[
            Parameter("backend", value="sport"),
            Parameter("sport_interface", value="eth0"),
            Parameter(
                "sport_response_timeout_seconds",
                value=1.5,
            ),
            Parameter(
                "sport_velocity_duration_seconds",
                value=1.0,
            ),
            Parameter("max_forward_mps", value=0.15),
            Parameter("max_lateral_mps", value=0.10),
            Parameter("max_yaw_rate_rps", value=0.40),
        ],
    )

    try:
        assert instance.backend is backend
        backend_factory.assert_called_once_with(
            interface="eth0",
            response_timeout_seconds=1.5,
            rpc_timeout_seconds=1.0,
            velocity_duration_seconds=1.0,
        )
    finally:
        instance.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_physical_sport_factory_uses_native_environment():
    from g1_locomotion_control.node import (
        physical_sport_backend_factory,
    )

    backend = Mock()
    environment_factory = Mock()
    supervised_factory = Mock(return_value=backend)

    result = physical_sport_backend_factory(
        interface="eth0",
        response_timeout_seconds=1.5,
        rpc_timeout_seconds=1.0,
        velocity_duration_seconds=1.0,
        environment_factory=environment_factory,
        supervised_factory=supervised_factory,
    )

    assert result is backend
    supervised_factory.assert_called_once_with(
        interface="eth0",
        response_timeout_seconds=1.5,
        rpc_timeout_seconds=1.0,
        velocity_duration_seconds=1.0,
        environment_factory=environment_factory,
    )


def test_destroy_closes_owned_sport_transport():
    if not rclpy.ok():
        rclpy.init()

    client = Mock()
    client.healthy.return_value = True
    backend = SportBackend(client)
    backend.shutdown_transport = Mock()
    instance = LocomotionArbiterNode(
        sport_backend_factory=Mock(return_value=backend),
        parameter_overrides=[
            Parameter("backend", value="sport"),
            Parameter("sport_interface", value="eth0"),
            Parameter(
                "sport_response_timeout_seconds",
                value=1.5,
            ),
            Parameter("max_forward_mps", value=0.15),
            Parameter("max_lateral_mps", value=0.10),
            Parameter("max_yaw_rate_rps", value=0.40),
        ],
    )

    instance.destroy_node()

    backend.shutdown_transport.assert_called_once_with()
    if rclpy.ok():
        rclpy.shutdown()


def test_rejects_sport_backend_under_recording_identity():
    if not rclpy.ok():
        rclpy.init()

    client = Mock()
    client.healthy.return_value = True
    backend = SportBackend(client)

    try:
        with pytest.raises(
            ValueError,
            match="backend identity does not match injected backend",
        ):
            LocomotionArbiterNode(
                backend=backend,
                parameter_overrides=[
                    Parameter("backend", value="recording"),
                    Parameter("max_forward_mps", value=0.15),
                    Parameter("max_lateral_mps", value=0.10),
                    Parameter("max_yaw_rate_rps", value=0.40),
                ],
            )
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_rejects_unsupported_backend_configuration():
    if not rclpy.ok():
        rclpy.init()

    try:
        with pytest.raises(
            ValueError,
            match="unsupported locomotion backend",
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

def test_constructs_explicit_simulation_backend():
    """Simulation selection creates only the ROS transport."""
    if not rclpy.ok():
        rclpy.init()

    instance = LocomotionArbiterNode(
        parameter_overrides=[
            Parameter("backend", value="simulation"),
            Parameter("max_forward_mps", value=0.15),
            Parameter("max_lateral_mps", value=0.01),
            Parameter("max_yaw_rate_rps", value=0.40),
        ],
    )

    try:
        assert isinstance(
            instance.backend,
            SimulationBackend,
        )
        assert (
            instance.simulation_command_publisher.topic_name
            == "/g1/simulation/locomotion_command"
        )
        assert instance.simulation_command_qos.depth == 1
        assert (
            instance.simulation_command_qos.reliability
            == ReliabilityPolicy.BEST_EFFORT
        )
        assert (
            instance.simulation_command_qos.durability
            == DurabilityPolicy.VOLATILE
        )
    finally:
        instance.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_physical_main_injects_only_physical_sport_factory(
    monkeypatch,
):
    import g1_locomotion_control.node as node_module

    instance = Mock()
    node_factory = Mock(return_value=instance)
    monkeypatch.setattr(
        node_module,
        "LocomotionArbiterNode",
        node_factory,
    )
    monkeypatch.setattr(node_module.rclpy, "init", Mock())
    monkeypatch.setattr(node_module.rclpy, "spin", Mock())
    monkeypatch.setattr(
        node_module.rclpy,
        "ok",
        Mock(return_value=False),
    )

    node_module.physical_main(args=["--ros-args"])

    node_factory.assert_called_once_with(
        sport_backend_factory=(
            node_module.physical_sport_backend_factory
        ),
    )
    instance.destroy_node.assert_called_once_with()
