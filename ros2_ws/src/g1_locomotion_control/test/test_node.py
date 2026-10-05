"""ROS-facing tests for the hardware-free locomotion arbiter."""

import pytest
import rclpy

from rclpy.parameter import Parameter

from g1_locomotion_interfaces.msg import LocomotionState
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
