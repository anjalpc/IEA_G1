"""ROS-facing tests for the Nav2 command adapter."""

import pytest
import rclpy

from geometry_msgs.msg import Twist
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    ReliabilityPolicy,
)

from g1_locomotion_interfaces.msg import LocomotionCommand
from g1_locomotion_interfaces.srv import AcquireLocomotion
from g1_robot_interfaces.locomotion import (
    OWNER_MANUAL,
    OWNER_NAVIGATION,
)

from g1_navigation_control.node import (
    NavigationCommandAdapterNode,
)


class PublisherRecorder:
    """Record published messages without ROS graph timing."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        """Record one published message."""
        self.messages.append(message)


class PendingFuture:
    """Record completion callbacks for a pending request."""

    def __init__(self):
        self.callbacks = []

    def add_done_callback(self, callback):
        """Record a future completion callback."""
        self.callbacks.append(callback)


class ReadyClient:
    """Record asynchronous service requests."""

    def __init__(self):
        self.requests = []
        self.future = PendingFuture()

    def service_is_ready(self):
        """Report that the fake service is available."""
        return True

    def call_async(self, request):
        """Record and return one pending request."""
        self.requests.append(request)
        return self.future


@pytest.fixture
def node():
    """Create an isolated navigation adapter node."""
    if not rclpy.ok():
        rclpy.init()

    instance = NavigationCommandAdapterNode()
    yield instance

    instance.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def nav2_twist():
    """Return one valid forward-only Nav2 command."""
    message = Twist()
    message.linear.x = 0.10
    message.angular.z = -0.20
    return message


def test_creates_public_ros_boundary(node):
    """The node connects Nav2 only to public locomotion APIs."""
    assert (
        node.raw_command_subscription.topic_name
        == "/g1/navigation/cmd_vel_raw"
    )
    assert (
        node.command_publisher.topic_name
        == "/g1/locomotion/command"
    )
    assert (
        node.acquire_client.srv_name
        == "/g1/locomotion/acquire"
    )
    assert (
        node.release_client.srv_name
        == "/g1/locomotion/release"
    )


def test_uses_bounded_command_stream_qos(node):
    """Both command-stream boundaries remain bounded."""
    assert node.raw_command_qos.depth == 1
    assert (
        node.raw_command_qos.reliability
        == ReliabilityPolicy.RELIABLE
    )
    assert (
        node.raw_command_qos.durability
        == DurabilityPolicy.VOLATILE
    )

    assert node.locomotion_command_qos.depth == 1
    assert (
        node.locomotion_command_qos.reliability
        == ReliabilityPolicy.BEST_EFFORT
    )
    assert (
        node.locomotion_command_qos.durability
        == DurabilityPolicy.VOLATILE
    )


def test_unleased_command_requests_navigation_ownership(node):
    """Demand triggers acquisition but no unauthorized command."""
    publisher = PublisherRecorder()
    acquire_client = ReadyClient()
    node.command_publisher = publisher
    node.acquire_client = acquire_client

    node.command_callback(nav2_twist())

    assert publisher.messages == []
    assert len(acquire_client.requests) == 1
    request = acquire_client.requests[0]
    assert request.requested_owner == request.OWNER_NAVIGATION
    assert request.requester_id == "g1-navigation-control"
    assert len(acquire_client.future.callbacks) == 1


def test_active_lease_publishes_validated_command(node):
    """A granted lease authorizes conversion and publication."""
    publisher = PublisherRecorder()
    node.command_publisher = publisher
    node.gate.grant("navigation-lease")

    node.command_callback(nav2_twist())

    assert len(publisher.messages) == 1
    command = publisher.messages[0]
    assert isinstance(command, LocomotionCommand)
    assert command.owner == command.OWNER_NAVIGATION
    assert command.lease_id == "navigation-lease"
    assert command.twist.linear.x == pytest.approx(0.10)
    assert command.twist.linear.y == 0.0
    assert command.twist.angular.z == pytest.approx(-0.20)

class CompletedFuture:
    """Return one completed service result."""

    def __init__(self, response):
        self.response = response

    def result(self):
        """Return the configured response."""
        return self.response


class FakeClock:
    """Provide deterministic monotonic time."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        """Return the current test time."""
        return self.now


def granted_response(lease_id="navigation-lease"):
    """Return a successful navigation lease response."""
    response = AcquireLocomotion.Response()
    response.granted = True
    response.lease_id = lease_id
    response.active_owner = OWNER_NAVIGATION
    response.reason = "LEASE_GRANTED"
    return response


def test_granted_response_activates_navigation_lease(node):
    """Only a successful navigation response activates commands."""
    node.acquire_response_callback(
        CompletedFuture(granted_response())
    )

    assert node.gate.active is True


def test_denied_response_does_not_activate_navigation(node):
    """A denied request leaves navigation unauthorized."""
    response = AcquireLocomotion.Response()
    response.granted = False
    response.active_owner = OWNER_MANUAL
    response.reason = "LEASE_ACTIVE"

    node.acquire_response_callback(CompletedFuture(response))

    assert node.gate.active is False


def test_invalid_active_command_stops_and_releases(node):
    """Unsafe Nav2 output immediately abandons authority."""
    publisher = PublisherRecorder()
    release_client = ReadyClient()
    node.command_publisher = publisher
    node.release_client = release_client
    node.gate.grant("navigation-lease")

    message = nav2_twist()
    message.linear.x = float("inf")
    node.command_callback(message)

    assert len(publisher.messages) == 1
    stop = publisher.messages[0]
    assert stop.twist.linear.x == 0.0
    assert stop.twist.linear.y == 0.0
    assert stop.twist.angular.z == 0.0

    assert len(release_client.requests) == 1
    request = release_client.requests[0]
    assert request.owner == OWNER_NAVIGATION
    assert request.lease_id == "navigation-lease"
    assert node.gate.active is False


def test_watchdog_stops_and_releases_silent_navigation():
    """A stalled Nav2 stream cannot retain movement authority."""
    if not rclpy.ok():
        rclpy.init()

    clock = FakeClock()
    instance = NavigationCommandAdapterNode(
        monotonic=clock,
        parameter_overrides=[
            Parameter(
                "command_timeout_seconds",
                value=0.25,
            ),
            Parameter(
                "watchdog_rate_hz",
                value=20.0,
            ),
        ],
    )
    publisher = PublisherRecorder()
    release_client = ReadyClient()
    instance.command_publisher = publisher
    instance.release_client = release_client

    try:
        instance.acquire_response_callback(
            CompletedFuture(granted_response())
        )
        clock.now += 0.251

        instance.watchdog_callback()

        assert len(publisher.messages) == 1
        assert publisher.messages[0].twist.linear.x == 0.0
        assert len(release_client.requests) == 1
        assert instance.gate.active is False
    finally:
        instance.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def test_pending_acquisition_is_not_duplicated(node):
    """One pending request suppresses duplicate acquisitions."""
    acquire_client = ReadyClient()
    node.acquire_client = acquire_client

    node.command_callback(nav2_twist())
    node.command_callback(nav2_twist())
    node.command_callback(nav2_twist())

    assert len(acquire_client.requests) == 1


def test_destroy_stops_and_releases_active_navigation(node):
    """Shutdown abandons navigation authority fail-closed."""
    publisher = PublisherRecorder()
    release_client = ReadyClient()
    node.command_publisher = publisher
    node.release_client = release_client
    node.gate.grant("navigation-lease")

    node.destroy_node()

    assert len(publisher.messages) == 1
    stop = publisher.messages[0]
    assert stop.twist.linear.x == 0.0
    assert stop.twist.linear.y == 0.0
    assert stop.twist.angular.z == 0.0
    assert len(release_client.requests) == 1
    assert release_client.requests[0].lease_id == (
        "navigation-lease"
    )
    assert node.gate.active is False

