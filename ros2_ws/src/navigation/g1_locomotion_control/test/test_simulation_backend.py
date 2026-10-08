"""Tests for the arbiter-owned simulation backend."""

import math

import pytest

from geometry_msgs.msg import Twist

from g1_locomotion_control.simulation_backend import (
    SimulationBackend,
    SimulationBackendError,
)


class PublisherRecorder:
    """Record simulation velocity messages."""

    def __init__(self, subscription_count=1):
        self.messages = []
        self.subscription_count = subscription_count
        self.failure = None

    def publish(self, message):
        """Record or inject one publication failure."""
        if self.failure is not None:
            raise self.failure
        self.messages.append(message)

    def get_subscription_count(self):
        """Return the configured bridge-listener count."""
        return self.subscription_count


def test_activation_does_not_publish_motion():
    """Preparing the backend cannot move the carrier."""
    publisher = PublisherRecorder()
    backend = SimulationBackend(publisher)

    backend.activate()

    assert backend.active is True
    assert backend.stopped is True
    assert publisher.messages == []


def test_rejects_command_before_activation():
    """Inactive backends cannot publish."""
    publisher = PublisherRecorder()
    backend = SimulationBackend(publisher)

    with pytest.raises(
        SimulationBackendError,
        match="SIMULATION_BACKEND_INACTIVE",
    ):
        backend.command_velocity(0.10, 0.0, 0.20)

    assert publisher.messages == []


def test_publishes_only_planar_twist():
    """Approved planar velocity maps exactly to Twist."""
    publisher = PublisherRecorder()
    backend = SimulationBackend(publisher)
    backend.activate()

    backend.command_velocity(0.10, 0.0, -0.20)

    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert isinstance(message, Twist)
    assert message.linear.x == 0.10
    assert message.linear.y == 0.0
    assert message.linear.z == 0.0
    assert message.angular.x == 0.0
    assert message.angular.y == 0.0
    assert message.angular.z == -0.20
    assert backend.stopped is False


@pytest.mark.parametrize(
    "value",
    (math.nan, math.inf, -math.inf),
)
def test_rejects_nonfinite_velocity(value):
    """Invalid values never reach the simulation topic."""
    publisher = PublisherRecorder()
    backend = SimulationBackend(publisher)
    backend.activate()

    with pytest.raises(
        SimulationBackendError,
        match="SIMULATION_COMMAND_FAILED",
    ):
        backend.command_velocity(value, 0.0, 0.0)

    assert publisher.messages == []


def test_health_requires_live_simulation_subscriber():
    """The arbiter grants no lease without the bridge."""
    publisher = PublisherRecorder(subscription_count=0)
    backend = SimulationBackend(publisher)
    backend.activate()

    assert backend.healthy() is False

    publisher.subscription_count = 1

    assert backend.healthy() is True


def test_stop_publishes_explicit_zero():
    """Every active stop request reaches the bridge."""
    publisher = PublisherRecorder()
    backend = SimulationBackend(publisher)
    backend.activate()
    backend.command_velocity(0.10, 0.0, 0.20)

    backend.stop()

    assert len(publisher.messages) == 2
    stop = publisher.messages[-1]
    assert stop.linear.x == 0.0
    assert stop.linear.y == 0.0
    assert stop.angular.z == 0.0
    assert backend.stopped is True


def test_publication_failure_faults_backend_without_details():
    """Transport failures become one stable backend error."""
    publisher = PublisherRecorder()
    backend = SimulationBackend(publisher)
    backend.activate()
    publisher.failure = RuntimeError(
        "private DDS transport detail"
    )

    with pytest.raises(
        SimulationBackendError,
        match="SIMULATION_COMMAND_FAILED",
    ):
        backend.command_velocity(0.10, 0.0, 0.20)

    assert backend.faulted is True
    assert backend.healthy() is False
