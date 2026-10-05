"""Tests for validated locomotion command dispatch."""

from g1_robot_interfaces.locomotion import (
    OWNER_NAVIGATION,
    PlanarVelocityLimits,
)

from g1_locomotion_control.arbiter import LocomotionArbiter
from g1_locomotion_control.backend import RecordingBackend


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 20.0

    def __call__(self):
        return self.now


class FailingCommandBackend(RecordingBackend):
    """Raise while dispatching an otherwise valid command."""

    def command_velocity(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        del forward_mps, lateral_mps, yaw_rate_rps
        raise RuntimeError("injected backend failure")


def arbiter(backend=None):
    backend = backend or RecordingBackend()
    clock = FakeClock()
    instance = LocomotionArbiter(
        backend=backend,
        velocity_limits=PlanarVelocityLimits(
            forward_mps=0.5,
            lateral_mps=0.25,
            yaw_rate_rps=0.4,
        ),
        command_timeout_seconds=0.250,
        monotonic=clock,
        lease_factory=lambda: "active-lease",
    )
    lease = instance.acquire(
        OWNER_NAVIGATION, "nav2-adapter"
    ).lease_id
    return instance, backend, clock, lease


def valid_command(instance, lease_id):
    return instance.command(
        owner=OWNER_NAVIGATION,
        lease_id=lease_id,
        linear_x=0.2,
        linear_y=-0.1,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.3,
    )


def test_valid_command_reaches_backend_and_updates_public_state():
    instance, backend, _clock, lease = arbiter()

    result = valid_command(instance, lease)
    snapshot = instance.snapshot()

    assert result.accepted is True
    assert result.reason == "COMMAND_ACCEPTED"
    assert backend.commands == [(0.2, -0.1, 0.3)]
    assert snapshot.stopped is False
    assert snapshot.reason == "COMMAND_ACCEPTED"


def test_zero_command_is_accepted_and_refreshes_command_age():
    instance, backend, clock, lease = arbiter()
    clock.now = 20.2

    result = instance.command(
        owner=OWNER_NAVIGATION,
        lease_id=lease,
        linear_x=0.0,
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    )

    assert result.accepted is True
    assert backend.commands == [(0.0, 0.0, 0.0)]
    assert instance.snapshot().command_age_seconds == 0.0
    assert instance.snapshot().stopped is True


def test_wrong_lease_is_rejected_without_disrupting_active_owner():
    instance, backend, _clock, _lease = arbiter()

    result = valid_command(instance, "expired-lease")
    snapshot = instance.snapshot()

    assert result.accepted is False
    assert result.reason == "LEASE_MISMATCH"
    assert snapshot.active_owner == OWNER_NAVIGATION
    assert snapshot.reason == "LEASE_GRANTED"
    assert backend.commands == []
    assert backend.stop_count == 0


def test_wrong_owner_is_rejected_without_disrupting_active_owner():
    instance, backend, _clock, lease = arbiter()

    result = instance.command(
        owner=1,
        lease_id=lease,
        linear_x=0.1,
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    )

    assert result.accepted is False
    assert result.reason == "OWNER_MISMATCH"
    assert instance.snapshot().active_owner == OWNER_NAVIGATION
    assert backend.commands == []
    assert backend.stop_count == 0


def test_invalid_active_command_stops_and_revokes_owner():
    instance, backend, _clock, lease = arbiter()

    result = instance.command(
        owner=OWNER_NAVIGATION,
        lease_id=lease,
        linear_x=float("nan"),
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    )
    snapshot = instance.snapshot()

    assert result.accepted is False
    assert result.reason == "INVALID_COMMAND"
    assert snapshot.active_owner == 0
    assert snapshot.stopped is True
    assert snapshot.reason == "INVALID_COMMAND"
    assert backend.commands == []
    assert backend.stop_count == 1


def test_unhealthy_backend_stops_and_revokes_owner():
    instance, backend, _clock, lease = arbiter()
    backend.set_healthy(False)

    result = valid_command(instance, lease)
    snapshot = instance.snapshot()

    assert result.accepted is False
    assert result.reason == "BACKEND_UNAVAILABLE"
    assert snapshot.active_owner == 0
    assert snapshot.ready is False
    assert snapshot.stopped is True
    assert backend.commands == []
    assert backend.stop_count == 1


def test_backend_dispatch_failure_stops_and_revokes_owner():
    backend = FailingCommandBackend()
    instance, backend, _clock, lease = arbiter(backend)

    result = valid_command(instance, lease)
    snapshot = instance.snapshot()

    assert result.accepted is False
    assert result.reason == "BACKEND_FAILURE"
    assert snapshot.active_owner == 0
    assert snapshot.ready is False
    assert snapshot.stopped is True
    assert backend.stop_count == 1
