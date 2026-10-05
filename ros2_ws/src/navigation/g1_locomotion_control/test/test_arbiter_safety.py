"""Deterministic safety-transition tests for locomotion control."""

from g1_robot_interfaces.locomotion import (
    OWNER_NAVIGATION,
    OWNER_NONE,
    PlanarVelocityLimits,
)

from g1_locomotion_control.arbiter import LocomotionArbiter
from g1_locomotion_control.backend import RecordingBackend


class FakeClock:
    """Controllable monotonic clock without real sleeps."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class LeaseFactory:
    """Return a new deterministic token for each acquisition."""

    def __init__(self):
        self.tokens = iter(("lease-one", "lease-two"))

    def __call__(self):
        return next(self.tokens)


def arbiter():
    backend = RecordingBackend()
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
        lease_factory=LeaseFactory(),
    )
    return instance, backend, clock


def acquire(instance):
    return instance.acquire(
        OWNER_NAVIGATION, "nav2-adapter"
    ).lease_id


def zero_command(instance, lease):
    return instance.command(
        owner=OWNER_NAVIGATION,
        lease_id=lease,
        linear_x=0.0,
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    )


def test_watchdog_does_not_expire_before_deadline():
    instance, backend, clock = arbiter()
    acquire(instance)
    clock.now += 0.249

    changed = instance.check_watchdog()

    assert changed is False
    assert instance.snapshot().active_owner == OWNER_NAVIGATION
    assert backend.stop_count == 0


def test_watchdog_stops_and_revokes_at_exact_deadline():
    instance, backend, clock = arbiter()
    acquire(instance)
    clock.now += 0.250

    changed = instance.check_watchdog()
    snapshot = instance.snapshot()

    assert changed is True
    assert snapshot.active_owner == OWNER_NONE
    assert snapshot.stopped is True
    assert snapshot.reason == "COMMAND_TIMEOUT"
    assert backend.stop_count == 1


def test_valid_zero_command_refreshes_watchdog_deadline():
    instance, backend, clock = arbiter()
    lease = acquire(instance)
    clock.now += 0.200
    zero_command(instance, lease)
    clock.now += 0.249

    assert instance.check_watchdog() is False
    assert instance.snapshot().active_owner == OWNER_NAVIGATION
    assert backend.stop_count == 0


def test_expired_lease_cannot_disrupt_new_owner():
    instance, backend, clock = arbiter()
    old_lease = acquire(instance)
    clock.now += 0.250
    assert instance.check_watchdog() is True

    new_lease = acquire(instance)
    result = zero_command(instance, old_lease)

    assert new_lease == "lease-two"
    assert result.accepted is False
    assert result.reason == "LEASE_MISMATCH"
    assert instance.snapshot().active_owner == OWNER_NAVIGATION
    assert backend.stop_count == 1


def test_emergency_stop_latches_and_blocks_acquisition():
    instance, backend, _clock = arbiter()
    acquire(instance)

    instance.set_emergency_stop()
    denied = instance.acquire(
        OWNER_NAVIGATION, "replacement"
    )
    snapshot = instance.snapshot()

    assert snapshot.emergency_stop is True
    assert snapshot.active_owner == OWNER_NONE
    assert snapshot.stopped is True
    assert snapshot.reason == "EMERGENCY_STOP"
    assert denied.granted is False
    assert denied.reason == "EMERGENCY_STOP"
    assert backend.stop_count == 1


def test_shutdown_stops_revokes_and_deactivates_backend():
    instance, backend, _clock = arbiter()
    acquire(instance)

    instance.shutdown()
    snapshot = instance.snapshot()

    assert snapshot.active_owner == OWNER_NONE
    assert snapshot.ready is False
    assert snapshot.stopped is True
    assert snapshot.reason == "SHUTDOWN"
    assert backend.events[-2:] == [
        ("stop",),
        ("deactivate",),
    ]


class FailingStopBackend(RecordingBackend):
    """Fail every requested backend stop."""

    def stop(self):
        self.stop_count += 1
        raise RuntimeError("injected stop failure")


def test_rejects_nonfinite_command_timeout():
    for timeout in (float("nan"), float("inf")):
        try:
            LocomotionArbiter(
                backend=RecordingBackend(),
                velocity_limits=PlanarVelocityLimits(
                    forward_mps=0.5,
                    lateral_mps=0.25,
                    yaw_rate_rps=0.4,
                ),
                command_timeout_seconds=timeout,
                monotonic=FakeClock(),
                lease_factory=LeaseFactory(),
            )
        except ValueError as exc:
            assert "finite and positive" in str(exc)
        else:
            raise AssertionError(
                "non-finite command timeout was accepted"
            )


def test_release_stop_failure_revokes_and_reports_backend_failure():
    backend = FailingStopBackend()
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
        lease_factory=LeaseFactory(),
    )
    lease = acquire(instance)

    result = instance.release(OWNER_NAVIGATION, lease)
    snapshot = instance.snapshot()

    assert result.released is False
    assert result.reason == "BACKEND_FAILURE"
    assert snapshot.active_owner == OWNER_NONE
    assert snapshot.ready is False
    assert snapshot.stopped is False
    assert snapshot.reason == "BACKEND_FAILURE"
    assert backend.stop_count == 1
