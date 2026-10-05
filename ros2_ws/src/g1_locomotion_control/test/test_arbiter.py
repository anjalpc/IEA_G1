"""Tests for pure locomotion ownership and lease lifecycle."""

from g1_robot_interfaces.locomotion import (
    OWNER_MANUAL,
    OWNER_NAVIGATION,
    OWNER_NONE,
)

from g1_locomotion_control.arbiter import LocomotionArbiter
from g1_locomotion_control.backend import RecordingBackend


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class LeaseFactory:
    """Return deterministic opaque lease tokens."""

    def __init__(self):
        self.tokens = iter(("lease-one", "lease-two"))

    def __call__(self):
        return next(self.tokens)


def arbiter():
    backend = RecordingBackend()
    clock = FakeClock()
    instance = LocomotionArbiter(
        backend=backend,
        command_timeout_seconds=0.250,
        monotonic=clock,
        lease_factory=LeaseFactory(),
    )
    return instance, backend, clock


def test_grants_one_opaque_lease_and_activates_backend():
    instance, backend, _clock = arbiter()

    result = instance.acquire(OWNER_NAVIGATION, "nav2-adapter")

    assert result.granted is True
    assert result.lease_id == "lease-one"
    assert result.active_owner == OWNER_NAVIGATION
    assert result.reason == "LEASE_GRANTED"
    assert backend.events == [("activate",)]


def test_snapshot_never_exposes_lease_or_requester():
    instance, _backend, _clock = arbiter()
    instance.acquire(OWNER_NAVIGATION, "private-requester")

    snapshot = instance.snapshot()

    assert snapshot.active_owner == OWNER_NAVIGATION
    assert snapshot.reason == "LEASE_GRANTED"
    assert not hasattr(snapshot, "lease_id")
    assert not hasattr(snapshot, "requester_id")
    assert "lease-one" not in repr(snapshot)
    assert "private-requester" not in repr(snapshot)


def test_rejects_conflicting_acquisition_without_leaking_lease():
    instance, _backend, _clock = arbiter()
    instance.acquire(OWNER_NAVIGATION, "nav2-adapter")

    result = instance.acquire(OWNER_MANUAL, "operator")

    assert result.granted is False
    assert result.lease_id == ""
    assert result.active_owner == OWNER_NAVIGATION
    assert result.reason == "LEASE_ACTIVE"


def test_rejects_empty_requester_without_acquiring():
    instance, _backend, _clock = arbiter()

    result = instance.acquire(OWNER_NAVIGATION, "   ")

    assert result.granted is False
    assert result.lease_id == ""
    assert result.active_owner == OWNER_NONE
    assert result.reason == "INVALID_REQUEST"


def test_wrong_lease_cannot_release_or_stop_current_owner():
    instance, backend, _clock = arbiter()
    instance.acquire(OWNER_NAVIGATION, "nav2-adapter")

    result = instance.release(OWNER_NAVIGATION, "wrong-lease")

    assert result.released is False
    assert result.active_owner == OWNER_NAVIGATION
    assert result.reason == "LEASE_MISMATCH"
    assert backend.stop_count == 0


def test_wrong_owner_cannot_release_current_lease():
    instance, backend, _clock = arbiter()
    lease = instance.acquire(
        OWNER_NAVIGATION, "nav2-adapter"
    ).lease_id

    result = instance.release(OWNER_MANUAL, lease)

    assert result.released is False
    assert result.active_owner == OWNER_NAVIGATION
    assert result.reason == "OWNER_MISMATCH"
    assert backend.stop_count == 0


def test_successful_release_stops_before_clearing_owner():
    instance, backend, _clock = arbiter()
    lease = instance.acquire(
        OWNER_NAVIGATION, "nav2-adapter"
    ).lease_id

    result = instance.release(OWNER_NAVIGATION, lease)
    snapshot = instance.snapshot()

    assert result.released is True
    assert result.active_owner == OWNER_NONE
    assert result.reason == "RELEASED"
    assert backend.events[-1] == ("stop",)
    assert snapshot.active_owner == OWNER_NONE
    assert snapshot.stopped is True
    assert snapshot.reason == "RELEASED"
