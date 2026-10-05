"""Concurrency invariants for locomotion ownership transitions."""

import threading

from g1_robot_interfaces.locomotion import (
    OWNER_NAVIGATION,
    OWNER_NONE,
    PlanarVelocityLimits,
)

from g1_locomotion_control.arbiter import LocomotionArbiter
from g1_locomotion_control.backend import RecordingBackend


class FakeClock:
    """Thread-safe enough immutable test clock."""

    def __init__(self):
        self.now = 50.0

    def __call__(self):
        return self.now


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
        lease_factory=lambda: "race-lease",
    )
    lease = instance.acquire(
        OWNER_NAVIGATION, "nav2-adapter"
    ).lease_id
    return instance, backend, clock, lease


def command(instance, lease):
    return instance.command(
        owner=OWNER_NAVIGATION,
        lease_id=lease,
        linear_x=0.2,
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    )


def race(left, right):
    """Release two operations simultaneously and collect results."""
    barrier = threading.Barrier(3)
    results = {}
    errors = []

    def run(name, operation):
        try:
            barrier.wait()
            results[name] = operation()
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("left", left)),
        threading.Thread(target=run, args=("right", right)),
    ]
    for thread in threads:
        thread.start()

    barrier.wait()

    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert errors == []
    return results


def test_command_and_timeout_have_one_serialized_winner():
    instance, backend, clock, lease = arbiter()
    clock.now += 0.250

    results = race(
        lambda: command(instance, lease),
        instance.check_watchdog,
    )

    command_result = results["left"]
    timeout_changed = results["right"]
    snapshot = instance.snapshot()

    if command_result.accepted:
        assert timeout_changed is False
        assert backend.commands == [(0.2, 0.0, 0.0)]
        assert backend.stop_count == 0
        assert snapshot.active_owner == OWNER_NAVIGATION
    else:
        assert timeout_changed is True
        assert backend.commands == []
        assert backend.stop_count == 1
        assert snapshot.active_owner == OWNER_NONE


def test_command_and_release_cannot_leave_partial_ownership():
    instance, backend, _clock, lease = arbiter()

    results = race(
        lambda: command(instance, lease),
        lambda: instance.release(OWNER_NAVIGATION, lease),
    )

    assert results["right"].released is True
    assert len(backend.commands) <= 1
    assert backend.stop_count == 1
    assert instance.snapshot().active_owner == OWNER_NONE
    assert instance.snapshot().stopped is True


def test_emergency_stop_wins_over_concurrent_command():
    instance, backend, _clock, lease = arbiter()

    race(
        lambda: command(instance, lease),
        instance.set_emergency_stop,
    )
    snapshot = instance.snapshot()

    assert len(backend.commands) <= 1
    assert backend.stop_count == 1
    assert snapshot.active_owner == OWNER_NONE
    assert snapshot.emergency_stop is True
    assert snapshot.stopped is True
    assert snapshot.reason == "EMERGENCY_STOP"
