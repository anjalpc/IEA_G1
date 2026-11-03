"""Fail-closed contract tests for the GLIM navigation health boundary."""

import math

from g1_state_estimation.glim_health import (
    GlimHealthBoundary,
    GlimHealthLimits,
)


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def boundary(*, warmup_samples=3):
    clock = FakeClock()
    limits = GlimHealthLimits(
        odometry_stale_after_seconds=0.35,
        cloud_stale_after_seconds=0.35,
        imu_stale_after_seconds=0.10,
        warmup_samples=warmup_samples,
        maximum_step_translation_m=0.15,
        maximum_step_rotation_rad=math.radians(15.0),
    )
    return GlimHealthBoundary(limits=limits, monotonic=clock), clock


def observe_inputs(instance, *, stamp=100.0):
    instance.observe_cloud(stamp=stamp, frame_id="mid360_link")
    instance.observe_imu(stamp=stamp, frame_id="livox_frame")


def observe_odometry(instance, *, stamp, x=0.0, yaw=0.0):
    instance.observe_odometry(
        stamp=stamp,
        frame_id="glim_odom",
        child_frame_id="livox_frame",
        position=(x, 0.0, 0.0),
        orientation=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
    )


def test_requires_all_three_live_streams():
    instance, _clock = boundary()
    observe_odometry(instance, stamp=100.0)

    snapshot = instance.snapshot()

    assert snapshot.ready is False
    assert snapshot.reason == "WAITING_FOR_INPUTS"


def test_requires_consecutive_warmup_samples():
    instance, _clock = boundary(warmup_samples=3)
    observe_inputs(instance)

    observe_odometry(instance, stamp=100.0)
    observe_odometry(instance, stamp=100.1)

    assert instance.snapshot().ready is False
    assert instance.snapshot().reason == "WARMING_UP"

    observe_odometry(instance, stamp=100.2)

    assert instance.snapshot().ready is True
    assert instance.snapshot().reason == "GLIM_HEALTHY"


def test_rejects_non_monotonic_odometry_timestamp():
    instance, _clock = boundary()
    observe_inputs(instance)
    observe_odometry(instance, stamp=100.0)
    observe_odometry(instance, stamp=100.0)

    snapshot = instance.snapshot()

    assert snapshot.ready is False
    assert snapshot.reason == "ODOMETRY_INVALID"


def test_rejects_wrong_contract_frames():
    instance, _clock = boundary()
    instance.observe_cloud(stamp=100.0, frame_id="wrong_lidar")
    instance.observe_imu(stamp=100.0, frame_id="livox_frame")
    observe_odometry(instance, stamp=100.0)

    snapshot = instance.snapshot()

    assert snapshot.ready is False
    assert snapshot.reason == "INPUT_INVALID"


def test_rejects_nonfinite_pose():
    instance, _clock = boundary()
    observe_inputs(instance)
    instance.observe_odometry(
        stamp=100.0,
        frame_id="glim_odom",
        child_frame_id="livox_frame",
        position=(math.nan, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )

    assert instance.snapshot().reason == "ODOMETRY_INVALID"


def test_rejects_translation_jump_and_resets_warmup():
    instance, _clock = boundary(warmup_samples=2)
    observe_inputs(instance)
    observe_odometry(instance, stamp=100.0, x=0.0)
    observe_odometry(instance, stamp=100.1, x=0.01)
    assert instance.snapshot().ready is True

    observe_odometry(instance, stamp=100.2, x=0.30)

    assert instance.snapshot().ready is False
    assert instance.snapshot().reason == "ODOMETRY_JUMP"


def test_rejects_rotation_jump():
    instance, _clock = boundary()
    observe_inputs(instance)
    observe_odometry(instance, stamp=100.0, yaw=0.0)
    observe_odometry(instance, stamp=100.1, yaw=math.radians(16.0))

    assert instance.snapshot().reason == "ODOMETRY_JUMP"


def test_each_required_stream_fails_closed_when_stale():
    instance, clock = boundary(warmup_samples=1)
    observe_inputs(instance)
    observe_odometry(instance, stamp=100.0)
    assert instance.snapshot().ready is True

    clock.now += 0.11

    snapshot = instance.snapshot()
    assert snapshot.ready is False
    assert snapshot.reason == "INPUT_STALE"
