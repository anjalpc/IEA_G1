"""Fail-closed tests for physical pelvis-IMU qualification."""

import math

from g1_state_estimation.pelvis_imu_health import (
    PelvisImuHealthBoundary,
    PelvisImuHealthLimits,
)


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def boundary(*, warmup_samples=3):
    clock = FakeClock()
    limits = PelvisImuHealthLimits(
        stale_after_seconds=0.10,
        warmup_samples=warmup_samples,
        minimum_acceleration_norm=4.0,
        maximum_acceleration_norm=15.0,
        maximum_angular_velocity_norm=10.0,
    )
    return (
        PelvisImuHealthBoundary(
            limits=limits,
            monotonic=clock,
        ),
        clock,
    )


def observe(
    instance,
    *,
    stamp=100.0,
    frame_id="imu_in_pelvis",
    orientation=(0.0, 0.0, 0.0, 1.0),
    angular_velocity=(0.0, 0.0, 0.0),
    linear_acceleration=(0.0, 0.0, 9.81),
    covariance=(0.0,) * 9,
):
    instance.observe(
        stamp=stamp,
        frame_id=frame_id,
        orientation=orientation,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
        orientation_covariance=covariance,
        angular_velocity_covariance=covariance,
        linear_acceleration_covariance=covariance,
    )


def test_requires_advancing_warmup_samples():
    instance, _clock = boundary(warmup_samples=3)

    observe(instance, stamp=100.0)
    observe(instance, stamp=100.01)

    assert instance.snapshot().ready is False
    assert instance.snapshot().reason == "WARMING_UP"

    observe(instance, stamp=100.02)

    assert instance.snapshot().ready is True
    assert instance.snapshot().reason == "PELVIS_IMU_HEALTHY"


def test_rejects_wrong_frame_and_resets_warmup():
    instance, _clock = boundary(warmup_samples=1)

    observe(instance, frame_id="wrong_imu")

    assert instance.snapshot().ready is False
    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"


def test_rejects_nonadvancing_timestamp():
    instance, _clock = boundary(warmup_samples=2)

    observe(instance, stamp=100.0)
    observe(instance, stamp=100.0)

    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"


def test_rejects_nonfinite_or_nonunit_orientation():
    instance, _clock = boundary(warmup_samples=1)

    observe(
        instance,
        orientation=(math.nan, 0.0, 0.0, 1.0),
    )
    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"

    observe(
        instance,
        stamp=100.1,
        orientation=(0.0, 0.0, 0.0, 2.0),
    )
    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"


def test_rejects_implausible_acceleration():
    instance, _clock = boundary(warmup_samples=1)

    observe(
        instance,
        linear_acceleration=(0.0, 0.0, 2.0),
    )
    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"

    observe(
        instance,
        stamp=100.1,
        linear_acceleration=(0.0, 0.0, 20.0),
    )
    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"


def test_rejects_implausible_angular_velocity():
    instance, _clock = boundary(warmup_samples=1)

    observe(
        instance,
        angular_velocity=(11.0, 0.0, 0.0),
    )

    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"


def test_requires_pinned_unknown_covariance_contract():
    instance, _clock = boundary(warmup_samples=1)

    covariance = [0.0] * 9
    covariance[0] = 0.1
    observe(instance, covariance=tuple(covariance))

    assert instance.snapshot().reason == "PELVIS_IMU_INVALID"


def test_fails_closed_when_input_becomes_stale():
    instance, clock = boundary(warmup_samples=1)
    observe(instance)

    assert instance.snapshot().ready is True

    clock.now += 0.11

    assert instance.snapshot().ready is False
    assert instance.snapshot().reason == "PELVIS_IMU_STALE"
