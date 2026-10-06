"""Contract tests for state-estimation health evaluation."""

import math

import pytest

from g1_state_estimation.health import (
    HealthState,
    ObservationQuality,
    StateHealthMonitor,
)
from g1_state_estimation.validation import ValidationResult


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def valid_result():
    return ValidationResult(True, "STATE_VALID")


def monitor():
    clock = FakeClock()
    instance = StateHealthMonitor(
        stale_after_seconds=0.5,
        monotonic=clock,
    )
    return instance, clock


def test_starts_offline_and_not_ready():
    instance, _clock = monitor()

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.OFFLINE
    assert snapshot.ready is False
    assert snapshot.reason == "NO_STATE"


def test_waits_for_odometry_after_first_valid_imu():
    instance, _clock = monitor()

    instance.observe_imu(valid_result())
    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.INITIALIZING
    assert snapshot.ready is False
    assert snapshot.reason == "WAITING_FOR_ODOMETRY"


def test_waits_for_imu_after_first_valid_odometry():
    instance, _clock = monitor()

    instance.observe_odometry(valid_result())
    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.INITIALIZING
    assert snapshot.ready is False
    assert snapshot.reason == "WAITING_FOR_IMU"


def test_becomes_valid_only_after_both_streams_are_valid():
    instance, _clock = monitor()

    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())
    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.VALID
    assert snapshot.ready is True
    assert snapshot.reason == "STATE_VALID"


def invalid_result(private_reason="PRIVATE_SOURCE_DETAIL"):
    return ValidationResult(False, private_reason)


def test_invalid_observation_fails_closed():
    instance, _clock = monitor()
    instance.observe_imu(invalid_result())
    instance.observe_odometry(valid_result())

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.INVALID
    assert snapshot.ready is False
    assert snapshot.reason == "STATE_INVALID"


def test_invalid_reason_is_not_exposed():
    instance, _clock = monitor()
    instance.observe_imu(
        invalid_result("private DDS interface and device identifier")
    )
    instance.observe_odometry(valid_result())

    snapshot = instance.snapshot()

    assert snapshot.reason == "STATE_INVALID"
    assert "DDS" not in repr(snapshot)
    assert "device" not in repr(snapshot)


def test_both_streams_become_stale_after_timeout():
    instance, clock = monitor()
    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())
    clock.now += 0.501

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.STALE
    assert snapshot.ready is False
    assert snapshot.reason == "STATE_STALE"


def test_refreshing_only_one_stream_does_not_restore_readiness():
    instance, clock = monitor()
    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())
    clock.now += 0.501
    instance.observe_imu(valid_result())

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.STALE
    assert snapshot.ready is False
    assert snapshot.reason == "STATE_STALE"


def test_fresh_valid_inputs_recover_from_stale_state():
    instance, clock = monitor()
    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())
    clock.now += 0.501

    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())
    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.VALID
    assert snapshot.ready is True
    assert snapshot.reason == "STATE_VALID"


@pytest.mark.parametrize(
    "timeout",
    [0.0, -0.1, math.inf, math.nan],
)
def test_rejects_invalid_stale_timeout(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        StateHealthMonitor(
            stale_after_seconds=timeout,
            monotonic=FakeClock(),
        )


def test_degraded_input_blocks_navigation_readiness():
    instance, _clock = monitor()
    instance.observe_imu(
        valid_result(),
        quality=ObservationQuality.DEGRADED,
    )
    instance.observe_odometry(valid_result())

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.DEGRADED
    assert snapshot.ready is False
    assert snapshot.reason == "STATE_DEGRADED"


def test_fresh_valid_input_recovers_from_degraded_state():
    instance, _clock = monitor()
    instance.observe_imu(
        valid_result(),
        quality=ObservationQuality.DEGRADED,
    )
    instance.observe_odometry(valid_result())

    instance.observe_imu(valid_result())
    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.VALID
    assert snapshot.ready is True
    assert snapshot.reason == "STATE_VALID"


def test_rejects_unsupported_observation_quality():
    instance, _clock = monitor()

    with pytest.raises(ValueError, match="observation quality"):
        instance.observe_imu(
            valid_result(),
            quality="degraded",
        )


def test_state_remains_fresh_at_exact_timeout_boundary():
    instance, clock = monitor()
    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())
    clock.now += 0.5

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.VALID
    assert snapshot.ready is True


def test_nonfinite_monotonic_time_fails_closed():
    instance, clock = monitor()
    clock.now = math.inf
    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.INVALID
    assert snapshot.ready is False
    assert snapshot.reason == "TIME_SOURCE_INVALID"


def test_monotonic_clock_rollback_fails_closed():
    instance, clock = monitor()
    instance.observe_imu(valid_result())
    instance.observe_odometry(valid_result())
    clock.now -= 0.1

    snapshot = instance.snapshot()

    assert snapshot.state is HealthState.INVALID
    assert snapshot.ready is False
    assert snapshot.reason == "TIME_SOURCE_INVALID"
