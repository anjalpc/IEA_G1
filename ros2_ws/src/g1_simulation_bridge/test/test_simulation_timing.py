"""Contract tests for simulation-time publication scheduling."""

import math

import pytest

from g1_simulation_bridge.simulation_timing import (
    LidarPublicationSchedule,
    SimulationPublicationSchedule,
    SimulationTimestamp,
    simulation_timestamp,
)


def test_converts_simulation_seconds_to_ros_timestamp():
    assert simulation_timestamp(12.345678901) == (
        SimulationTimestamp(sec=12, nanosec=345678901)
    )


def test_normalizes_nanoseconds_rounded_into_next_second():
    assert simulation_timestamp(1.9999999996) == (
        SimulationTimestamp(sec=2, nanosec=0)
    )


def test_does_not_publish_raw_state_at_simulation_start():
    schedule = SimulationPublicationSchedule(
        state_period_seconds=0.02,
    )

    decision = schedule.after_step(0.0)

    assert decision.clock_stamp == SimulationTimestamp(
        sec=0,
        nanosec=0,
    )
    assert decision.publish_state is False


def test_publishes_clock_every_step_and_state_at_50_hz():
    schedule = SimulationPublicationSchedule(
        state_period_seconds=0.02,
    )
    step_times = [
        0.005,
        0.010,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
    ]

    decisions = [
        schedule.after_step(step_time)
        for step_time in step_times
    ]

    assert [
        decision.clock_stamp
        for decision in decisions
    ] == [
        simulation_timestamp(step_time)
        for step_time in step_times
    ]
    assert [
        decision.publish_state
        for decision in decisions
    ] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


def test_uses_simulation_time_instead_of_callback_count():
    schedule = SimulationPublicationSchedule(
        state_period_seconds=0.02,
    )
    irregular_step_times = [
        0.003,
        0.008,
        0.013,
        0.019,
        0.020,
        0.027,
    ]

    decisions = [
        schedule.after_step(step_time)
        for step_time in irregular_step_times
    ]

    assert [
        decision.publish_state
        for decision in decisions
    ] == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "period",
    [
        0.0,
        -0.02,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_state_period(period):
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        SimulationPublicationSchedule(
            state_period_seconds=period,
        )


@pytest.mark.parametrize(
    "simulation_time",
    [
        -0.001,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_simulation_time(simulation_time):
    schedule = SimulationPublicationSchedule(
        state_period_seconds=0.02,
    )

    with pytest.raises(
        ValueError,
        match="finite and non-negative",
    ):
        schedule.after_step(simulation_time)


@pytest.mark.parametrize(
    "next_time",
    [
        0.010,
        0.005,
    ],
)
def test_rejects_duplicate_or_reversed_simulation_time(
    next_time,
):
    schedule = SimulationPublicationSchedule(
        state_period_seconds=0.02,
    )
    schedule.after_step(0.010)

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        schedule.after_step(next_time)


def test_schedules_lidar_at_ten_hz_simulation_time():
    schedule = LidarPublicationSchedule(
        period_seconds=0.1,
    )

    decisions = [
        schedule.after_step(simulation_time)
        for simulation_time in (
            0.02,
            0.08,
            0.10,
            0.19,
            0.20,
        )
    ]

    assert [
        decision.publish_lidar
        for decision in decisions
    ] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert decisions[2].stamp == (
        SimulationTimestamp(
            sec=0,
            nanosec=100000000,
        )
    )
    assert decisions[4].stamp == (
        SimulationTimestamp(
            sec=0,
            nanosec=200000000,
        )
    )


def test_lidar_schedule_uses_elapsed_simulation_time():
    schedule = LidarPublicationSchedule(
        period_seconds=0.1,
    )

    decisions = [
        schedule.after_step(simulation_time)
        for simulation_time in (
            0.03,
            0.25,
            0.29,
            0.30,
        )
    ]

    assert [
        decision.publish_lidar
        for decision in decisions
    ] == [
        False,
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(
    "period_seconds",
    [
        0.0,
        -0.1,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_lidar_period(
    period_seconds,
):
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        LidarPublicationSchedule(
            period_seconds=period_seconds,
        )
