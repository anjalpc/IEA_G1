"""Contract for physical G1 waist telemetry protocol."""

import json
import math

import pytest

from g1_hardware_bridge.low_state.protocol import (
    JOINT_INDICES,
    JOINT_NAMES,
    OUTPUT_PREFIX,
    LowStateGate,
    decode_sample,
    encode_sample,
    normalize_sample,
    normalize_atomic_sample,
)


def motor_positions():
    values = [0.01 * index for index in range(29)]
    values[12] = 0.12
    values[13] = -0.13
    values[14] = 0.14
    return values


def test_mode_five_waist_contract_is_exact():
    assert JOINT_NAMES == (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    assert JOINT_INDICES == (12, 13, 14)
    assert OUTPUT_PREFIX == "LOW_STATE="


def test_normalizes_only_required_positions():
    assert normalize_sample(
        123,
        motor_positions(),
    ) == {
        "tick": 123,
        "positions": [0.12, -0.13, 0.14],
    }


def test_rejects_short_motor_state():
    with pytest.raises(
        ValueError,
        match="at least 29",
    ):
        normalize_sample(
            123,
            motor_positions()[:28],
        )


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf],
)
def test_rejects_nonfinite_waist_position(value):
    positions = motor_positions()
    positions[13] = value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        normalize_sample(123, positions)


@pytest.mark.parametrize(
    "period",
    [0.0, -0.02, math.nan, math.inf],
)
def test_rejects_invalid_publish_period(period):
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        LowStateGate(
            period_seconds=period,
            monotonic=lambda: 0.0,
        )


def test_repeated_tick_is_not_republished():
    gate = LowStateGate(
        monotonic=lambda: 1.0,
    )

    imu = imu_measurement()

    assert gate.offer(
        tick=10,
        motor_positions=motor_positions(),
        **imu,
    ) is not None
    assert gate.offer(
        tick=10,
        motor_positions=motor_positions(),
        **imu,
    ) is None


def test_rate_limits_advancing_ticks_without_sleeping():
    times = iter((1.0, 1.01, 1.02))
    gate = LowStateGate(
        period_seconds=0.02,
        monotonic=lambda: next(times),
    )

    imu = imu_measurement()
    first = gate.offer(
        tick=10,
        motor_positions=motor_positions(),
        **imu,
    )
    early = gate.offer(
        tick=11,
        motor_positions=motor_positions(),
        **imu,
    )
    due = gate.offer(
        tick=12,
        motor_positions=motor_positions(),
        **imu,
    )

    assert first["tick"] == 10
    assert early is None
    assert due["tick"] == 12


def test_encodes_only_minimal_protocol_fields():
    imu = imu_measurement()
    sample = normalize_atomic_sample(
        tick=123,
        motor_positions=motor_positions(),
        **imu,
    )

    line = encode_sample(sample)

    assert line.startswith(OUTPUT_PREFIX)
    payload = json.loads(
        line[len(OUTPUT_PREFIX):]
    )
    assert payload == {
        "angular_velocity_xyz": [0.1, -0.2, 0.3],
        "linear_acceleration_xyz": [0.4, 0.5, 9.6],
        "orientation_wxyz": [0.5, 0.5, 0.5, 0.5],
        "positions": [0.12, -0.13, 0.14],
        "tick": 123,
    }
    assert set(payload) == {
        "tick",
        "positions",
        "orientation_wxyz",
        "angular_velocity_xyz",
        "linear_acceleration_xyz",
    }


def test_decodes_valid_line_and_ignores_noise():
    line = (
        'LOW_STATE={'
        '"angular_velocity_xyz":[0.1,-0.2,0.3],'
        '"linear_acceleration_xyz":[0.4,0.5,9.6],'
        '"orientation_wxyz":[0.5,0.5,0.5,0.5],'
        '"positions":[0.12,-0.13,0.14],'
        '"tick":123}'
    )

    assert decode_sample(line) == {
        "tick": 123,
        "positions": [0.12, -0.13, 0.14],
        "orientation_wxyz": [0.5, 0.5, 0.5, 0.5],
        "angular_velocity_xyz": [0.1, -0.2, 0.3],
        "linear_acceleration_xyz": [0.4, 0.5, 9.6],
    }
    assert decode_sample("native startup log") is None


@pytest.mark.parametrize(
    "line",
    [
        "LOW_STATE=not-json",
        "LOW_STATE=[]",
        "LOW_STATE={}",
        'LOW_STATE={"tick":1,"positions":[0.0,0.0]}',
        'LOW_STATE={"tick":1,"positions":[0.0,0.0,NaN]}',
        (
            'LOW_STATE={"tick":1,'
            '"positions":[0.0,0.0,0.0],'
            '"private":"forbidden"}'
        ),
    ],
)
def test_rejects_invalid_prefixed_payload(line):
    with pytest.raises(
        (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        )
    ):
        decode_sample(line)


def imu_measurement():
    return {
        "orientation_wxyz": [0.5, 0.5, 0.5, 0.5],
        "angular_velocity_xyz": [0.1, -0.2, 0.3],
        "linear_acceleration_xyz": [0.4, 0.5, 9.6],
    }


def test_normalizes_one_atomic_low_state_sample():
    imu = imu_measurement()

    assert normalize_atomic_sample(
        tick=123,
        motor_positions=motor_positions(),
        orientation_wxyz=imu["orientation_wxyz"],
        angular_velocity_xyz=imu["angular_velocity_xyz"],
        linear_acceleration_xyz=imu["linear_acceleration_xyz"],
    ) == {
        "tick": 123,
        "positions": [0.12, -0.13, 0.14],
        "orientation_wxyz": [0.5, 0.5, 0.5, 0.5],
        "angular_velocity_xyz": [0.1, -0.2, 0.3],
        "linear_acceleration_xyz": [0.4, 0.5, 9.6],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("orientation_wxyz", [1.0, 0.0, 0.0]),
        ("angular_velocity_xyz", [0.0, 0.0]),
        ("linear_acceleration_xyz", [0.0, 0.0]),
    ],
)
def test_rejects_invalid_imu_field_lengths(field, value):
    imu = imu_measurement()
    imu[field] = value

    with pytest.raises(ValueError):
        normalize_atomic_sample(
            tick=123,
            motor_positions=motor_positions(),
            **imu,
        )


@pytest.mark.parametrize(
    ("field", "index"),
    [
        ("orientation_wxyz", 2),
        ("angular_velocity_xyz", 1),
        ("linear_acceleration_xyz", 0),
    ],
)
def test_rejects_nonfinite_imu_values(field, index):
    imu = imu_measurement()
    imu[field][index] = math.nan

    with pytest.raises(ValueError, match="finite"):
        normalize_atomic_sample(
            tick=123,
            motor_positions=motor_positions(),
            **imu,
        )


@pytest.mark.parametrize(
    "orientation_wxyz",
    [
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
    ],
)
def test_rejects_invalid_quaternion_norm(orientation_wxyz):
    imu = imu_measurement()
    imu["orientation_wxyz"] = orientation_wxyz

    with pytest.raises(ValueError, match="unit length"):
        normalize_atomic_sample(
            tick=123,
            motor_positions=motor_positions(),
            **imu,
        )


def test_invalid_atomic_sample_does_not_consume_tick():
    gate = LowStateGate(
        monotonic=lambda: 1.0,
    )
    invalid = imu_measurement()
    invalid["angular_velocity_xyz"][0] = math.nan

    with pytest.raises(ValueError, match="finite"):
        gate.offer(
            tick=10,
            motor_positions=motor_positions(),
            **invalid,
        )

    accepted = gate.offer(
        tick=10,
        motor_positions=motor_positions(),
        **imu_measurement(),
    )

    assert accepted["tick"] == 10
