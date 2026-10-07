"""Contract for physical G1 waist telemetry protocol."""

import json
import math

import pytest

from g1_hardware_bridge.joint_state.protocol import (
    JOINT_INDICES,
    JOINT_NAMES,
    OUTPUT_PREFIX,
    JointStateGate,
    decode_sample,
    encode_sample,
    normalize_sample,
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
    assert OUTPUT_PREFIX == "JOINT_STATE="


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
        JointStateGate(
            period_seconds=period,
            monotonic=lambda: 0.0,
        )


def test_repeated_tick_is_not_republished():
    gate = JointStateGate(
        monotonic=lambda: 1.0,
    )

    assert gate.offer(10, motor_positions()) is not None
    assert gate.offer(10, motor_positions()) is None


def test_rate_limits_advancing_ticks_without_sleeping():
    times = iter((1.0, 1.01, 1.02))
    gate = JointStateGate(
        period_seconds=0.02,
        monotonic=lambda: next(times),
    )

    first = gate.offer(10, motor_positions())
    early = gate.offer(11, motor_positions())
    due = gate.offer(12, motor_positions())

    assert first["tick"] == 10
    assert early is None
    assert due["tick"] == 12


def test_encodes_only_minimal_protocol_fields():
    sample = normalize_sample(
        123,
        motor_positions(),
    )

    line = encode_sample(sample)

    assert line.startswith(OUTPUT_PREFIX)
    payload = json.loads(
        line[len(OUTPUT_PREFIX):]
    )
    assert payload == {
        "tick": 123,
        "positions": [0.12, -0.13, 0.14],
    }
    assert set(payload) == {
        "tick",
        "positions",
    }


def test_decodes_valid_line_and_ignores_noise():
    line = (
        'JOINT_STATE={"positions":[0.12,-0.13,0.14],'
        '"tick":123}'
    )

    assert decode_sample(line) == {
        "tick": 123,
        "positions": [0.12, -0.13, 0.14],
    }
    assert decode_sample("native startup log") is None


@pytest.mark.parametrize(
    "line",
    [
        "JOINT_STATE=not-json",
        "JOINT_STATE=[]",
        "JOINT_STATE={}",
        'JOINT_STATE={"tick":1,"positions":[0.0,0.0]}',
        'JOINT_STATE={"tick":1,"positions":[0.0,0.0,NaN]}',
        (
            'JOINT_STATE={"tick":1,'
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
