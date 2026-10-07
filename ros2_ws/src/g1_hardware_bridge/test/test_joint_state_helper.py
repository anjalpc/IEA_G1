"""Hardware-free tests for the native joint-state helper."""

import io
import math
from pathlib import Path
from types import SimpleNamespace

from g1_hardware_bridge.joint_state.helper import (
    ERROR_PREFIX,
    LowStateReader,
)
from g1_hardware_bridge.joint_state.protocol import (
    JointStateGate,
    decode_sample,
)


def low_state(
    *,
    tick=123,
    count=29,
    yaw=0.12,
    roll=-0.13,
    pitch=0.14,
):
    motors = [
        SimpleNamespace(q=0.01 * index)
        for index in range(count)
    ]
    if count > 14:
        motors[12].q = yaw
        motors[13].q = roll
        motors[14].q = pitch
    return SimpleNamespace(
        tick=tick,
        motor_state=motors,
    )


def test_emits_valid_minimal_protocol_line():
    lines = []
    reader = LowStateReader(
        gate=JointStateGate(
            monotonic=lambda: 1.0
        ),
        emit=lines.append,
    )

    assert reader.receive_state(low_state()) is True

    assert len(lines) == 1
    assert decode_sample(lines[0]) == {
        "tick": 123,
        "positions": [0.12, -0.13, 0.14],
    }


def test_repeated_and_rate_limited_samples_do_not_emit():
    times = iter((1.0, 1.01, 1.02))
    lines = []
    reader = LowStateReader(
        gate=JointStateGate(
            period_seconds=0.02,
            monotonic=lambda: next(times),
        ),
        emit=lines.append,
    )

    assert reader.receive_state(
        low_state(tick=10)
    ) is True
    assert reader.receive_state(
        low_state(tick=10)
    ) is False
    assert reader.receive_state(
        low_state(tick=11)
    ) is False
    assert reader.receive_state(
        low_state(tick=12)
    ) is True

    assert [
        decode_sample(line)["tick"]
        for line in lines
    ] == [10, 12]


def test_invalid_native_samples_are_suppressed():
    lines = []
    errors = io.StringIO()
    reader = LowStateReader(
        emit=lines.append,
        error_stream=errors,
    )

    invalid = (
        low_state(count=28),
        low_state(roll=math.nan),
        SimpleNamespace(
            tick=123,
            motor_state=None,
        ),
    )

    for message in invalid:
        assert reader.receive_state(message) is False

    assert lines == []
    error_lines = errors.getvalue().splitlines()
    assert len(error_lines) == 3
    assert all(
        line == ERROR_PREFIX + "invalid sample"
        for line in error_lines
    )


def test_helper_source_is_read_only():
    source = (
        Path(__file__).parents[1]
        / "g1_hardware_bridge"
        / "joint_state"
        / "helper.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "ChannelPublisher",
        "LowCmd",
        "rt/lowcmd",
        "rt/arm_sdk",
    )
    for token in forbidden:
        assert token not in source
