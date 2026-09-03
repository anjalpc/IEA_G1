import pytest

from g1_hardware_bridge.brainco_hand_helper import (
    COMMAND_CONFIRMATION,
    command_position,
)


def test_command_position_requires_explicit_interlock():
    with pytest.raises(ValueError, match="requires --confirm"):
        command_position("eth0", "right", [0.1] * 6, 0.75, 5.0, "")


def test_command_position_requires_six_positions():
    with pytest.raises(ValueError, match="six positions"):
        command_position(
            "eth0", "right", [0.1] * 5, 0.75, 5.0,
            COMMAND_CONFIRMATION)


@pytest.mark.parametrize("target", [[-0.01] * 6, [1.01] * 6])
def test_command_position_rejects_out_of_range_target(target):
    with pytest.raises(ValueError, match="between 0 and 1"):
        command_position(
            "eth0", "right", target, 0.75, 5.0,
            COMMAND_CONFIRMATION)


def test_command_position_rejects_unsupported_hand():
    with pytest.raises(ValueError, match="unsupported hand"):
        command_position(
            "eth0", "invalid", [0.1] * 6, 0.75, 5.0,
            COMMAND_CONFIRMATION)


@pytest.mark.parametrize("ramp_seconds", [0.19, 5.01])
def test_command_position_rejects_unsafe_ramp(ramp_seconds):
    with pytest.raises(ValueError, match="ramp duration must be between"):
        command_position(
            "eth0", "right", [0.1] * 6, ramp_seconds, 5.0,
            COMMAND_CONFIRMATION)
