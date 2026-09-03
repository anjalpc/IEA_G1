"""Tests for guarded presentation trajectory and state validation."""

import math

import pytest

from g1_hardware_bridge.experimental.left_arm_safety import (
    ArmState,
    LEFT_CARD_PRESENTATION_CANDIDATE,
    PresentationLimits,
    captured_arm_positions,
    interpolate,
    presentation_target,
    smoothstep5,
    validate_command_step,
    validate_state,
)


def stable_state(now=10.0):
    positions = tuple(index / 100.0 for index in range(35))
    return ArmState(now, 1, positions, (0.0,) * 35)


def test_quintic_trajectory_has_exact_endpoints():
    assert smoothstep5(0.0) == 0.0
    assert smoothstep5(1.0) == 1.0
    assert interpolate((1.0, 2.0), (3.0, 6.0), 0.0) == (1.0, 2.0)
    assert interpolate((1.0, 2.0), (3.0, 6.0), 1.0) == (3.0, 6.0)


def test_target_moves_only_left_arm_toward_candidate():
    captured = captured_arm_positions(stable_state())
    target = presentation_target(captured, 0.25)
    assert target[7:] == captured[7:]
    for actual, initial, final in zip(
            target[:7], captured[:7], LEFT_CARD_PRESENTATION_CANDIDATE):
        assert actual == pytest.approx(initial + 0.25 * (final - initial))


def test_rejects_stale_moving_unstable_and_nonfinite_state():
    with pytest.raises(ValueError, match="stale"):
        validate_state(stable_state(9.0), 10.0)
    moving = ArmState(10.0, 1, (0.0,) * 35, (0.2,) * 35)
    with pytest.raises(ValueError, match="stationary"):
        validate_state(moving, 10.0)
    tilted = ArmState(
        10.0, 1, (0.0,) * 35, (0.0,) * 35, rpy=(0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="roll or pitch"):
        validate_state(tilted, 10.0)
    positions = [0.0] * 35
    positions[15] = math.nan
    invalid = ArmState(10.0, 1, tuple(positions), (0.0,) * 35)
    with pytest.raises(ValueError, match="non-finite"):
        validate_state(invalid, 10.0)


def test_command_step_enforces_velocity_and_acceleration():
    limits = PresentationLimits(
        max_velocity_rad_s=1.0, max_acceleration_rad_s2=2.0)
    velocity = validate_command_step(
        (0.0,), (0.01,), (0.0,), 0.1, limits)
    assert velocity == pytest.approx((0.1,))
    with pytest.raises(ValueError, match="velocity"):
        validate_command_step((0.0,), (0.2,), (0.0,), 0.1, limits)
    with pytest.raises(ValueError, match="acceleration"):
        validate_command_step((0.0,), (0.01,), (-0.2,), 0.1, limits)
