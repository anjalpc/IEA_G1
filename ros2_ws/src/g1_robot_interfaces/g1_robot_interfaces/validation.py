"""Shared validation for backend-neutral G1 command payloads."""

import math


MIN_HAND_RAMP_S = 0.2
MAX_HAND_RAMP_S = 5.0


def validate_hand_name(value):
    """Return a supported hand name."""
    hand = str(value)
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    return hand


def validate_hand_target(value, parameter_name="hand target"):
    """Return a six-channel normalized hand target."""
    target = [float(position) for position in value]
    if len(target) != 6:
        raise ValueError(f"{parameter_name} must contain six positions")
    if any(
        not math.isfinite(position) or position < 0.0 or position > 1.0
        for position in target
    ):
        raise ValueError(f"{parameter_name} must be between 0 and 1")
    return target


def validate_hand_ramp_seconds(value, parameter_name="ramp_s"):
    """Return a hand ramp duration within the shared safety envelope."""
    duration = float(value)
    if (
        not math.isfinite(duration)
        or not MIN_HAND_RAMP_S <= duration <= MAX_HAND_RAMP_S
    ):
        raise ValueError(
            f"{parameter_name} must be between "
            f"{MIN_HAND_RAMP_S} and {MAX_HAND_RAMP_S} seconds")
    return duration
