"""Pure safety and trajectory logic for guarded G1 arm presentation."""

from dataclasses import dataclass
import math


LEFT_ARM_INDICES = tuple(range(15, 22))
RIGHT_ARM_INDICES = tuple(range(22, 29))
ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
WEIGHT_INDEX = 29

# Mirrored from median right-arm rt/lowstate measurements after action 27.
# This is a simulation/shadow candidate, not an enabled physical calibration.
LEFT_CARD_PRESENTATION_CANDIDATE = (
    -0.108721,
    0.063594,
    -0.094819,
    0.182400,
    -0.159618,
    -0.339249,
    0.061755,
)


@dataclass(frozen=True)
class ArmState:
    """Minimal observed state used by the guarded controller."""

    received_at: float
    mode_machine: int
    positions: tuple
    velocities: tuple
    rpy: tuple = (0.0, 0.0, 0.0)
    omega: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PresentationLimits:
    """Conservative limits shared by tests, shadow mode, and hardware."""

    state_timeout_s: float = 0.05
    stationary_velocity_rad_s: float = 0.15
    max_velocity_rad_s: float = 0.35
    max_acceleration_rad_s2: float = 0.7
    max_roll_pitch_rad: float = 0.0872664626
    max_angular_velocity_rad_s: float = 0.35


def smoothstep5(fraction):
    """Return a quintic interpolation fraction with smooth endpoints."""
    value = float(fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    return value ** 3 * (10.0 + value * (-15.0 + 6.0 * value))


def interpolate(start, target, fraction):
    """Interpolate equal-length joint vectors with a quintic time law."""
    if len(start) != len(target):
        raise ValueError("start and target must have equal lengths")
    blend = smoothstep5(fraction)
    return tuple(
        initial + blend * (final - initial)
        for initial, final in zip(start, target)
    )


def validate_state(state, now, limits=PresentationLimits()):
    """Reject incomplete, stale, moving, or unstable observed state."""
    if len(state.positions) <= max(ARM_INDICES):
        raise ValueError("state does not contain the 29 G1 body joints")
    if len(state.velocities) <= max(ARM_INDICES):
        raise ValueError("state does not contain all arm velocities")
    values = (
        tuple(state.positions[index] for index in ARM_INDICES)
        + tuple(state.velocities[index] for index in ARM_INDICES)
        + tuple(state.rpy[:2]) + tuple(state.omega)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("state contains a non-finite value")
    age = float(now) - float(state.received_at)
    if age < 0.0 or age > limits.state_timeout_s:
        raise ValueError("state is stale")
    if max(abs(state.velocities[index]) for index in ARM_INDICES) > \
            limits.stationary_velocity_rad_s:
        raise ValueError("arms are not stationary")
    if max(abs(value) for value in state.rpy[:2]) > \
            limits.max_roll_pitch_rad:
        raise ValueError("torso roll or pitch exceeds the shadow limit")
    if max(abs(value) for value in state.omega) > \
            limits.max_angular_velocity_rad_s:
        raise ValueError("torso angular velocity exceeds the shadow limit")
    return state


def captured_arm_positions(state):
    """Return the measured dual-arm vector in Unitree Arm SDK order."""
    return tuple(state.positions[index] for index in ARM_INDICES)


def presentation_target(captured, fraction=1.0):
    """Create a dual-arm target while preserving the measured right arm."""
    if len(captured) != 14:
        raise ValueError("captured arm vector must contain 14 joints")
    amount = float(fraction)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("presentation fraction must be between 0 and 1")
    left = tuple(
        initial + amount * (target - initial)
        for initial, target in zip(
            captured[:7], LEFT_CARD_PRESENTATION_CANDIDATE)
    )
    return left + tuple(captured[7:])


def validate_command_step(previous, current, previous_velocity, dt,
                          limits=PresentationLimits()):
    """Check command velocity and acceleration without modifying targets."""
    if len(previous) != len(current) or len(current) != len(previous_velocity):
        raise ValueError("command vectors must have equal lengths")
    if dt <= 0.0:
        raise ValueError("control period must be positive")
    velocity = tuple(
        (next_value - prior_value) / dt
        for prior_value, next_value in zip(previous, current)
    )
    if max(abs(value) for value in velocity) > limits.max_velocity_rad_s:
        raise ValueError("command exceeds the arm velocity limit")
    acceleration = tuple(
        (value - prior_value) / dt
        for value, prior_value in zip(velocity, previous_velocity)
    )
    if max(abs(value) for value in acceleration) > \
            limits.max_acceleration_rad_s2:
        raise ValueError("command exceeds the arm acceleration limit")
    return velocity
