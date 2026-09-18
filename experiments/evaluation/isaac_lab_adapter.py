"""Convert live Isaac Lab measurements into independent evaluator samples."""

from __future__ import annotations

from dataclasses import dataclass
import math

from experiments.evaluation.stair_outcomes import StepSample


@dataclass(frozen=True)
class IsaacLabStepState:
    """Simulator values captured for one environment at one control step."""

    time_s: float
    root_position_world_m: tuple[float, float, float]
    environment_origin_world_m: tuple[float, float, float]
    root_quaternion_wxyz: tuple[float, float, float, float]
    contacts: tuple[str, ...] = ()
    joint_violation: bool = False
    velocity_violation: bool = False
    torque_violation: bool = False
    slip_violation: bool = False
    timed_out: bool = False


def quaternion_wxyz_to_roll_pitch(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return roll and pitch from a normalized-or-near-normalized quaternion."""
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        return (math.nan, math.nan)

    w, x, y, z = (value / norm for value in quaternion)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    return (roll, pitch)


def course_x_m(
    state: IsaacLabStepState,
    *,
    start_x_m: float = 1.0,
) -> float:
    """Convert world X into the evaluator's staircase-course coordinate."""
    return (
        state.root_position_world_m[0]
        - state.environment_origin_world_m[0]
        + start_x_m
    )


def step_index_from_course_x(
    base_x_m: float,
    *,
    step_edges_m: tuple[float, ...] = (2.0, 2.25, 2.50),
) -> int:
    """Return the highest stair level whose leading edge was reached."""
    return sum(base_x_m >= edge for edge in step_edges_m)


def to_step_sample(
    state: IsaacLabStepState,
    *,
    start_x_m: float = 1.0,
    step_edges_m: tuple[float, ...] = (2.0, 2.25, 2.50),
) -> StepSample:
    """Create one reward-independent evaluator sample."""
    base_x = course_x_m(state, start_x_m=start_x_m)
    roll, pitch = quaternion_wxyz_to_roll_pitch(
        state.root_quaternion_wxyz
    )
    return StepSample.create(
        time_s=state.time_s,
        base_x_m=base_x,
        base_height_m=(
            state.root_position_world_m[2]
            - state.environment_origin_world_m[2]
        ),
        roll_rad=roll,
        pitch_rad=pitch,
        step_index=step_index_from_course_x(
            base_x,
            step_edges_m=step_edges_m,
        ),
        contacts=state.contacts,
        joint_violation=state.joint_violation,
        velocity_violation=state.velocity_violation,
        torque_violation=state.torque_violation,
        slip_violation=state.slip_violation,
        timed_out=state.timed_out,
    )
