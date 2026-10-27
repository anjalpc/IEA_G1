"""Pure footstep-policy contract: observations, continuous scores and statistics.

No Isaac import and no dependency on historical supported-ascent rewards. The
simulator adapter must provide measured contacts, actions and terrain-specific
references; this module does not assert physical stair success.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot, pi, sin

from .footstep_targets import FootstepProgress, relative_target


@dataclass(frozen=True)
class FootstepLearningConfig:
    target_position_scale_m: float
    velocity_error_scale_mps: float
    target_distance_scale_m: float
    contact_force_scale_n: float
    swing_speed_scale_mps: float

    def __post_init__(self) -> None:
        if min(self.target_position_scale_m, self.velocity_error_scale_mps,
               self.target_distance_scale_m, self.contact_force_scale_n,
               self.swing_speed_scale_mps) <= 0:
            raise ValueError('learning scales must be positive')


def observation(progress: FootstepProgress, *, root_x: float, root_y: float,
                root_yaw: float, root_z: float, phase_fraction: float,
                cfg: FootstepLearningConfig) -> tuple[float, ...]:
    """22 channels: clock(2), next two targets (2 * 10).

    Per target: active, left, right, dx, dy, dz, dyaw, normalized level,
    sequence-position fraction, is-current. Empty targets have zero channels.
    No future-state or success/failure-label leakage.
    """
    if not 0 <= phase_fraction < 1:
        raise ValueError('phase_fraction must be in [0, 1)')
    result = [sin(2*pi*phase_fraction), cos(2*pi*phase_fraction)]
    for offset, target in enumerate(progress.next_two()):
        if target is None:
            result.extend((0.0,) * 10)
            continue
        x, y, z, yaw = relative_target(target, root_x, root_y, root_yaw)
        result.extend((
            1.0, float(target.foot == 0), float(target.foot == 1),
            x/cfg.target_position_scale_m, y/cfg.target_position_scale_m,
            (z-root_z)/cfg.target_position_scale_m, yaw/pi,
            target.terrain_level / max(1, max(t.terrain_level for t in progress.targets)),
            target.sequence_index / max(1, len(progress.targets)-1),
            float(offset == 0),
        ))
    return tuple(result)


def learning_scores(*, assigned_foot_distance_m: float, root_midpoint_distance_m: float,
                    measured_forward_mps: float, commanded_forward_mps: float,
                    swing_foot_force_n: float, support_foot_force_n: float,
                    swing_foot_speed_mps: float, phase_swing_fraction: float,
                    cfg: FootstepLearningConfig) -> dict[str, float]:
    """Independent continuous scores; no strict-event or one-shot bonus.

    phase_swing_fraction is a caller-supplied desired swing activation in [0,1].
    Contact phase score rewards unload of swing foot and loaded support foot;
    the velocity score encourages swing motion only during scheduled swing.
    These scores must be empirically audited before use as PPO rewards.
    """
    vals = (assigned_foot_distance_m, root_midpoint_distance_m,
            swing_foot_force_n, support_foot_force_n, swing_foot_speed_mps)
    if any(v < 0 for v in vals) or not 0 <= phase_swing_fraction <= 1:
        raise ValueError('invalid unsigned measurements or phase')
    swing_load = min(1., swing_foot_force_n / cfg.contact_force_scale_n)
    support_load = min(1., support_foot_force_n / cfg.contact_force_scale_n)
    normalized_swing_speed = min(1., swing_foot_speed_mps / cfg.swing_speed_scale_mps)
    return {
        'footstep_target': exp(-assigned_foot_distance_m/cfg.target_distance_scale_m),
        'footstep_root_progress': exp(-root_midpoint_distance_m/cfg.target_distance_scale_m),
        'footstep_velocity_tracking': exp(-((measured_forward_mps-commanded_forward_mps)/cfg.velocity_error_scale_mps)**2),
        'footstep_force_phase': (1-phase_swing_fraction)*swing_load/2 +
                                phase_swing_fraction*(1-swing_load)/2 + support_load/2,
        'footstep_velocity_phase': (phase_swing_fraction*normalized_swing_speed +
                                    (1-phase_swing_fraction)*(1-normalized_swing_speed)),
    }
