"""Opt-in 16-D planned targets + gait clock observation diagnostic.

Read-only observation: progression is managed after physics, before reward.
No PPO integration.
"""
from __future__ import annotations

import torch

from .footstep_clock import clock_features
from .footstep_live_progress import ensure_coordinator, ensure_rolling_clock
from .footstep_isaac_adapter import (
    course_root_pose, course_tensors, target_observation,
)
from .footstep_tensor import NOMINAL_FOOT_SEPARATION_M


def planned_footsteps_with_phase(env) -> torch.Tensor:
    """[N,16] = next-two 14-D targets + 2 gait-clock inputs (sin, cos).

    Without ``footstep_clock`` the clock inputs stay zero (the plan says
    WHERE, the policy chooses WHEN). With it, flat tiles see the rolling
    clock and their rolling targets; stair rows are unchanged. Never
    mutates planner state.
    """
    coord = ensure_coordinator(env)
    flat = bool(getattr(env.cfg, "flat_ground_footsteps", False))
    state = ensure_rolling_clock(env)
    rows = getattr(env, "_g1_rolling_rows", None)
    if state is None or rows is None:
        targets = target_observation(
            env, coord.runtime, position_scale_m=1.0,
            foot_separation_m=NOMINAL_FOOT_SEPARATION_M, flat_ground=flat,
        )
        timing = torch.zeros((env.num_envs, 2), dtype=targets.dtype, device=targets.device)
    else:
        course = course_tensors(
            env, foot_separation_m=NOMINAL_FOOT_SEPARATION_M, flat_ground=flat,
        )
        plan, feet, valid = coord.runtime.targets(**course, flat_ground=flat)
        plan, feet, valid = plan.clone(), feet.clone(), valid.clone()
        plan[rows] = state.targets[rows]
        feet[rows] = state.feet[rows]
        valid[rows] = state.active[rows, None]
        xyz, yaw = course_root_pose(env)
        targets = coord.runtime.observation(
            targets=plan, feet=feet, valid=valid, root_xyz=xyz,
            root_yaw=yaw, position_scale=1.0,
        )
        timing = clock_features(state.phase, state.active & rows).to(targets.dtype)
    result = torch.cat((targets, timing), dim=-1)
    if result.shape != (env.num_envs, 16) or not bool(torch.isfinite(result).all()):
        raise RuntimeError('invalid planned footsteps plus gait phase observation')
    return result


def random_first_foot(count: int, device) -> torch.Tensor:
    """0=left / 1=right with equal probability, one per resetting env."""
    return torch.randint(0, 2, (count,), device=device, dtype=torch.long)


def reset_footsteps_with_phase(env, env_ids):
    """Isaac reset event; clears target state and clock for selected rows.

    The plan's lead foot is drawn 50/50 per episode. A fixed left lead
    taught a left/right asymmetry (short left steps, leftward turn) that
    leaked into flat walking through the shared network.
    """
    count = env.num_envs if env_ids is None else len(env_ids)
    ensure_coordinator(env).reset(
        env_ids, first_foot=random_first_foot(count, env.device)
    )
    ids = slice(None) if env_ids is None else env_ids
    state = ensure_rolling_clock(env)
    if state is not None:
        state.reset(env_ids, env.cfg.footstep_clock["period_s"])

    if hasattr(env, "_g1_footstep_best_distance"):
        env._g1_footstep_best_distance[ids] = float("inf")
    if hasattr(env, "_g1_footstep_progress_target_index"):
        env._g1_footstep_progress_target_index[ids] = -1
    if hasattr(env, "_g1_footstep_pace_target_index"):
        env._g1_footstep_pace_target_index[ids] = -1

    if hasattr(env, "_g1_footstep_peak_clearance"):
        env._g1_footstep_peak_clearance[ids] = 0.0
    if hasattr(env, "_g1_footstep_clearance_target_index"):
        env._g1_footstep_clearance_target_index[ids] = -1
    if hasattr(env, "_g1_footstep_clearance_progress"):
        env._g1_footstep_clearance_progress[ids] = 0.0

    if hasattr(env, "_g1_footstep_net_progress"):
        env._g1_footstep_net_progress[ids] = 0.0
    if hasattr(env, "_g1_footstep_since_advance"):
        env._g1_footstep_since_advance[ids] = 0.0
