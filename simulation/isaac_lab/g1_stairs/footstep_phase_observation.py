"""Opt-in 16-D planned targets + gait clock observation diagnostic.

Read-only observation: progression is managed after physics, before reward.
No PPO integration.
"""
from __future__ import annotations

import torch

from .footstep_live_progress import ensure_coordinator
from .footstep_isaac_adapter import target_observation
from .footstep_tensor import NOMINAL_FOOT_SEPARATION_M


def planned_footsteps_with_phase(env) -> torch.Tensor:
    """[N,16] = next-two 14-D targets + 2 timing inputs held at zero.

    Timing is chosen by the policy: the plan says WHERE, not WHEN. The two
    former gait-clock channels stay in the layout (688-D actor, prior
    transfer unchanged) but carry no signal. Never mutates planner state.
    """
    coord = ensure_coordinator(env)
    flat = bool(getattr(env.cfg, "flat_ground_footsteps", False))
    targets = target_observation(
        env, coord.runtime, position_scale_m=1.0,
        foot_separation_m=NOMINAL_FOOT_SEPARATION_M, flat_ground=flat,
    )
    timing = torch.zeros((env.num_envs, 2), dtype=targets.dtype, device=targets.device)
    result = torch.cat((targets, timing), dim=-1)
    if result.shape != (env.num_envs, 16) or not bool(torch.isfinite(result).all()):
        raise RuntimeError('invalid planned footsteps plus gait phase observation')
    return result


def reset_footsteps_with_phase(env, env_ids):
    """Isaac reset event; clears target state and clock for selected rows."""
    ensure_coordinator(env).reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids

    if hasattr(env, "_g1_footstep_best_distance"):
        env._g1_footstep_best_distance[ids] = float("inf")
    if hasattr(env, "_g1_footstep_progress_target_index"):
        env._g1_footstep_progress_target_index[ids] = -1

    if hasattr(env, "_g1_footstep_peak_clearance"):
        env._g1_footstep_peak_clearance[ids] = 0.0
    if hasattr(env, "_g1_footstep_clearance_target_index"):
        env._g1_footstep_clearance_target_index[ids] = -1
    if hasattr(env, "_g1_footstep_clearance_progress"):
        env._g1_footstep_clearance_progress[ids] = 0.0

    if hasattr(env, "_g1_footstep_net_progress"):
        env._g1_footstep_net_progress[ids] = 0.0
