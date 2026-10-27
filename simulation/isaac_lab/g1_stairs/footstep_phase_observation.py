"""Opt-in 16-D planned targets + gait clock observation diagnostic.

Read-only observation: progression is managed after physics, before reward.
No PPO integration.
"""
from __future__ import annotations

import torch

from .footstep_live_progress import ensure_coordinator
from .footstep_isaac_adapter import target_observation, course_tensors


def planned_footsteps_with_phase(env) -> torch.Tensor:
    """[N,16] = next-two 14-D targets + assigned-foot 2-D gait clock.

    Observation evaluation never mutates planner or phase state.
    """
    coord = ensure_coordinator(env)
    targets = target_observation(
        env, coord.runtime, position_scale_m=1.0, foot_separation_m=0.16,
    )
    course = course_tensors(env, foot_separation_m=0.16)
    _poses, feet, valid = coord.runtime.targets(**course)
    clock, _swing, _stance = coord.clock.features(
        target_foot=feet[:, 0], valid=valid[:, 0]
    )
    result = torch.cat((targets, clock), dim=-1)
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
    if hasattr(env, "_g1_footstep_net_progress"):
        env._g1_footstep_net_progress[ids] = 0.0
