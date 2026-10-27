"""Opt-in 16-D planned targets + gait clock observation diagnostic.

A single observation term evaluates the physical contact/progression update
before returning its new target and phase. This preserves observation coherency
without replacing the pinned Isaac Lab step implementation. No PPO integration.
"""
from __future__ import annotations

import torch

from .footstep_live_progress import ensure_coordinator, audit_after_step
from .footstep_isaac_adapter import target_observation, course_tensors


def planned_footsteps_with_phase(env) -> torch.Tensor:
    """[N,16] = next-two 14-D targets + assigned-foot 2-D gait clock.

    Observation manager can call this more than once per policy step. The
    coordinator's last_step gate prevents double progress/clock updates.
    Before the first policy step, only read initial state; do not infer contact.
    """
    coord = ensure_coordinator(env)
    step = int(env.common_step_counter)
    if step > 0 and coord.last_step != step:
        audit_after_step(env)
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
