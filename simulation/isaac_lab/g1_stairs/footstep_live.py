"""Opt-in live observation and reset integration; no progression side effects."""
from __future__ import annotations

import torch

from .footstep_runtime import FootstepRuntime
from .footstep_isaac_adapter import target_observation
from .terrain_state import stair_cfg


def ensure_runtime(env):
    runtime = getattr(env, "_g1_footstep_runtime", None)
    count = int(stair_cfg(env).step_count)
    if runtime is None:
        runtime = FootstepRuntime(
            num_envs=env.num_envs, device=env.device,
            step_count=count,
            first_foot=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        )
        env._g1_footstep_runtime = runtime
    if runtime.num_envs != env.num_envs or runtime.step_count != count:
        raise RuntimeError("footstep runtime does not match environment geometry")
    return runtime


def reset_footsteps(env, env_ids):
    """Isaac Lab reset event callback; resets only designated environments."""
    ensure_runtime(env).reset(env_ids)


def planned_footstep_targets(env):
    """Isaac observation callback; read-only [N,14], no advancement."""
    return target_observation(
        env, ensure_runtime(env), position_scale_m=1.0,
        foot_separation_m=0.16,
    )
