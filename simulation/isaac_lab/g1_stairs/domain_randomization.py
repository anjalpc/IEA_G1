"""Patch 82: domain randomisation for the stair task (replay probe first).

The stair task trains in a perfect world (StairEventsCfg sets friction, mass
and push events to None, observation noise off). This module switches on the
Unitree teacher's own ranges (unitree_rl_lab 4960b84 g1/29dof
velocity_env_cfg.EventCfg: friction 0.3-1.0, torso mass -1..+3 kg, push
+-0.5 m/s every 5 s, policy observation noise) and adds a stair position
belief error: the plan, the stair observation and the stair steering use
stairs shifted by an offset drawn per episode, while physics and the
evaluator keep the real stairs.

Config (top level): domain_randomization: {friction: bool, mass: bool,
push: bool, obs_noise: bool, stair_offset_m: [lo, hi]}. Offset +d means the
robot believes the stairs are d further away than they are.
"""

from __future__ import annotations

import torch

KEYS = {"friction", "mass", "push", "obs_noise", "stair_offset_m"}
MAX_OFFSET_M = 0.05


def validate_domain_cfg(cfg):
    """Normalised dict or None (off). Unknown keys and bad ranges raise."""
    if not cfg:
        return None
    if not isinstance(cfg, dict) or set(cfg) - KEYS:
        raise ValueError(f"domain_randomization keys must be from {sorted(KEYS)}")
    out = {k: bool(cfg.get(k, False)) for k in ("friction", "mass", "push", "obs_noise")}
    lo, hi = (float(v) for v in cfg.get("stair_offset_m", (0.0, 0.0)))
    if not -MAX_OFFSET_M <= lo <= hi <= MAX_OFFSET_M:
        raise ValueError(f"stair_offset_m must satisfy -{MAX_OFFSET_M} <= lo <= hi <= {MAX_OFFSET_M}")
    out["stair_offset_m"] = (lo, hi)
    return out


def apply_domain_randomization(env_cfg, dr) -> None:
    """Switch the Unitree teacher's randomisation terms back on (as upstream)."""
    if dr is None:
        return
    from .env_cfg import _UPSTREAM

    upstream = _UPSTREAM.EventCfg()
    if dr["friction"]:
        env_cfg.events.physics_material = upstream.physics_material
    if dr["mass"]:
        env_cfg.events.add_base_mass = upstream.add_base_mass
    if dr["push"]:
        env_cfg.events.push_robot = upstream.push_robot
    if dr["obs_noise"]:
        env_cfg.observations.policy.enable_corruption = True
    env_cfg.domain_randomization = dr


def stair_belief_offset(env):
    """[N] believed minus real first-riser x, or 0.0 when off."""
    offset = getattr(env, "_g1_stair_belief_offset", None)
    return 0.0 if offset is None else offset


def reset_stair_belief(env, env_ids) -> None:
    """Reset event: draw the stair position error per episode."""
    dr = getattr(env.cfg, "domain_randomization", None)
    if not dr:
        return
    lo, hi = dr["stair_offset_m"]
    if lo == 0.0 and hi == 0.0:
        return
    if getattr(env, "_g1_stair_belief_offset", None) is None:
        env._g1_stair_belief_offset = torch.zeros(env.num_envs, device=env.device)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    env._g1_stair_belief_offset[ids] = torch.empty(len(ids), device=env.device).uniform_(lo, hi)
