"""Patch 82: domain randomisation for the stair task (replay probe first).

The stair task trains in a perfect world (StairEventsCfg sets friction, mass
and push events to None, observation noise off). This module switches on the
Unitree teacher's own ranges (unitree_rl_lab 4960b84 g1/29dof
velocity_env_cfg.EventCfg: friction 0.3-1.0, torso mass -1..+3 kg, push
+-0.5 m/s every 5 s, policy observation noise) and adds a stair position
belief error: the plan, the stair observation and the stair steering use
stairs shifted by an offset drawn per episode, while physics and the
evaluator keep the real stairs, and so do rewards and tread acceptance
(patch 83, real_stair_frame).

Config (top level): domain_randomization: {friction: bool, mass: bool,
push: bool, obs_noise: bool, stair_offset_m: [lo, hi]}. Offset +d means the
robot believes the stairs are d further away than they are.
"""

from __future__ import annotations

import torch

KEYS = {"friction", "mass", "push", "obs_noise", "stair_offset_m", "height_scan_noise_m"}
MAX_OFFSET_M = 0.05
MAX_SCAN_NOISE_M = 0.10


# Older experiment configs carry an all-false block with these names (inherited
# into every resolved config); they map onto the keys above.
LEGACY = {"rigid_body_material": "friction", "added_mass": "mass", "pushes": "push",
          "observation_noise": "obs_noise"}
LEGACY_IGNORED = {"enabled"}
LEGACY_UNSUPPORTED = {"action_latency"}


def validate_domain_cfg(cfg):
    """Normalised dict, or None when nothing is randomised. Unknown keys raise."""
    if not cfg:
        return None
    if not isinstance(cfg, dict):
        raise ValueError("domain_randomization must be a mapping")
    unknown = set(cfg) - KEYS - set(LEGACY) - LEGACY_IGNORED - LEGACY_UNSUPPORTED
    if unknown:
        raise ValueError(f"unknown domain_randomization keys {sorted(unknown)}")
    if any(bool(cfg.get(k)) for k in LEGACY_UNSUPPORTED):
        raise ValueError("domain_randomization.action_latency is not implemented")
    out = {k: bool(cfg.get(k, False)) for k in ("friction", "mass", "push", "obs_noise")}
    for old, new in LEGACY.items():
        out[new] = out[new] or bool(cfg.get(old, False))
    lo, hi = (float(v) for v in cfg.get("stair_offset_m", (0.0, 0.0)))
    if not -MAX_OFFSET_M <= lo <= hi <= MAX_OFFSET_M:
        raise ValueError(f"stair_offset_m must satisfy -{MAX_OFFSET_M} <= lo <= hi <= {MAX_OFFSET_M}")
    out["stair_offset_m"] = (lo, hi)
    # Patch 84: uniform noise per height-scan ray (m). The scan is part of the
    # policy group, so it is applied only with obs_noise (enable_corruption).
    scan = float(cfg.get("height_scan_noise_m", 0.0))
    if not 0.0 <= scan <= MAX_SCAN_NOISE_M:
        raise ValueError(f"height_scan_noise_m must be in [0, {MAX_SCAN_NOISE_M}]")
    if scan > 0.0 and not out["obs_noise"]:
        raise ValueError("height_scan_noise_m needs obs_noise (policy-group corruption)")
    out["height_scan_noise_m"] = scan
    if not any(out[k] for k in ("friction", "mass", "push", "obs_noise")) and lo == hi == 0.0:
        return None
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
    if dr.get("height_scan_noise_m", 0.0) > 0.0:
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg
        n = dr["height_scan_noise_m"]
        env_cfg.observations.policy.height_scan.noise = AdditiveUniformNoiseCfg(n_min=-n, n_max=n)
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


def real_stair_frame(course, targets, offset, plan_rows):
    """Patch 83: course and targets for judging feet against the real stairs.

    The plan, the observations, the walk-in stop and the gate use the
    believed stairs (``course`` from course_tensors). Rewards and tread
    acceptance must use the real ones, or a landing that follows the
    believed plan is paid even when it misses the real tread (replays of
    137100 with a 3 cm error: 7/70 and 6/70 clean). Stair-plan targets
    (``plan_rows`` [N]) move by the error so they sit on the real treads
    and keep the grid the tread checks rely on; rolling-clock targets are
    relative to the feet and stay. Returns the inputs unchanged when off.
    """
    if not torch.is_tensor(offset):
        if offset != 0.0:
            raise ValueError("stair belief offset must be a tensor or 0.0")
        return course, targets
    real = dict(course)
    real["first_riser_x"] = course["first_riser_x"] - offset
    shift = torch.where(plan_rows, offset, torch.zeros_like(offset))
    targets = targets.clone()
    targets[..., 0] -= shift[:, None]
    return real, targets
