"""Flat + stair mixing for the footstep task (pure tensor logic, no Isaac).

Flat envs (riser 0) keep the Unitree joystick command range and have no
footstep plan. Stair envs get a forward-only speed sampled per command.
Envs that leave the safe region of their tile end as a time-out, so flat
envs can use the full joystick range without walking into a neighbouring
stair tile or off the terrain border.
"""
from __future__ import annotations

import torch

# Unitree G1 29-DoF velocity task limit ranges (unitree_rl_lab 4960b84,
# velocity_env_cfg.py CommandsCfg.limit_ranges).
UNITREE_LIMITS = {
    "lin_vel_x": (-0.5, 1.0),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.2, 0.2),
}
# Course-local safe region. Robot starts at x=0; the tile spans x in
# [-1.0, 4.75] with a 1 m border, and the next row's stairs start at 6.75.
SAFE_X_M = (-1.5, 4.5)
SAFE_ABS_Y_M = 1.5
# Stair envs: the course is straight ahead. Leaving the 1.2 m wide tile,
# walking back past the start, or turning more than 60 degrees away is
# avoiding the stairs and counts as a failure, not a free time-out.
STAIR_ABS_Y_M = 0.6
STAIR_MIN_X_M = -0.5
STAIR_MAX_ABS_YAW_RAD = 1.0471975511965976


def _range(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, (int, float)) for v in value)
            or float(value[0]) > float(value[1])):
        raise ValueError(f"{name} must be [low, high] with low <= high")
    return float(value[0]), float(value[1])


def validate_terrain_split(cfg: dict) -> dict:
    """Validate the resolved ``command.terrain_split`` mapping."""
    if not isinstance(cfg, dict) or set(cfg) != {"flat", "stairs"}:
        raise ValueError("terrain_split needs exactly 'flat' and 'stairs'")
    flat = cfg["flat"]
    if not isinstance(flat, dict):
        raise ValueError("terrain_split.flat must be a mapping")
    out = {"flat": {}, "stairs": {}}
    for name, (low, high) in UNITREE_LIMITS.items():
        lo, hi = _range(flat.get(name), f"flat.{name}")
        if lo < low or hi > high:
            raise ValueError(f"flat.{name} must stay inside Unitree limits {low, high}")
        out["flat"][name] = (lo, hi)
    standing = float(flat.get("rel_standing_envs", 0.0))
    if not 0.0 <= standing <= 1.0:
        raise ValueError("flat.rel_standing_envs must be in [0, 1]")
    out["flat"]["rel_standing_envs"] = standing
    stairs = cfg["stairs"]
    if not isinstance(stairs, dict) or set(stairs) != {"lin_vel_x"}:
        raise ValueError("terrain_split.stairs takes only lin_vel_x")
    lo, hi = _range(stairs["lin_vel_x"], "stairs.lin_vel_x")
    if not 0.0 < lo <= hi <= 0.5:
        raise ValueError("stairs.lin_vel_x must satisfy 0 < low <= high <= 0.5")
    out["stairs"]["lin_vel_x"] = (lo, hi)
    return out


def apply_stair_forward_commands(vel_command_b, is_standing_env, env_ids,
                                 on_stairs, lin_vel_x):
    """After a normal resample: stair envs get forward-only speed, no standing."""
    ids = torch.as_tensor(env_ids, device=vel_command_b.device, dtype=torch.long).flatten()
    ids = ids[on_stairs[ids]]
    if ids.numel():
        speed = torch.empty(ids.numel(), device=vel_command_b.device)
        vel_command_b[ids, 0] = speed.uniform_(*lin_vel_x)
        vel_command_b[ids, 1:] = 0.0
        is_standing_env[ids] = False
    return ids


def outside_safe_region(root_xy_local, *, x_range=SAFE_X_M, abs_y=SAFE_ABS_Y_M):
    """[N] True when the root left the safe course region (course frame)."""
    x, y = root_xy_local[:, 0], root_xy_local[:, 1]
    return (x < x_range[0]) | (x > x_range[1]) | (y.abs() > abs_y)


def stair_escape(root_xy_local, root_yaw, on_stairs):
    """[N] Stair env left the corridor, walked back, or turned away (failure)."""
    x, y = root_xy_local[:, 0], root_xy_local[:, 1]
    away = ((y.abs() > STAIR_ABS_Y_M) | (x < STAIR_MIN_X_M)
            | (root_yaw.abs() > STAIR_MAX_ABS_YAW_RAD))
    return on_stairs & away


def course_exit_timeout(root_xy_local, root_yaw, on_stairs):
    """[N] Safe-region exit that is NOT a stair escape (time-out, no penalty)."""
    return (outside_safe_region(root_xy_local)
            & ~stair_escape(root_xy_local, root_yaw, on_stairs))
