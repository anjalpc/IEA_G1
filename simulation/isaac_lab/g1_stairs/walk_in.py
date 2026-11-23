"""Patch 76: walk in, stop, then climb on stair rows.

Stair rows start ``start_distance_m`` before the first riser (behind the usual
start; the tile has ``approach_length - start_clearance`` of level ground
behind it). The frozen Unitree teacher walks in alone (residual gate 0; the
forward command ramps up), the command is released so the robot stops near a
drawn stop distance, it settles, and only then does the climb begin: the
footstep plan and gait clock are reset exactly as at an episode start
(reset_footsteps_with_phase) and the residual fades in (balance_residual gate
"after_stop"). So the climb starts from the state a real stop leaves (pose,
residual sway, teacher action history) instead of the reset pose.
Flat rows are untouched. Config: footstep_clock.walk_in.
"""
from __future__ import annotations

import torch

WALK, STOP, CLIMB = 0, 1, 2
KEYS = {"enabled", "start_distance_m", "stop_distance_m", "walk_speed_mps",
        "ramp_s", "stop_ramp_s", "settle_s"}


def validate_walk_in_cfg(cfg, *, start_clearance_m=1.0, approach_length_m=2.0) -> dict | None:
    """Normalised walk-in config, or None when absent / disabled."""
    if cfg in (None, {}):
        return None
    if not isinstance(cfg, dict):
        raise ValueError("footstep_clock.walk_in must be a mapping")
    if not bool(cfg.get("enabled", False)):
        return None
    if set(cfg) != KEYS:
        raise ValueError(f"footstep_clock.walk_in needs exactly {sorted(KEYS)}")
    start = float(cfg["start_distance_m"])
    stop_lo, stop_hi = (float(v) for v in cfg["stop_distance_m"])
    settle_lo, settle_hi = (float(v) for v in cfg["settle_s"])
    speed, ramp, stop_ramp = (float(cfg[k]) for k in ("walk_speed_mps", "ramp_s", "stop_ramp_s"))
    behind = approach_length_m - start_clearance_m
    # 0.3 m kept clear of the tile's back edge: the tile behind belongs to
    # another course and may end in a raised platform.
    if not start_clearance_m <= start <= start_clearance_m + behind - 0.3 + 1e-9:
        raise ValueError(f"walk_in.start_distance_m must be in [{start_clearance_m}, "
                         f"{start_clearance_m + behind - 0.3}] (level ground behind the start)")
    if not 0.3 <= stop_lo <= stop_hi <= start - 0.3:
        raise ValueError("walk_in.stop_distance_m must satisfy 0.3 <= min <= max <= start - 0.3")
    if not 0.1 <= speed <= 0.5:
        raise ValueError("walk_in.walk_speed_mps must be in [0.1, 0.5]")
    if not 0.2 <= ramp <= 3.0 or not 0.2 <= stop_ramp <= 2.0:
        raise ValueError("walk_in.ramp_s must be in [0.2, 3] and stop_ramp_s in [0.2, 2]")
    if not 0.0 <= settle_lo <= settle_hi <= 3.0:
        raise ValueError("walk_in.settle_s must satisfy 0 <= min <= max <= 3")
    return {"enabled": True, "start_distance_m": start, "stop_distance_m": (stop_lo, stop_hi),
            "walk_speed_mps": speed, "ramp_s": ramp, "stop_ramp_s": stop_ramp,
            "settle_s": (settle_lo, settle_hi)}


def stop_trigger_x(first_riser_x, stop_distance, speed, stop_ramp_s):
    """[N] course x at which to release the command so the pelvis comes to rest
    near ``first_riser_x - stop_distance``: a linear ramp-down covers about
    speed * stop_ramp_s / 2."""
    return first_riser_x - stop_distance - 0.5 * speed * stop_ramp_s


def advance_phase(phase, t, t_stop, settle, x, trigger_x, stairs, stop_ramp_s):
    """One step of the walk-in phase machine. Returns (phase, t_stop, climb_now)."""
    reach = stairs & (phase == WALK) & (x >= trigger_x)
    t_stop = torch.where(reach, t, t_stop)
    phase = torch.where(reach, torch.full_like(phase, STOP), phase)
    climb_now = stairs & (phase == STOP) & (t - t_stop >= stop_ramp_s + settle)
    phase = torch.where(climb_now, torch.full_like(phase, CLIMB), phase)
    return phase, t_stop, climb_now


def walk_in_forward_command(phase, t, t_stop, speed, ramp_s, stop_ramp_s):
    """[N] forward command for WALK / STOP rows (CLIMB rows: NaN, not set here)."""
    walk = speed * (t / ramp_s).clamp(0.0, 1.0)
    stop = speed * (1.0 - (t - t_stop) / stop_ramp_s).clamp(0.0, 1.0)
    out = torch.where(phase == WALK, walk, stop)
    return torch.where(phase == CLIMB, torch.full_like(out, float("nan")), out)


def _state(env):
    st = getattr(env, "_g1_walk_in", None)
    if st is None:
        n, dev = env.num_envs, env.device
        st = {"phase": torch.full((n,), CLIMB, dtype=torch.long, device=dev),
              "t_stop": torch.zeros(n, device=dev), "settle": torch.zeros(n, device=dev),
              "stop_distance": torch.zeros(n, device=dev)}
        env._g1_walk_in = st
    return st


def reset_walk_in(env, env_ids):
    """Isaac reset event (after reset_base and the footstep reset): stair rows
    move back to start_distance_m and start walking in; flat rows climb-phase
    (no effect: the gate needs a riser)."""
    cfg = (getattr(env.cfg, "footstep_clock", None) or {}).get("walk_in")
    if not cfg:
        return
    from .terrain_state import riser_height_per_env
    from .footstep_isaac_adapter import course_tensors
    from .footstep_tensor import NOMINAL_FOOT_SEPARATION_M
    st = _state(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else \
        torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    stairs = riser_height_per_env(env)[ids] > 0.0
    lo, hi = cfg["stop_distance_m"]
    slo, shi = cfg["settle_s"]
    st["phase"][ids] = torch.where(stairs, WALK, CLIMB)
    st["t_stop"][ids] = 0.0
    st["stop_distance"][ids] = torch.empty(len(ids), device=env.device).uniform_(lo, hi)
    st["settle"][ids] = torch.empty(len(ids), device=env.device).uniform_(slo, shi)
    climbing = getattr(env, "_g1_walk_in_climbing", None)
    if climbing is not None:
        climbing[ids] = False
    move = ids[stairs]
    if move.numel():
        robot = env.scene["robot"]
        first = course_tensors(env, foot_separation_m=NOMINAL_FOOT_SEPARATION_M)["first_riser_x"]
        back = cfg["start_distance_m"] - first[move]       # first_riser_x = start_clearance
        pose = robot.data.default_root_state[move, :7].clone()
        pose[:, :3] += env.scene.env_origins[move]
        pose[:, 0] -= back
        robot.write_root_pose_to_sim(pose, env_ids=move)


def walk_in_step(env, resets):
    """Per step, after the stair command is set: WALK / STOP rows get the walk
    or release command; a row whose settle time is over starts the climb
    (footstep plan and clock reset). Sets env._g1_walk_in_climbing."""
    cfg = (getattr(env.cfg, "footstep_clock", None) or {}).get("walk_in")
    if not cfg:
        return
    from .terrain_state import riser_height_per_env
    from .footstep_isaac_adapter import course_root_pose, course_tensors
    from .footstep_tensor import NOMINAL_FOOT_SEPARATION_M
    from .footstep_phase_observation import reset_footsteps_with_phase
    st = _state(env)
    stairs = riser_height_per_env(env) > 0.0
    t = env.episode_length_buf.float() * float(env.step_dt)
    xyz, _ = course_root_pose(env)
    first = course_tensors(env, foot_separation_m=NOMINAL_FOOT_SEPARATION_M)["first_riser_x"]
    trigger = stop_trigger_x(first, st["stop_distance"], cfg["walk_speed_mps"], cfg["stop_ramp_s"])
    live = stairs & ~resets
    st["phase"], st["t_stop"], climb_now = advance_phase(
        st["phase"], t, st["t_stop"], st["settle"], xyz[:, 0], trigger, live, cfg["stop_ramp_s"])
    ids = climb_now.nonzero().flatten()
    if ids.numel():
        reset_footsteps_with_phase(env, ids)
    before = stairs & (st["phase"] != CLIMB)
    if bool(before.any()):
        command = env.command_manager.get_command("base_velocity")
        fwd = walk_in_forward_command(st["phase"], t, st["t_stop"], cfg["walk_speed_mps"],
                                      cfg["ramp_s"], cfg["stop_ramp_s"])
        command[before, 0] = fwd[before]
        command[before, 1:] = 0.0
        since = getattr(env, "_g1_footstep_since_advance", None)
        if since is not None:
            since[before] = 0.0                     # no plan-stall clock before the climb
    env._g1_walk_in_climbing = stairs & (st["phase"] == CLIMB)
