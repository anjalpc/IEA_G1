"""Per-environment gait clock and rolling footstep targets for flat tiles.

The clock decides WHEN each foot swings; the targets decide WHERE it lands.
Every swing gets a target, so no foot ever walks without one (the stair
approach had a single target 0.6 m ahead and the other foot just followed).

Each target is placed one step ahead of the stance foot, along a commanded
heading: it starts at the robot's heading and then turns only with the
joystick turn command. Placing targets along the robot's own heading let
every bit of drift carry into the next step (model 499 turned left
1-1.5 deg/s at zero turn command). Anchoring the target POSITION to a
commanded path as well (run footstep_flat_clock_20261028_213244) made the
robot run 12-25 cm ahead of its targets, miss most of them and stop
following the clock, so positions stay anchored to the stance foot. The
heading re-syncs to the robot only beyond the leash.

Pure tensors; no Isaac imports.
"""
from __future__ import annotations

import math

import torch

DOUBLE_SUPPORT_FRACTION = 0.15   # of one cycle, each of the two
SINGLE_SUPPORT_FRACTION = 0.35   # of one cycle, each swing
STAND_SPEED_MPS = 0.1            # below this and STAND_YAW_RATE: stand
STAND_YAW_RATE_RADPS = 0.1
REFERENCE_LEASH_YAW_RAD = 0.5     # re-sync the heading beyond this error (~29 deg)


def validate_clock_cfg(cfg) -> dict | None:
    """Return the normalised ``footstep_clock`` config, or None if disabled."""
    if cfg in (None, {}):
        return None
    if not isinstance(cfg, dict):
        raise ValueError("footstep_clock must be a mapping")
    if not bool(cfg.get("enabled", False)):
        return None
    unknown = set(cfg) - {"enabled", "period_s", "max_stride_m"}
    if unknown:
        raise ValueError(f"unknown footstep_clock keys: {sorted(unknown)}")
    period = cfg.get("period_s")
    if (not isinstance(period, (list, tuple)) or len(period) != 2
            or not all(isinstance(v, (int, float)) for v in period)):
        raise ValueError("footstep_clock.period_s must be [low, high]")
    low, high = float(period[0]), float(period[1])
    if not 0.4 <= low <= high <= 2.0:
        raise ValueError("footstep_clock.period_s must satisfy 0.4 <= low <= high <= 2.0")
    stride = float(cfg.get("max_stride_m", 0.0))
    if not 0.2 <= stride <= 0.8:
        raise ValueError("footstep_clock.max_stride_m must be in [0.2, 0.8]")
    return {"enabled": True, "period_s": (low, high), "max_stride_m": stride}


def effective_period(base_period_s, forward_speed, max_stride_m):
    """[N] cycle time: the episode's own, shortened only if the stride
    (speed x cycle time) would exceed max_stride_m."""
    if base_period_s.shape != forward_speed.shape:
        raise ValueError("period and speed shapes differ")
    speed = forward_speed.abs().clamp_min(1e-6)
    return torch.minimum(base_period_s, max_stride_m / speed)


def moving(command):
    """[N] True when the command asks for walking rather than standing."""
    return ((command[:, 0].abs() >= STAND_SPEED_MPS)
            | (command[:, 1].abs() >= STAND_SPEED_MPS)
            | (command[:, 2].abs() >= STAND_YAW_RATE_RADPS))


def clock_schedule(phase):
    """Return ([N,2] scheduled swing [left,right], [N] assigned foot).

    The assigned foot is the one swinging now, or next in double support:
    left from the start of the cycle until its swing ends, then right.
    """
    if bool(((phase < 0) | (phase >= 1)).any()):
        raise ValueError("phase must be in [0, 1)")
    ds, ss = DOUBLE_SUPPORT_FRACTION, SINGLE_SUPPORT_FRACTION
    left = (phase >= ds) & (phase < ds + ss)
    right = (phase >= 2 * ds + ss) & (phase < 1.0)
    assigned = torch.where(phase < ds + ss, 0, 1).to(torch.long)
    return torch.stack((left, right), dim=-1), assigned


def clock_features(phase, active):
    """[N,2] sin/cos of the cycle; zero while standing."""
    angle = 2 * math.pi * phase
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    return torch.where(active[:, None], clock, torch.zeros_like(clock))


def heading_targets(stance_xy, swing_foot, ref_yaw, command, period_s,
                    foot_separation_m):
    """[N,2,4] targets (x, y, z=0, yaw) and [N,2] feet for the next two steps.

    Slot 0: the swing foot, one step (speed x cycle / 2) ahead of the stance
    foot in its own lane, along the commanded heading at its touchdown (half
    a cycle after assignment). Slot 1: the following step of the other foot.
    A sideways command shifts each step by vy x cycle / 2.
    """
    n = stance_xy.shape[0]
    if (stance_xy.shape != (n, 2) or swing_foot.shape != (n,) or ref_yaw.shape != (n,)
            or command.shape != (n, 3) or period_s.shape != (n,)):
        raise ValueError("invalid heading target shapes")
    if foot_separation_m <= 0:
        raise ValueError("foot_separation_m must be positive")
    side = torch.where(swing_foot == 0, 1.0, -1.0).to(stance_xy.dtype)
    half = 0.5 * period_s
    step_len = command[:, 0] * half
    shift = command[:, 1] * half

    def step(origin, yaw, lane):
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        left = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        return (origin + step_len[:, None] * forward
                + (lane * foot_separation_m + shift)[:, None] * left)

    yaw1 = ref_yaw + command[:, 2] * half
    yaw2 = yaw1 + command[:, 2] * half
    first = step(stance_xy, yaw1, side)
    second = step(first, yaw2, -side)
    zero = torch.zeros_like(ref_yaw)
    targets = torch.stack((
        torch.cat((first, zero[:, None], yaw1[:, None]), dim=-1),
        torch.cat((second, zero[:, None], yaw2[:, None]), dim=-1),
    ), dim=1)
    feet = torch.stack((swing_foot, 1 - swing_foot), dim=-1)
    return targets, feet


def sync_reference(state, root_yaw, rows):
    """Re-sync the commanded heading of rows that are fresh or beyond the leash."""
    err = torch.atan2(torch.sin(root_yaw - state.ref_yaw), torch.cos(root_yaw - state.ref_yaw))
    far = err.abs() > REFERENCE_LEASH_YAW_RAD
    state.ref_resyncs += (rows & state.ref_valid & far).long()
    sync = rows & (~state.ref_valid | far)
    state.ref_yaw[sync] = root_yaw[sync]
    state.ref_valid |= rows
    return sync


class RollingClockState:
    """Per-env clock phase, cycle time and the current rolling targets."""

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.phase = torch.zeros(num_envs, device=self.device)
        self.base_period_s = torch.ones(num_envs, device=self.device)
        self.period_s = torch.ones(num_envs, device=self.device)
        self.assigned = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.window = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.acquired = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.targets = torch.zeros((num_envs, 2, 4), device=self.device)
        self.feet = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.ref_yaw = torch.zeros(num_envs, device=self.device)
        self.ref_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.ref_resyncs = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids, period_range_s):
        """New cycle time per episode; random lead foot via half-cycle offset."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None \
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        low, high = period_range_s
        count = ids.numel()
        self.base_period_s[ids] = torch.empty(count, device=self.device).uniform_(low, high)
        self.period_s[ids] = self.base_period_s[ids]
        self.phase[ids] = 0.5 * torch.randint(0, 2, (count,), device=self.device).float()
        self.assigned[ids] = -1
        self.window[ids] = 0
        self.acquired[ids] = False
        self.targets[ids] = 0.0
        self.active[ids] = False
        self.ref_valid[ids] = False
        self.ref_resyncs[ids] = 0

    def advance(self, dt, rows, command):
        """Move the phase and the commanded heading of walking rows by dt."""
        step = torch.where(rows, dt / self.period_s, torch.zeros_like(self.phase))
        self.phase = torch.remainder(self.phase + step, 1.0)
        self.ref_yaw = torch.where(rows, self.ref_yaw + command[:, 2] * dt, self.ref_yaw)
