"""Per-environment gait clock and rolling footstep targets for flat tiles.

The clock decides WHEN each foot swings; the targets decide WHERE it lands.
Every swing gets a target, placed from the stance foot at the moment the
swing foot is assigned, so no foot ever walks without one (the stair
approach had a single target 0.6 m ahead and the other foot just followed).

Cycle: double support -> left swing -> double support -> right swing.
Each episode draws its own cycle time, so the policy learns to follow any
clock in the range rather than one fixed rhythm; the stairs later use a
slow one. The cycle shortens only when the commanded speed would need a
stride longer than ``max_stride_m``.

Pure tensors; no Isaac imports.
"""
from __future__ import annotations

import math

import torch

DOUBLE_SUPPORT_FRACTION = 0.15   # of one cycle, each of the two
SINGLE_SUPPORT_FRACTION = 0.35   # of one cycle, each swing
STAND_SPEED_MPS = 0.1            # below this and STAND_YAW_RATE: stand
STAND_YAW_RATE_RADPS = 0.1


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


def rolling_targets(stance_xy, swing_foot, yaw, step_length_m,
                    lateral_offset_m, foot_separation_m):
    """[N,2,4] targets (x, y, z=0, yaw) and [N,2] feet for the next two steps.

    Slot 0: the swing foot, one step ahead of the stance foot, in its own
    lane. Slot 1: the step after that, for the stance foot. Directions use
    the robot heading at assignment; lateral_offset_m shifts each step
    sideways (sideways command).
    """
    n = stance_xy.shape[0]
    if (stance_xy.shape != (n, 2) or swing_foot.shape != (n,)
            or yaw.shape != (n,) or step_length_m.shape != (n,)
            or lateral_offset_m.shape != (n,)):
        raise ValueError("invalid rolling target shapes")
    if foot_separation_m <= 0:
        raise ValueError("foot_separation_m must be positive")
    # Left (0) lies at +y of the right foot in the heading frame.
    side = torch.where(swing_foot == 0, 1.0, -1.0).to(stance_xy.dtype)
    forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    left = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)

    def step(origin, lane_sign):
        return (origin + step_length_m[:, None] * forward
                + (lane_sign * foot_separation_m + lateral_offset_m)[:, None] * left)

    first = step(stance_xy, side)
    second = step(first, -side)
    zero = torch.zeros_like(yaw)
    targets = torch.stack((
        torch.cat((first, zero[:, None], yaw[:, None]), dim=-1),
        torch.cat((second, zero[:, None], yaw[:, None]), dim=-1),
    ), dim=1)
    feet = torch.stack((swing_foot, 1 - swing_foot), dim=-1)
    return targets, feet


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

    def advance(self, dt, rows):
        """Move the phase of walking rows by dt / cycle time."""
        step = torch.where(rows, dt / self.period_s, torch.zeros_like(self.phase))
        self.phase = torch.remainder(self.phase + step, 1.0)
