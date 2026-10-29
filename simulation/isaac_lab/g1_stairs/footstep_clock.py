"""Per-environment gait clock and rolling footstep targets for flat tiles.

The clock decides WHEN each foot swings; the targets decide WHERE it lands.
Every swing gets a target, so no foot ever walks without one (the stair
approach had a single target 0.6 m ahead and the other foot just followed).

Targets follow a commanded reference pose that starts at the robot and then
moves only with the joystick command (speed and turn rate x time). Placing
them from the stance foot in the current heading (first version) let every
overshoot and every bit of drift carry into the next step: model 499 walked
~0.1 m/s too fast at all speeds and turned left 1-1.5 deg/s at zero turn
command. The reference re-syncs to the robot only if it falls behind by more
than the leash, so targets never run away.

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
REFERENCE_LEASH_M = 0.3           # re-sync the reference beyond this offset
REFERENCE_LEASH_YAW_RAD = 0.5     # or beyond this heading error (~29 deg)
# Uncommanded rotation: yaw-rate error averaged over each full gait cycle.
# The in-step pelvis twist cancels over a cycle; a steady drift does not.
# Rates only: no heading is stored, a past turn is never undone.
YAW_DRIFT_STD_RADPS = 0.035       # 2 deg/s cycle-average error costs 63%


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


def yaw_drift_cost(cycle_yaw_error):
    """[N] cost in [0, 1): 0 when the cycle-average turn rate matches the command."""
    return 1.0 - torch.exp(-(cycle_yaw_error / YAW_DRIFT_STD_RADPS) ** 2)


def follow_reference(ref_xy, ref_yaw, command, dt):
    """Integrate the reference pose with the body-frame command (vx, vy, wz).

    dt is a float or an [N] tensor of durations.
    """
    dt = torch.as_tensor(dt, dtype=ref_xy.dtype, device=ref_xy.device)
    mid = ref_yaw + 0.5 * command[:, 2] * dt
    c, s = torch.cos(mid), torch.sin(mid)
    vx, vy = command[:, 0], command[:, 1]
    step = torch.stack((c * vx - s * vy, s * vx + c * vy), dim=-1)
    xy = ref_xy + (dt[:, None] if dt.ndim else dt) * step
    return xy, ref_yaw + command[:, 2] * dt


def reference_targets(ref_xy, ref_yaw, swing_foot, command, period_s,
                      foot_separation_m):
    """[N,2,4] targets (x, y, z=0, yaw) and [N,2] feet for the next two steps.

    The swing foot touches down half a cycle after assignment; at touchdown
    it lies a quarter stride (speed x cycle / 4) ahead of the reference and
    half the foot separation to its own side. Slot 1 is the following step
    of the other foot, half a cycle later.
    """
    n = ref_xy.shape[0]
    if (ref_xy.shape != (n, 2) or ref_yaw.shape != (n,) or swing_foot.shape != (n,)
            or command.shape != (n, 3) or period_s.shape != (n,)):
        raise ValueError("invalid reference target shapes")
    if foot_separation_m <= 0:
        raise ValueError("foot_separation_m must be positive")
    side = torch.where(swing_foot == 0, 1.0, -1.0).to(ref_xy.dtype)
    half = 0.5 * period_s

    def landing(xy, yaw, lane):
        xy, yaw = follow_reference(xy, yaw, command, half)
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        left = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        foot = (xy + (command[:, 0] * period_s / 4)[:, None] * forward
                + (lane * foot_separation_m / 2)[:, None] * left)
        return xy, yaw, foot

    xy1, yaw1, first = landing(ref_xy, ref_yaw, side)
    _, yaw2, second = landing(xy1, yaw1, -side)
    zero = torch.zeros_like(ref_yaw)
    targets = torch.stack((
        torch.cat((first, zero[:, None], yaw1[:, None]), dim=-1),
        torch.cat((second, zero[:, None], yaw2[:, None]), dim=-1),
    ), dim=1)
    feet = torch.stack((swing_foot, 1 - swing_foot), dim=-1)
    return targets, feet


def sync_reference(state, root_xy, root_yaw, rows):
    """Re-sync rows that are standing, fresh, or beyond the leash."""
    err = torch.atan2(torch.sin(root_yaw - state.ref_yaw), torch.cos(root_yaw - state.ref_yaw))
    far = ((torch.linalg.vector_norm(root_xy - state.ref_xy, dim=-1) > REFERENCE_LEASH_M)
           | (err.abs() > REFERENCE_LEASH_YAW_RAD))
    state.ref_resyncs += (rows & state.ref_valid & far).long()
    sync = rows & (~state.ref_valid | far)
    state.ref_xy[sync] = root_xy[sync]
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
        self.ref_xy = torch.zeros((num_envs, 2), device=self.device)
        self.ref_yaw = torch.zeros(num_envs, device=self.device)
        self.ref_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.ref_resyncs = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.yaw_error_sum = torch.zeros(num_envs, device=self.device)
        self.yaw_error_time = torch.zeros(num_envs, device=self.device)
        self.cycle_yaw_error = torch.zeros(num_envs, device=self.device)

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
        self.yaw_error_sum[ids] = 0.0
        self.yaw_error_time[ids] = 0.0
        self.cycle_yaw_error[ids] = 0.0

    def advance(self, dt, rows, command, yaw_rate_error=None):
        """Move the phase and the reference pose of walking rows by dt.

        ``yaw_rate_error`` [N] (measured minus commanded, rad/s) is averaged
        over each cycle; the average is held for the next cycle. Rows not
        walking clear it.
        """
        step = torch.where(rows, dt / self.period_s, torch.zeros_like(self.phase))
        previous = self.phase
        self.phase = torch.remainder(self.phase + step, 1.0)
        if yaw_rate_error is not None:
            self.yaw_error_sum = torch.where(
                rows, self.yaw_error_sum + yaw_rate_error * dt, torch.zeros_like(self.yaw_error_sum))
            self.yaw_error_time = torch.where(
                rows, self.yaw_error_time + dt, torch.zeros_like(self.yaw_error_time))
            wrapped = rows & (self.phase < previous)
            average = self.yaw_error_sum / self.yaw_error_time.clamp_min(dt)
            self.cycle_yaw_error = torch.where(
                wrapped, average,
                torch.where(rows, self.cycle_yaw_error, torch.zeros_like(average)))
            self.yaw_error_sum = torch.where(wrapped, torch.zeros_like(average), self.yaw_error_sum)
            self.yaw_error_time = torch.where(wrapped, torch.zeros_like(average), self.yaw_error_time)
        xy, yaw = follow_reference(self.ref_xy, self.ref_yaw, command, dt)
        self.ref_xy = torch.where(rows[:, None], xy, self.ref_xy)
        self.ref_yaw = torch.where(rows, yaw, self.ref_yaw)
