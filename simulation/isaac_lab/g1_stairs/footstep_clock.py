"""Per-environment gait clock and rolling footstep targets for flat tiles.

The clock decides WHEN each foot swings; the targets decide WHERE it lands.
Every swing gets a target, so no foot ever walks without one (the stair
approach had a single target 0.6 m ahead and the other foot just followed).

Body-relative, like a joystick: each target is placed one step ahead of
the stance foot, along the robot's own heading plus the commanded turn
until touchdown. Nothing remembers a start position or heading. Holding
the commanded direction is the policy's job (uncommanded-rotation cost),
not the planner's.

Cycle time follows the commanded speed (slow and long at low speed,
quicker at high speed, with a small per-episode jitter); step length is
speed x half a cycle, capped by ``max_stride_m`` (above the cap only the
rhythm gets quicker).

Stair tiles: the approach is walked like a flat tile (rolling targets,
speed-based cycle), except that the steps count down to the plan's fixed
ground placement before the first riser: the distance left is split into
equal steps no longer than a normal step (the last may be 25% longer), so
the adjustment to the stairs is spread over the run-up instead of one odd
step at the end. When one step is left, the tile hands over to the stair
plan, whose first foot is the foot about to swing; from then on the cycle is fixed
(``period_stairs_s``) and one tread is one step. The plan decides which foot
swings: when a swing window opens for a foot the plan is not waiting for
(after a miss), the clock is shifted half a cycle. A stair target only
counts during its foot's own swing window and the double support after it,
so extra steps outside the rhythm earn nothing (as on flat tiles, where a
target lives for one window). Once the plan is complete the tile walks on
like a flat tile (rolling targets at the platform height, the joystick
command), so one policy climbs and then walks.

A swing ends when its foot reaches its target: the rest of that window
counts as stance, so the clock never pays a foot to lift again after a
good placement.

Pure tensors; no Isaac imports.
"""
from __future__ import annotations

import math

import torch

DOUBLE_SUPPORT_FRACTION = 0.15   # of one cycle, each of the two
SINGLE_SUPPORT_FRACTION = 0.35   # of one cycle, each swing
STAND_SPEED_MPS = 0.1            # below this and STAND_YAW_RATE: stand
STAND_YAW_RATE_RADPS = 0.1
PERIOD_SLOW_SPEED_MPS = 0.1      # period_slow_s applies at or below this speed
PERIOD_FAST_SPEED_MPS = 0.6      # period_fast_s applies at or above this speed
# Uncommanded rotation: yaw-rate error averaged over each full gait cycle.
# The in-step pelvis twist cancels over a cycle; a steady drift does not.
# Rates only: no heading is stored, a past turn is never undone.
YAW_DRIFT_STD_RADPS = 0.035       # 2 deg/s cycle-average error costs 63%
APPROACH_STEP_TOLERANCE = 0.25    # last approach step may be this much longer than normal


def validate_clock_cfg(cfg) -> dict | None:
    """Return the normalised ``footstep_clock`` config, or None if disabled."""
    if cfg in (None, {}):
        return None
    if not isinstance(cfg, dict):
        raise ValueError("footstep_clock must be a mapping")
    if not bool(cfg.get("enabled", False)):
        return None
    keys = {"enabled", "period_slow_s", "period_fast_s", "period_stairs_s",
            "period_jitter", "max_stride_m"}
    unknown = set(cfg) - keys
    if unknown:
        raise ValueError(f"unknown footstep_clock keys: {sorted(unknown)}")
    if set(cfg) != keys:
        raise ValueError(f"footstep_clock needs keys {sorted(keys)}")
    slow, fast = float(cfg["period_slow_s"]), float(cfg["period_fast_s"])
    if not 0.4 <= fast <= slow <= 2.0:
        raise ValueError("footstep_clock needs 0.4 <= period_fast_s <= period_slow_s <= 2.0")
    stairs = float(cfg["period_stairs_s"])
    if not 0.6 <= stairs <= 2.0:
        raise ValueError("footstep_clock.period_stairs_s must be in [0.6, 2.0]")
    jitter = float(cfg["period_jitter"])
    if not 0.0 <= jitter <= 0.3:
        raise ValueError("footstep_clock.period_jitter must be in [0, 0.3]")
    stride = float(cfg["max_stride_m"])
    if not 0.2 <= stride <= 0.8:
        raise ValueError("footstep_clock.max_stride_m must be in [0.2, 0.8]")
    return {"enabled": True, "period_slow_s": slow, "period_fast_s": fast,
            "period_stairs_s": stairs, "period_jitter": jitter, "max_stride_m": stride}


def speed_period(forward_speed, scale, cfg):
    """[N] cycle time for the commanded forward speed: linear from
    period_slow_s (<= 0.1 m/s) to period_fast_s (>= 0.6 m/s), times the
    episode's jitter scale, then shortened if the stride (speed x cycle)
    would exceed max_stride_m."""
    if forward_speed.shape != scale.shape:
        raise ValueError("speed and scale shapes differ")
    speed = forward_speed.abs()
    blend = ((speed - PERIOD_SLOW_SPEED_MPS)
             / (PERIOD_FAST_SPEED_MPS - PERIOD_SLOW_SPEED_MPS)).clamp(0.0, 1.0)
    period = scale * (cfg["period_slow_s"] + blend * (cfg["period_fast_s"] - cfg["period_slow_s"]))
    return torch.minimum(period, cfg["max_stride_m"] / speed.clamp_min(1e-6))


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


def follow_plan_foot(phase, was_swinging, plan_foot, rows):
    """[N] phase shifted half a cycle where a swing window has just opened
    for a foot the plan is not waiting for (after a miss the same foot
    swings again). Only rows in ``rows`` are touched; inside a swing nothing
    moves (an early hit already switches the plan to the next foot)."""
    swing, assigned = clock_schedule(phase)
    opened = swing.any(dim=-1) & ~was_swinging
    shift = rows & opened & (assigned != plan_foot)
    return torch.where(shift, torch.remainder(phase + 0.5, 1.0), phase)


def own_window(phase, foot):
    """[N] True during ``foot``'s swing window and the double support after
    it (left: phase 0.15-0.65, right: 0.65-1.0 and 0.0-0.15)."""
    ds, ss = DOUBLE_SUPPORT_FRACTION, SINGLE_SUPPORT_FRACTION
    left = (phase >= ds) & (phase < 2 * ds + ss)
    return torch.where(foot == 0, left, ~left)


def approach_steps_left(distance, step_len):
    """[N] whole steps (>= 1) from the stance foot to the first stair
    placement ``distance`` ahead; the last may be up to 25% longer than
    ``step_len``. A passed placement counts as one step."""
    return torch.ceil(distance / step_len.clamp_min(1e-3)
                      - APPROACH_STEP_TOLERANCE).clamp_min(1.0)


def clock_features(phase, active):
    """[N,2] sin/cos of the cycle; zero while standing."""
    angle = 2 * math.pi * phase
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    return torch.where(active[:, None], clock, torch.zeros_like(clock))


def yaw_drift_cost(cycle_yaw_error):
    """[N] cost in [0, 1): 0 when the cycle-average turn rate matches the command."""
    return 1.0 - torch.exp(-(cycle_yaw_error / YAW_DRIFT_STD_RADPS) ** 2)


def heading_targets(stance_xy, swing_foot, root_yaw, command, period_s,
                    foot_separation_m, step_len=None):
    """[N,2,4] targets (x, y, z=0, yaw) and [N,2] feet for the next two steps.

    Slot 0: the swing foot, one step (speed x cycle / 2, or ``step_len``)
    ahead of the stance foot in its own lane, along the commanded heading at
    its touchdown (half a cycle after assignment). Slot 1: the following
    step of the other foot. A sideways command shifts each step by
    vy x cycle / 2.
    """
    n = stance_xy.shape[0]
    if (stance_xy.shape != (n, 2) or swing_foot.shape != (n,) or root_yaw.shape != (n,)
            or command.shape != (n, 3) or period_s.shape != (n,)):
        raise ValueError("invalid heading target shapes")
    if foot_separation_m <= 0:
        raise ValueError("foot_separation_m must be positive")
    side = torch.where(swing_foot == 0, 1.0, -1.0).to(stance_xy.dtype)
    half = 0.5 * period_s
    if step_len is None:
        step_len = command[:, 0] * half
    elif step_len.shape != (n,):
        raise ValueError("invalid step length shape")
    shift = command[:, 1] * half

    def step(origin, yaw, lane):
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        left = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        return (origin + step_len[:, None] * forward
                + (lane * foot_separation_m + shift)[:, None] * left)

    yaw1 = root_yaw + command[:, 2] * half
    yaw2 = yaw1 + command[:, 2] * half
    first = step(stance_xy, yaw1, side)
    second = step(first, yaw2, -side)
    zero = torch.zeros_like(root_yaw)
    targets = torch.stack((
        torch.cat((first, zero[:, None], yaw1[:, None]), dim=-1),
        torch.cat((second, zero[:, None], yaw2[:, None]), dim=-1),
    ), dim=1)
    feet = torch.stack((swing_foot, 1 - swing_foot), dim=-1)
    return targets, feet


class RollingClockState:
    """Per-env clock phase, cycle time and the current rolling targets."""

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.phase = torch.zeros(num_envs, device=self.device)
        self.period_scale = torch.ones(num_envs, device=self.device)
        self.period_s = torch.ones(num_envs, device=self.device)
        self.assigned = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.window = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.acquired = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.targets = torch.zeros((num_envs, 2, 4), device=self.device)
        self.feet = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.swinging = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.on_plan = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # Stair plan completed: the row walks on like a flat tile.
        self.climbed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # [N,2] foot reached its target in its current swing window: the
        # swing is over, the rest of the window counts as stance.
        self.landed = torch.zeros((num_envs, 2), dtype=torch.bool, device=self.device)
        self.yaw_error_sum = torch.zeros(num_envs, device=self.device)
        self.yaw_error_time = torch.zeros(num_envs, device=self.device)
        self.cycle_yaw_error = torch.zeros(num_envs, device=self.device)

    def reset(self, env_ids, jitter):
        """New cycle-time scale per episode (1 +- jitter); random lead foot
        via half-cycle offset."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None \
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = ids.numel()
        self.period_scale[ids] = torch.empty(count, device=self.device).uniform_(
            1.0 - jitter, 1.0 + jitter)
        self.phase[ids] = 0.5 * torch.randint(0, 2, (count,), device=self.device).float()
        self.assigned[ids] = -1
        self.window[ids] = 0
        self.acquired[ids] = False
        self.targets[ids] = 0.0
        self.active[ids] = False
        self.swinging[ids] = False
        self.on_plan[ids] = False
        self.climbed[ids] = False
        self.landed[ids] = False
        self.yaw_error_sum[ids] = 0.0
        self.yaw_error_time[ids] = 0.0
        self.cycle_yaw_error[ids] = 0.0

    def advance(self, dt, rows, yaw_rate_error=None):
        """Move the phase of walking rows by dt.

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
