"""Vectorized, reward-independent outcomes for geometry locomotion V03.

No Isaac imports: the same state machine is tested on CPU and used by training
and batch evaluation. Coordinates are relative to the assigned terrain origin.
"""

from dataclasses import dataclass
from enum import IntEnum
import math

import torch


class Result(IntEnum):
    RUNNING = 0
    SUCCESS = 1
    INVALID_STATE = 2
    FALL = 3
    FORBIDDEN_CONTACT = 4
    COURSE_BOUNDARY = 5
    JOINT_LIMIT = 6
    VELOCITY_LIMIT = 7
    TORQUE_LIMIT = 8
    WRONG_DIRECTION = 9
    STALL = 10
    TIMEOUT = 11


@dataclass(frozen=True)
class EpisodeContract:
    platform_continuation_m: float = 2.0
    supported_dwell_s: float = 1.0
    minimum_base_clearance_m: float = 0.45
    maximum_roll_pitch_rad: float = 0.8
    forbidden_contact_force_n: float = 1.0
    support_force_n: float = 30.0
    sole_height_tolerance_m: float = 0.025
    boundary_margin_m: float = 0.15
    stall_duration_s: float = 2.0
    progress_epsilon_m: float = 0.02
    wrong_direction_tolerance_m: float = 0.25

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"episode_contract.{name} must be finite and positive")
        if self.maximum_roll_pitch_rad >= math.pi / 2:
            raise ValueError("maximum_roll_pitch_rad must be below pi/2")


class EpisodeTracker:
    def __init__(self, num_envs, device, contract, *, first_riser, tread,
                 step_count, course_end, course_start, half_width, dt):
        self.cfg = contract
        self.first_riser = float(first_riser)
        self.tread = float(tread)
        self.step_count = int(step_count)
        self.platform_start = first_riser + step_count * tread
        self.completion_x = self.platform_start + contract.platform_continuation_m
        self.course_end = float(course_end)
        self.course_start = float(course_start)
        self.half_width = float(half_width)
        self.dt = float(dt)
        if not (dt > 0 and tread > 0 and step_count > 0):
            raise ValueError("invalid episode geometry/timestep")
        if not (course_start < first_riser < self.completion_x
                < course_end - contract.boundary_margin_m):
            raise ValueError("completion must be inside the assigned terrain tile")
        if half_width <= contract.boundary_margin_m:
            raise ValueError("boundary margin leaves no traversable width")
        self.reason = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.started = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.start_x = torch.zeros(num_envs, device=device)
        self.best_x = torch.zeros(num_envs, device=device)
        self.stall_time = torch.zeros(num_envs, device=device)
        self.dwell = torch.zeros(num_envs, device=device)
        self.platform_seen = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.entered_stairs = torch.zeros_like(self.platform_seen)
        self.phase = torch.zeros_like(self.reason)

    def reset(self, ids):
        for value in (self.reason, self.started, self.start_x, self.best_x,
                      self.stall_time, self.dwell, self.platform_seen,
                      self.entered_stairs, self.phase):
            value[ids] = 0

    def update(self, *, position, roll_pitch, riser_height, foot_position,
               foot_force_z, forbidden_contact, invalid_state, joint_limit,
               velocity_limit, torque_limit, timed_out):
        active = self.reason == Result.RUNNING
        x, y, z = position.unbind(-1)
        new = active & ~self.started
        self.start_x[new] = x[new]
        self.best_x[new] = x[new]
        self.started |= active
        progress = x >= self.best_x + self.cfg.progress_epsilon_m
        self.best_x = torch.where(active & progress, x, self.best_x)
        self.stall_time = torch.where(
            active & (new | progress), 0.0,
            self.stall_time + active * self.dt,
        )
        base_level = torch.clamp(
            torch.floor((x - self.first_riser) / self.tread) + 1,
            0, self.step_count,
        )
        ground_z = base_level * riser_height
        upright = (roll_pitch.abs() <= self.cfg.maximum_roll_pitch_rad).all(-1)
        clearance_ok = z - ground_z >= self.cfg.minimum_base_clearance_m
        boundary = (
            (x < self.course_start + self.cfg.boundary_margin_m)
            | (x > self.course_end - self.cfg.boundary_margin_m)
            | (y.abs() > self.half_width - self.cfg.boundary_margin_m)
        )
        platform_z = self.step_count * riser_height
        # Require upward ground reaction AND a sole on this platform. A riser
        # collision or airborne crossing must not count as platform support.
        support = (
            (foot_force_z >= self.cfg.support_force_n)
            & (foot_position[:, :, 0] >= self.platform_start)
            & (foot_position[:, :, 0] < self.course_end)
            & (foot_position[:, :, 1].abs() < self.half_width)
            & ((foot_position[:, :, 2] - platform_z[:, None]).abs()
               <= self.cfg.sole_height_tolerance_m)
        ).any(-1)
        supported_platform = (x >= self.platform_start) & support & upright & clearance_ok
        valid = active & ~invalid_state & ~forbidden_contact & ~boundary
        self.platform_seen |= valid & supported_platform
        self.entered_stairs |= active & (x >= self.first_riser) & (riser_height > 0)
        self.phase = torch.where(self.platform_seen, 2,
                                 torch.where(x >= self.first_riser, 1, 0))
        dwelling = valid & supported_platform & (x >= self.completion_x)
        self.dwell = torch.where(active, torch.where(dwelling, self.dwell + self.dt, 0.0), self.dwell)
        # Fixed failure priority. Success never overrides a failure at the same
        # control step; timeout is the only bootstrapped truncation.
        failures = (
            (Result.INVALID_STATE, invalid_state),
            (Result.FALL, ~upright | ~clearance_ok),
            (Result.FORBIDDEN_CONTACT, forbidden_contact),
            (Result.COURSE_BOUNDARY, boundary),
            (Result.JOINT_LIMIT, joint_limit),
            (Result.VELOCITY_LIMIT, velocity_limit),
            (Result.TORQUE_LIMIT, torque_limit),
            (Result.WRONG_DIRECTION, x < self.start_x - self.cfg.wrong_direction_tolerance_m),
            (Result.STALL, self.stall_time >= self.cfg.stall_duration_s),
            (Result.SUCCESS, self.dwell + 1.e-6 >= self.cfg.supported_dwell_s),
            (Result.TIMEOUT, timed_out),
        )
        for reason, mask in failures:
            self.reason[(self.reason == Result.RUNNING) & mask] = reason
        return self.reason
