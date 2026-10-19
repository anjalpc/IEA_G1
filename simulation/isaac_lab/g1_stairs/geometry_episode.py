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
    STEP_OVER_CONTRACT = 12


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

    strict_stair_gait: bool = False
    stable_finish: bool = False
    single_acquisition_success: bool = False
    training_recovery_s: float = 0.0
    training_permissive_contact: bool = False
    finish_sole_margin_m: float = 0.05
    finish_max_linear_speed_mps: float = 0.15
    finish_max_angular_speed_radps: float = 0.30
    finish_max_foot_speed_mps: float = 0.10
    finish_max_roll_pitch_rad: float = 0.20

    def __post_init__(self):
        if not isinstance(self.strict_stair_gait, bool):
            raise ValueError("strict_stair_gait must be a boolean")
        if not isinstance(self.stable_finish, bool):
            raise ValueError("stable_finish must be a boolean")
        if not isinstance(self.single_acquisition_success, bool):
            raise ValueError("single_acquisition_success must be a boolean")
        if not isinstance(self.training_permissive_contact, bool):
            raise ValueError("training_permissive_contact must be a boolean")
        if self.training_permissive_contact and not self.strict_stair_gait:
            raise ValueError("permissive contact training requires strict stair gait")
        if self.training_permissive_contact and self.training_recovery_s:
            raise ValueError("permissive contact training cannot combine with bounded recovery")
        if not math.isfinite(self.training_recovery_s) or self.training_recovery_s < 0:
            raise ValueError("training_recovery_s must be finite and nonnegative")
        if self.training_recovery_s and not self.strict_stair_gait:
            raise ValueError("training recovery requires strict stair gait")
        if self.single_acquisition_success and not self.strict_stair_gait:
            raise ValueError(
                "single_acquisition_success requires strict_stair_gait"
            )
        for name, value in vars(self).items():
            if name in (
                "strict_stair_gait",
                "stable_finish",
                "single_acquisition_success",
                "training_recovery_s",
                "training_permissive_contact",
            ):
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"episode_contract.{name} must be finite and positive")
        if self.maximum_roll_pitch_rad >= math.pi / 2:
            raise ValueError("maximum_roll_pitch_rad must be below pi/2")
        if self.finish_max_roll_pitch_rad > self.maximum_roll_pitch_rad:
            raise ValueError("finish tilt limit must not exceed fall tilt limit")


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
        self.best_dwell = torch.zeros(num_envs, device=device)
        self.dwell_peak_increment = torch.zeros(num_envs, device=device)
        # Diagnostic milestones are intentionally separate from the strict
        # finish predicate. platform_contact_seen means at least one foot made
        # valid load-bearing contact on the upper platform. platform_seen keeps
        # its historical meaning: the episode satisfied the currently active
        # supported-platform predicate (strict settling when stable_finish=True).
        self.platform_contact_seen = torch.zeros(
            num_envs, device=device, dtype=torch.bool
        )
        self.platform_seen = torch.zeros_like(self.platform_contact_seen)
        self.entered_stairs = torch.zeros_like(self.platform_contact_seen)
        self.phase = torch.zeros_like(self.reason)

    def reset(self, ids):
        for value in (self.reason, self.started, self.start_x, self.best_x,
                      self.stall_time, self.dwell, self.best_dwell,
                      self.dwell_peak_increment, self.platform_contact_seen,
                      self.platform_seen, self.entered_stairs, self.phase):
            value[ids] = 0

    def update(self, *, position, roll_pitch, riser_height, foot_position,
               foot_force_z, forbidden_contact, invalid_state, joint_limit,
               velocity_limit, torque_limit, timed_out,
               step_over_contract_violation=None, strict_gait_complete=None,
               strict_first_acquisition=None,
               sole_bounds=None, body_linear_velocity=None,
               body_angular_velocity=None, foot_linear_velocity=None):
        active = self.reason == Result.RUNNING

        if step_over_contract_violation is None:
            step_over_contract_violation = torch.zeros_like(active)
        else:
            if step_over_contract_violation.shape != active.shape:
                raise ValueError(
                    "step_over_contract_violation must match episode shape"
                )
            step_over_contract_violation = (
                step_over_contract_violation
                .to(device=active.device, dtype=torch.bool)
            )

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
        # Preserve a physically meaningful climbing milestone before the V05
        # stable-finish branch tightens support to two feet + low velocities.
        # This is diagnostic only and does not affect termination or reward.
        upper_platform_contact = (
            (x >= self.platform_start) & support & upright & clearance_ok
        )
        if self.cfg.stable_finish:
            if any(v is None for v in (
                sole_bounds, body_linear_velocity,
                body_angular_velocity, foot_linear_velocity,
            )):
                raise ValueError("stable finish requires sole bounds and velocities")
            # Bounds are [xmin, xmax, ymin, ymax, zmin, zmax] per foot.
            margin = self.cfg.finish_sole_margin_m
            finite_finish = torch.ones_like(active)
            for value in (
                sole_bounds, body_linear_velocity,
                body_angular_velocity, foot_linear_velocity,
            ):
                finite_finish &= torch.isfinite(value).flatten(1).all(-1)
            invalid_state = invalid_state | ~finite_finish
            placement = (
                (sole_bounds[:, :, 0] >= self.platform_start + margin)
                & (sole_bounds[:, :, 1] <= self.course_end - self.cfg.boundary_margin_m)
                & (sole_bounds[:, :, 2] >= -self.half_width + margin)
                & (sole_bounds[:, :, 3] <= self.half_width - margin)
                & ((sole_bounds[:, :, 4] - platform_z[:, None]).abs()
                   <= self.cfg.sole_height_tolerance_m)
                & ((sole_bounds[:, :, 5] - platform_z[:, None]).abs()
                   <= self.cfg.sole_height_tolerance_m)
            )
            support = (
                placement
                & (foot_force_z >= self.cfg.support_force_n)
                & (torch.linalg.vector_norm(foot_linear_velocity, dim=-1)
                   <= self.cfg.finish_max_foot_speed_mps)
            ).all(-1)
            settled = (
                (torch.linalg.vector_norm(body_linear_velocity, dim=-1)
                 <= self.cfg.finish_max_linear_speed_mps)
                & (torch.linalg.vector_norm(body_angular_velocity, dim=-1)
                   <= self.cfg.finish_max_angular_speed_radps)
                & (roll_pitch.abs() <= self.cfg.finish_max_roll_pitch_rad).all(-1)
            )
            support &= settled & finite_finish

        supported_platform = (x >= self.platform_start) & support & upright & clearance_ok
        valid = active & ~invalid_state & ~forbidden_contact & ~boundary
        # Keep physical platform diagnostics independent of the strict gait
        # success gate. A robot that reaches the platform with bad gait must
        # remain visible as physical traversal, not disappear from metrics.
        self.platform_contact_seen |= valid & upper_platform_contact
        self.platform_seen |= valid & supported_platform
        self.entered_stairs |= active & (x >= self.first_riser) & (riser_height > 0)
        success_valid = valid
        first_acquisition_success = torch.zeros_like(active)

        if self.cfg.single_acquisition_success:
            if strict_first_acquisition is None:
                raise ValueError(
                    "single acquisition success requires strict gait evidence"
                )
            if strict_first_acquisition.shape != active.shape:
                raise ValueError(
                    "strict_first_acquisition must match episode shape"
                )
            first_acquisition_success = (
                success_valid
                & strict_first_acquisition.to(
                    device=active.device, dtype=torch.bool
                )
            )
        elif self.cfg.strict_stair_gait:
            if strict_gait_complete is None:
                raise ValueError("strict stair success requires physical gait evidence")
            if strict_gait_complete.shape != active.shape:
                raise ValueError("strict_gait_complete must match episode shape")
            success_valid = success_valid & strict_gait_complete.to(
                device=active.device, dtype=torch.bool
            )

        # Phase is diagnostic. Reaching a load-bearing upper-platform contact
        # is enough to classify a later failure as a platform failure even if
        # the strict V05 settling predicate was never satisfied.
        self.phase = torch.where(
            self.platform_contact_seen, 2,
            torch.where(x >= self.first_riser, 1, 0),
        )
        dwelling = success_valid & supported_platform & (x >= self.completion_x)
        self.dwell = torch.where(active, torch.where(dwelling, self.dwell + self.dt, 0.0), self.dwell)
        new_best_dwell = torch.maximum(self.best_dwell, self.dwell)
        self.dwell_peak_increment = torch.where(
            active,
            (new_best_dwell - self.best_dwell).clamp_min(0.0),
            torch.zeros_like(self.dwell_peak_increment),
        )
        self.best_dwell = torch.where(active, new_best_dwell, self.best_dwell)
        if self.cfg.stable_finish:
            # An intentionally stationary, valid finishing pose is not a stall.
            self.stall_time = torch.where(dwelling, 0.0, self.stall_time)
        # Fixed failure priority. Success never overrides a failure at the same
        # control step; timeout is the only bootstrapped truncation.
        failures = (
            (Result.INVALID_STATE, invalid_state),
            (Result.STEP_OVER_CONTRACT, step_over_contract_violation),
            (Result.FALL, ~upright | ~clearance_ok),
            (Result.FORBIDDEN_CONTACT, forbidden_contact),
            (Result.COURSE_BOUNDARY, boundary),
            (Result.JOINT_LIMIT, joint_limit),
            (Result.VELOCITY_LIMIT, velocity_limit),
            (Result.TORQUE_LIMIT, torque_limit),
            (Result.WRONG_DIRECTION, x < self.start_x - self.cfg.wrong_direction_tolerance_m),
            (Result.STALL, self.stall_time >= self.cfg.stall_duration_s),
            (
                Result.SUCCESS,
                first_acquisition_success
                if self.cfg.single_acquisition_success
                else self.dwell + 1.e-6 >= self.cfg.supported_dwell_s,
            ),
            (Result.TIMEOUT, timed_out),
        )
        for reason, mask in failures:
            self.reason[(self.reason == Result.RUNNING) & mask] = reason
        return self.reason
