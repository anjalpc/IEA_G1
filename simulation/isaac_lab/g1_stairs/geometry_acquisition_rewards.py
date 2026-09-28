"""V04 geometry-relative stair acquisition rewards.

This module intentionally contains no prescribed left/right gait sequence and
no specialist phase machine.

Each foot independently targets its next unacquired riser. A foot advances to
the next target only after verified physical support on the corresponding tread.

Rewards:
- terrain_relative_clearance: monotonic bounded progress toward sufficient
  sole clearance above the foot's next riser.
- forward_insertion: monotonic bounded progress into the safe next-tread
  landing region, gated by sufficient clearance.

The shared runtime also exposes reward-independent activation, touchdown and
contact diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .contracts import (
    G1_SOLE_SPHERE_CENTRES_M,
    G1_SOLE_SPHERE_RADIUS_M,
)
from .terrain_state import riser_height_per_env, stair_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

FORBIDDEN_CONTACT_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_rubber_hand",
    "right_rubber_hand",
)


def _required_indices(available_names, required_names) -> list[int]:
    index = {name: i for i, name in enumerate(available_names)}
    missing = [name for name in required_names if name not in index]
    if missing:
        raise RuntimeError(
            "V04 could not resolve required bodies: "
            + ", ".join(missing)
        )
    return [index[name] for name in required_names]


def _available_indices(available_names, requested_names):
    index = {name: i for i, name in enumerate(available_names)}
    names = [name for name in requested_names if name in index]
    return names, [index[name] for name in names]


def _quat_rotate_wxyz(q: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by WXYZ quaternions."""
    xyz = q[..., 1:]
    uv = torch.cross(xyz, vectors, dim=-1)
    return (
        vectors
        + 2.0
        * (
            q[..., :1] * uv
            + torch.cross(xyz, uv, dim=-1)
        )
    )


def _smooth_rising(
    value: torch.Tensor,
    start: float,
    full: float,
) -> torch.Tensor:
    if not full > start:
        raise ValueError("smooth rising gate requires full > start")
    return ((value - start) / (full - start)).clamp(0.0, 1.0)


def _smooth_falling(
    value: torch.Tensor,
    full: float,
    zero: float,
) -> torch.Tensor:
    if not zero > full:
        raise ValueError("smooth falling gate requires zero > full")
    return ((zero - value) / (zero - full)).clamp(0.0, 1.0)


class GeometryAcquisitionRuntime:
    """Shared per-environment physical state for the two V04 reward terms."""

    def __init__(self, env: "ManagerBasedRLEnv"):
        self.robot = env.scene["robot"]

        sensor = env.scene.sensors.get("contact_forces")
        if sensor is None:
            raise RuntimeError("V04 requires contact_forces sensor")
        self.sensor = sensor

        self.robot_foot_ids = _required_indices(
            self.robot.body_names,
            FOOT_BODY_NAMES,
        )
        self.sensor_foot_ids = _required_indices(
            self.sensor.body_names,
            FOOT_BODY_NAMES,
        )

        (
            self.forbidden_names,
            self.forbidden_ids,
        ) = _available_indices(
            self.sensor.body_names,
            FORBIDDEN_CONTACT_BODY_NAMES,
        )

        self.sole_centres = torch.tensor(
            G1_SOLE_SPHERE_CENTRES_M,
            dtype=torch.float32,
            device=env.device,
        )

        shape = (env.num_envs, 2)

        # Each foot independently tracks the next stair it still needs to
        # acquire. 0 -> first tread, 1 -> second, 2 -> third.
        self.target_step = torch.zeros(
            shape,
            dtype=torch.long,
            device=env.device,
        )

        # Monotonic episode-local maxima prevent reward recycling by repeatedly
        # moving backward and forward through the same geometry.
        self.best_clearance = torch.zeros(
            shape,
            dtype=torch.float32,
            device=env.device,
        )
        self.best_insertion = torch.zeros_like(self.best_clearance)

        self.clearance_reward = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self.insertion_reward = torch.zeros_like(self.clearance_reward)

        self.last_step = -1

        # Iteration diagnostics. These counters are consumed by the metrics
        # wrapper without changing episode-local acquisition state.
        self.candidate_samples = torch.zeros(
            (), dtype=torch.float32, device=env.device
        )
        self.clearance_active_samples = torch.zeros_like(
            self.candidate_samples
        )
        self.insertion_active_samples = torch.zeros_like(
            self.candidate_samples
        )
        self.clearance_progress_events = torch.zeros_like(
            self.candidate_samples
        )
        self.insertion_progress_events = torch.zeros_like(
            self.candidate_samples
        )
        self.touchdowns = torch.zeros(
            3,
            dtype=torch.long,
            device=env.device,
        )

        count = len(self.forbidden_ids)
        self.contact_max_force = torch.zeros(
            count, dtype=torch.float32, device=env.device
        )
        self.contact_events = torch.zeros(
            count, dtype=torch.long, device=env.device
        )
        self.contact_total_duration = torch.zeros(
            count, dtype=torch.float32, device=env.device
        )
        self.contact_max_duration = torch.zeros_like(
            self.contact_total_duration
        )
        self.contact_current_duration = torch.zeros(
            (env.num_envs, count),
            dtype=torch.float32,
            device=env.device,
        )
        self.contact_previous_over = torch.zeros(
            (env.num_envs, count),
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self.target_step[env_ids] = 0
        self.best_clearance[env_ids] = 0.0
        self.best_insertion[env_ids] = 0.0

        if self.contact_current_duration.shape[1] > 0:
            self.contact_current_duration[env_ids] = 0.0
            self.contact_previous_over[env_ids] = False

    def update(
        self,
        env: "ManagerBasedRLEnv",
        *,
        activation_distance_m: float,
        clearance_margin_m: float,
        tread_margin_m: float,
        insertion_target_m: float,
        unload_full_force_n: float,
        unload_zero_force_n: float,
        opposite_support_start_n: float,
        opposite_support_full_n: float,
        insertion_clearance_fraction: float,
        touchdown_force_n: float,
        touchdown_speed_mps: float,
        touchdown_height_tolerance_m: float,
        diagnostic_contact_threshold_n: float,
    ) -> None:
        step = int(env.common_step_counter)
        if self.last_step == step:
            return
        self.last_step = step

        if activation_distance_m <= 0.0:
            raise ValueError("activation_distance_m must be positive")
        if clearance_margin_m < 0.0:
            raise ValueError("clearance_margin_m cannot be negative")
        if tread_margin_m <= 0.0:
            raise ValueError("tread_margin_m must be positive")
        if insertion_target_m <= 0.0:
            raise ValueError("insertion_target_m must be positive")
        if not 0.0 <= insertion_clearance_fraction < 1.0:
            raise ValueError(
                "insertion_clearance_fraction must be in [0, 1)"
            )

        cfg = stair_cfg(env)
        riser_height = riser_height_per_env(env)
        tread = float(cfg.tread_depth)
        first_riser = float(cfg.start_clearance)
        step_count = int(cfg.step_count)

        if step_count != 3:
            raise RuntimeError(
                f"V04 currently expects exactly 3 steps, got {step_count}"
            )
        if not 2.0 * tread_margin_m < tread:
            raise ValueError("tread margin leaves no safe landing region")

        data = self.robot.data

        foot_pos = (
            data.body_pos_w[:, self.robot_foot_ids]
            - env.scene.env_origins[:, None, :]
        )
        foot_quat = data.body_quat_w[:, self.robot_foot_ids]

        offsets = self.sole_centres.to(
            dtype=foot_pos.dtype
        )[None, None].expand(
            env.num_envs,
            2,
            -1,
            -1,
        )

        quaternions = foot_quat.unsqueeze(2).expand(
            -1,
            -1,
            offsets.shape[2],
            -1,
        )

        sole_points = (
            foot_pos.unsqueeze(2)
            + _quat_rotate_wxyz(quaternions, offsets)
        )

        radius = float(G1_SOLE_SPHERE_RADIUS_M)

        sole_bottom_points = (
            sole_points[..., 2] - radius
        )
        sole_bottom = sole_bottom_points.amin(dim=2)

        rear_x = (
            sole_points[..., 0].amin(dim=2) - radius
        )
        front_x = (
            sole_points[..., 0].amax(dim=2) + radius
        )

        if hasattr(data, "body_lin_vel_w"):
            foot_speed = torch.linalg.vector_norm(
                data.body_lin_vel_w[:, self.robot_foot_ids],
                dim=2,
            )
        else:
            foot_speed = torch.zeros_like(sole_bottom)

        vertical_force = (
            self.sensor.data.net_forces_w[
                :,
                self.sensor_foot_ids,
                2,
            ]
            .clamp_min(0.0)
        )

        valid_target = (
            (self.target_step < step_count)
            & (riser_height[:, None] > 1.0e-8)
        )

        target_index = self.target_step.clamp(
            min=0,
            max=step_count - 1,
        )

        next_riser_x = (
            first_riser
            + target_index.to(foot_pos.dtype) * tread
        )

        support_height = (
            target_index.to(foot_pos.dtype)
            * riser_height[:, None]
        )

        target_tread_height = (
            (target_index + 1).to(foot_pos.dtype)
            * riser_height[:, None]
        )

        relative_height = (
            sole_bottom - support_height
        ).clamp_min(0.0)

        target_clearance = (
            riser_height[:, None]
            + float(clearance_margin_m)
        ).clamp_min(1.0e-6)

        clearance_fraction = (
            relative_height / target_clearance
        ).clamp(0.0, 1.0)

        # Activate smoothly while approaching the target edge.
        distance_to_edge = next_riser_x - front_x
        approach_gate = (
            1.0
            - distance_to_edge / float(activation_distance_m)
        ).clamp(0.0, 1.0)

        # No "already airborne" condition. A candidate foot begins earning
        # shaping while unloading, provided the opposite foot is supporting.
        own_unload = _smooth_falling(
            vertical_force,
            float(unload_full_force_n),
            float(unload_zero_force_n),
        )

        opposite_force = torch.stack(
            (
                vertical_force[:, 1],
                vertical_force[:, 0],
            ),
            dim=1,
        )

        opposite_support = _smooth_rising(
            opposite_force,
            float(opposite_support_start_n),
            float(opposite_support_full_n),
        )

        swing_gate = (
            own_unload
            * opposite_support
            * approach_gate
            * valid_target.to(foot_pos.dtype)
        )

        clearance_score = (
            clearance_fraction * swing_gate
        ).clamp(0.0, 1.0)

        new_clearance_peak = torch.maximum(
            self.best_clearance,
            clearance_score,
        )

        clearance_delta = (
            new_clearance_peak - self.best_clearance
        ).clamp_min(0.0)

        self.best_clearance.copy_(new_clearance_peak)

        # Insertion begins only once substantial clearance has been achieved.
        clearance_gate = (
            (
                clearance_fraction
                - float(insertion_clearance_fraction)
            )
            / (
                1.0
                - float(insertion_clearance_fraction)
            )
        ).clamp(0.0, 1.0)

        safe_start = next_riser_x + float(tread_margin_m)
        safe_end = (
            next_riser_x
            + tread
            - float(tread_margin_m)
        )

        insertion_target = torch.minimum(
            safe_start + float(insertion_target_m),
            safe_end - 1.0e-3,
        )

        insertion_denominator = (
            insertion_target - safe_start
        ).clamp_min(1.0e-4)

        insertion_fraction = (
            (rear_x - safe_start)
            / insertion_denominator
        ).clamp(0.0, 1.0)

        # Do not reward a foot that has already overshot the safe tread.
        safe_end_gate = (
            (safe_end - front_x)
            / float(tread_margin_m)
        ).clamp(0.0, 1.0)

        insertion_score = (
            insertion_fraction
            * clearance_gate
            * swing_gate
            * safe_end_gate
        ).clamp(0.0, 1.0)

        new_insertion_peak = torch.maximum(
            self.best_insertion,
            insertion_score,
        )

        insertion_delta = (
            new_insertion_peak - self.best_insertion
        ).clamp_min(0.0)

        self.best_insertion.copy_(new_insertion_peak)

        # RewardManager multiplies the returned value by dt. Dividing the
        # bounded monotonic increments by dt makes total credit independent
        # of controller frequency.
        dt = float(env.step_dt)

        self.clearance_reward.copy_(
            clearance_delta.sum(dim=1) / dt
        )
        self.insertion_reward.copy_(
            insertion_delta.sum(dim=1) / dt
        )

        # Verified physical touchdown. This advances only the corresponding
        # foot's target; there is no prescribed gait order.
        touchdown = (
            valid_target
            & (rear_x >= safe_start)
            & (front_x <= safe_end)
            & (
                (
                    sole_bottom_points
                    - target_tread_height.unsqueeze(2)
                ).abs().amax(dim=2)
                <= float(touchdown_height_tolerance_m)
            )
            & (
                vertical_force
                >= float(touchdown_force_n)
            )
            & (
                foot_speed
                <= float(touchdown_speed_mps)
            )
        )

        if bool(touchdown.any()):
            old_target = self.target_step.clone()

            for stair_index in range(step_count):
                self.touchdowns[stair_index] += torch.count_nonzero(
                    touchdown
                    & (old_target == stair_index)
                )

            self.target_step.copy_(
                torch.where(
                    touchdown,
                    self.target_step + 1,
                    self.target_step,
                )
            )

            # New stair = new non-recyclable potential.
            self.best_clearance.copy_(
                torch.where(
                    touchdown,
                    torch.zeros_like(self.best_clearance),
                    self.best_clearance,
                )
            )
            self.best_insertion.copy_(
                torch.where(
                    touchdown,
                    torch.zeros_like(self.best_insertion),
                    self.best_insertion,
                )
            )

        # Activation / progress diagnostics.
        self.candidate_samples += valid_target.sum()
        self.clearance_active_samples += (
            (swing_gate > 0.05) & valid_target
        ).sum()
        self.insertion_active_samples += (
            (clearance_gate * swing_gate > 0.05)
            & valid_target
        ).sum()
        self.clearance_progress_events += (
            clearance_delta > 1.0e-6
        ).sum()
        self.insertion_progress_events += (
            insertion_delta > 1.0e-6
        ).sum()

        # Diagnostic-only forbidden-body contact force and duration.
        if self.forbidden_ids:
            forces = self.sensor.data.net_forces_w[
                :,
                self.forbidden_ids,
            ]
            magnitudes = torch.linalg.vector_norm(
                forces,
                dim=-1,
            )

            self.contact_max_force.copy_(
                torch.maximum(
                    self.contact_max_force,
                    magnitudes.amax(dim=0),
                )
            )

            over = (
                magnitudes
                > float(diagnostic_contact_threshold_n)
            )

            new_event = (
                over & ~self.contact_previous_over
            )

            self.contact_events += new_event.sum(dim=0)

            self.contact_total_duration += (
                over.to(torch.float32).sum(dim=0) * dt
            )

            self.contact_current_duration.copy_(
                torch.where(
                    over,
                    self.contact_current_duration + dt,
                    torch.zeros_like(
                        self.contact_current_duration
                    ),
                )
            )

            self.contact_max_duration.copy_(
                torch.maximum(
                    self.contact_max_duration,
                    self.contact_current_duration.amax(dim=0),
                )
            )

            self.contact_previous_over.copy_(over)

    def consume_metrics(self) -> dict[str, float]:
        """Return iteration diagnostics without resetting episode progress."""

        candidate = float(self.candidate_samples.item())

        result = {
            "v04_clearance_activation_fraction": (
                float(self.clearance_active_samples.item())
                / candidate
                if candidate > 0.0
                else 0.0
            ),
            "v04_insertion_activation_fraction": (
                float(self.insertion_active_samples.item())
                / candidate
                if candidate > 0.0
                else 0.0
            ),
            "v04_clearance_progress_events": float(
                self.clearance_progress_events.item()
            ),
            "v04_insertion_progress_events": float(
                self.insertion_progress_events.item()
            ),
            "v04_supported_touchdowns_total": float(
                self.touchdowns.sum().item()
            ),
            "v04_supported_touchdowns_step1": float(
                self.touchdowns[0].item()
            ),
            "v04_supported_touchdowns_step2": float(
                self.touchdowns[1].item()
            ),
            "v04_supported_touchdowns_step3": float(
                self.touchdowns[2].item()
            ),
        }

        for i, name in enumerate(self.forbidden_names):
            prefix = f"v04_contact_{name}"
            result[f"{prefix}_max_force_n"] = float(
                self.contact_max_force[i].item()
            )
            result[f"{prefix}_events_above_threshold"] = float(
                self.contact_events[i].item()
            )
            result[f"{prefix}_total_duration_s"] = float(
                self.contact_total_duration[i].item()
            )
            result[f"{prefix}_max_continuous_duration_s"] = float(
                self.contact_max_duration[i].item()
            )

        self.candidate_samples.zero_()
        self.clearance_active_samples.zero_()
        self.insertion_active_samples.zero_()
        self.clearance_progress_events.zero_()
        self.insertion_progress_events.zero_()
        self.touchdowns.zero_()

        self.contact_max_force.zero_()
        self.contact_events.zero_()
        self.contact_total_duration.zero_()
        self.contact_max_duration.zero_()

        return result


def _runtime(env: "ManagerBasedRLEnv") -> GeometryAcquisitionRuntime:
    runtime = getattr(
        env,
        "_g1_v04_acquisition",
        None,
    )
    if runtime is None:
        runtime = GeometryAcquisitionRuntime(env)
        env._g1_v04_acquisition = runtime
    return runtime


_COMMON_DEFAULTS = dict(
    activation_distance_m=0.30,
    clearance_margin_m=0.015,
    tread_margin_m=0.015,
    insertion_target_m=0.040,
    unload_full_force_n=15.0,
    unload_zero_force_n=80.0,
    opposite_support_start_n=30.0,
    opposite_support_full_n=100.0,
    insertion_clearance_fraction=0.70,
    touchdown_force_n=30.0,
    touchdown_speed_mps=0.20,
    touchdown_height_tolerance_m=0.025,
    diagnostic_contact_threshold_n=1.0,
)


class terrain_relative_clearance(ManagerTermBase):
    """Bounded non-recyclable sole-clearance progress."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        self.runtime.reset(env_ids)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
            "contact_forces"
        ),
        activation_distance_m: float = 0.30,
        clearance_margin_m: float = 0.015,
        tread_margin_m: float = 0.015,
        insertion_target_m: float = 0.040,
        unload_full_force_n: float = 15.0,
        unload_zero_force_n: float = 80.0,
        opposite_support_start_n: float = 30.0,
        opposite_support_full_n: float = 100.0,
        insertion_clearance_fraction: float = 0.70,
        touchdown_force_n: float = 30.0,
        touchdown_speed_mps: float = 0.20,
        touchdown_height_tolerance_m: float = 0.025,
        diagnostic_contact_threshold_n: float = 1.0,
    ) -> torch.Tensor:
        del asset_cfg, contact_sensor_cfg
        self.runtime.update(
            env,
            activation_distance_m=activation_distance_m,
            clearance_margin_m=clearance_margin_m,
            tread_margin_m=tread_margin_m,
            insertion_target_m=insertion_target_m,
            unload_full_force_n=unload_full_force_n,
            unload_zero_force_n=unload_zero_force_n,
            opposite_support_start_n=opposite_support_start_n,
            opposite_support_full_n=opposite_support_full_n,
            insertion_clearance_fraction=insertion_clearance_fraction,
            touchdown_force_n=touchdown_force_n,
            touchdown_speed_mps=touchdown_speed_mps,
            touchdown_height_tolerance_m=touchdown_height_tolerance_m,
            diagnostic_contact_threshold_n=diagnostic_contact_threshold_n,
        )
        return self.runtime.clearance_reward


class forward_insertion(ManagerTermBase):
    """Bounded non-recyclable safe-tread insertion progress."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        self.runtime.reset(env_ids)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
            "contact_forces"
        ),
        activation_distance_m: float = 0.30,
        clearance_margin_m: float = 0.015,
        tread_margin_m: float = 0.015,
        insertion_target_m: float = 0.040,
        unload_full_force_n: float = 15.0,
        unload_zero_force_n: float = 80.0,
        opposite_support_start_n: float = 30.0,
        opposite_support_full_n: float = 100.0,
        insertion_clearance_fraction: float = 0.70,
        touchdown_force_n: float = 30.0,
        touchdown_speed_mps: float = 0.20,
        touchdown_height_tolerance_m: float = 0.025,
        diagnostic_contact_threshold_n: float = 1.0,
    ) -> torch.Tensor:
        del asset_cfg, contact_sensor_cfg
        self.runtime.update(
            env,
            activation_distance_m=activation_distance_m,
            clearance_margin_m=clearance_margin_m,
            tread_margin_m=tread_margin_m,
            insertion_target_m=insertion_target_m,
            unload_full_force_n=unload_full_force_n,
            unload_zero_force_n=unload_zero_force_n,
            opposite_support_start_n=opposite_support_start_n,
            opposite_support_full_n=opposite_support_full_n,
            insertion_clearance_fraction=insertion_clearance_fraction,
            touchdown_force_n=touchdown_force_n,
            touchdown_speed_mps=touchdown_speed_mps,
            touchdown_height_tolerance_m=touchdown_height_tolerance_m,
            diagnostic_contact_threshold_n=diagnostic_contact_threshold_n,
        )
        return self.runtime.insertion_reward
