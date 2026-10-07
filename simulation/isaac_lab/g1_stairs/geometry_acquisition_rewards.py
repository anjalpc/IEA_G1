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
from .reward_math import normalized_excess_forward_velocity
from .terrain_state import riser_height_per_env, stair_cfg
from .reward_math import (
    alternating_foot_gate,
    anticipatory_clearance_target,
    bounded_impact_severity,
    bounded_vertical_progress_credit,
    first_touchdown_placement_score,
    global_stair_target_update,
    nonpreferred_acquisition_event,
    preferred_transition_contact_update,
    step_over_transition_update,
    escrowed_transition_reward,
    update_alternating_preferred_foot,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

KNEE_BODY_NAMES = (
    "left_knee_link",
    "right_knee_link",
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

        self.support_credit_dwell = float(
            getattr(env.cfg, "verified_insertion_dwell_s", 0.0)
        )
        shape = (env.num_envs, 2)
        self.support_elapsed = torch.zeros(shape, device=env.device)

        # Optional first-loaded-contact placement latch. When enabled, a poor
        # initial tread landing cannot become a valid touchdown merely by
        # shuffling forward while continuously supported.
        self.require_first_touchdown_placement = bool(
            getattr(env.cfg, "require_first_touchdown_placement", False)
        )
        self.first_touchdown_shaping_start_margin_m = float(
            getattr(
                env.cfg,
                "first_touchdown_shaping_start_margin_m",
                -0.150,
            )
        )
        self.global_stair_acquisition = bool(
            getattr(env.cfg, "global_stair_acquisition", False)
        )
        self.dynamic_alternating_rewards = bool(
            getattr(env.cfg, "dynamic_alternating_rewards", False)
        )
        if (
            self.dynamic_alternating_rewards
            and not self.global_stair_acquisition
        ):
            raise ValueError(
                "dynamic_alternating_rewards requires "
                "global_stair_acquisition"
            )
        self.preferred_foot = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )
        self.touchdown_attempt_active = torch.zeros(
            shape, dtype=torch.bool, device=env.device
        )
        self.touchdown_attempt_good = torch.zeros(
            shape, dtype=torch.bool, device=env.device
        )
        self.touchdown_placement_scored = torch.zeros(
            shape, dtype=torch.bool, device=env.device
        )

        # Stair target state. In historical mode each foot advances
        # independently. In global mode both columns remain synchronized.
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

        # Clearance progress remains provisional until the complete sole
        # has safely crossed the current target riser.
        self.pending_clearance = torch.zeros_like(self.best_clearance)

        # Previous forward foot velocity is used only to reject static/support
        # contact when identifying a forward riser-face impact.
        self.previous_foot_vx = torch.zeros_like(self.best_clearance)
        self.previous_foot_vx_initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

        self.clearance_reward = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self.insertion_reward = torch.zeros_like(self.clearance_reward)
        self.first_touchdown_placement_reward = torch.zeros_like(
            self.clearance_reward
        )
        self.nonpreferred_acquisition_reward = torch.zeros_like(
            self.clearance_reward
        )

        # Per-step physical events are preserved before global target state is
        # advanced so downstream transition rewards can reason about the exact
        # stair/contact that occurred on this simulation step.
        self.physical_touchdown_event = torch.zeros(
            shape, dtype=torch.bool, device=env.device
        )
        self.step_acquired_event = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.acquired_target_index = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.sample_target_index = torch.full_like(
            self.acquired_target_index, -1
        )
        self.riser_face_strike_sample = torch.zeros(
            shape, dtype=torch.bool, device=env.device
        )
        self.riser_strike_force_x = torch.zeros_like(self.best_clearance)

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
        self.preferred_foot[env_ids] = -1
        self.support_elapsed[env_ids] = 0.0
        self.touchdown_attempt_active[env_ids] = False
        self.touchdown_attempt_good[env_ids] = False
        self.touchdown_placement_scored[env_ids] = False
        self.best_clearance[env_ids] = 0.0
        self.best_insertion[env_ids] = 0.0
        self.pending_clearance[env_ids] = 0.0
        self.previous_foot_vx[env_ids] = 0.0
        self.previous_foot_vx_initialized[env_ids] = False
        self.first_touchdown_placement_reward[env_ids] = 0.0
        self.nonpreferred_acquisition_reward[env_ids] = 0.0
        self.physical_touchdown_event[env_ids] = False
        self.step_acquired_event[env_ids] = False
        self.acquired_target_index[env_ids] = -1
        self.sample_target_index[env_ids] = -1
        self.riser_face_strike_sample[env_ids] = False
        self.riser_strike_force_x[env_ids] = 0.0

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
        suppress_riser_face_clearance_credit: bool = False,
        escrow_clearance_until_riser_crossing: bool = False,
        riser_face_band_m: float = 0.025,
        riser_below_top_margin_m: float = 0.005,
        riser_minimum_horizontal_force_n: float = 30.0,
        riser_minimum_previous_forward_speed_mps: float = 0.10,
    ) -> None:
        step = int(env.common_step_counter)
        if self.last_step == step:
            return
        self.last_step = step
        self.nonpreferred_acquisition_reward.zero_()
        self.physical_touchdown_event.zero_()
        self.step_acquired_event.zero_()
        self.acquired_target_index.fill_(-1)
        self.riser_face_strike_sample.zero_()
        self.riser_strike_force_x.zero_()
        self.sample_target_index.copy_(self.target_step[:, 0])

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
        if riser_face_band_m <= 0.0:
            raise ValueError("riser_face_band_m must be positive")
        if riser_below_top_margin_m < 0.0:
            raise ValueError("riser_below_top_margin_m cannot be negative")
        if riser_minimum_horizontal_force_n <= 0.0:
            raise ValueError(
                "riser_minimum_horizontal_force_n must be positive"
            )
        if riser_minimum_previous_forward_speed_mps < 0.0:
            raise ValueError(
                "riser_minimum_previous_forward_speed_mps cannot be negative"
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

        if hasattr(data, "body_lin_vel_w"):
            current_vx = data.body_lin_vel_w[
                :,
                self.robot_foot_ids,
                0,
            ]
        else:
            current_vx = torch.zeros_like(sole_bottom)

        previous_vx = torch.where(
            self.previous_foot_vx_initialized[:, None],
            self.previous_foot_vx,
            current_vx,
        )

        force_x = self.sensor.data.net_forces_w[
            :,
            self.sensor_foot_ids,
            0,
        ]

        near_riser_face = (
            (front_x >= next_riser_x - float(riser_face_band_m))
            & (rear_x <= next_riser_x + float(riser_face_band_m))
        )

        below_riser_top = (
            sole_bottom
            <= target_tread_height - float(riser_below_top_margin_m)
        )

        riser_face_strike_sample = (
            valid_target
            & near_riser_face
            & below_riser_top
            & (
                force_x
                <= -float(riser_minimum_horizontal_force_n)
            )
            & (
                previous_vx
                >= float(riser_minimum_previous_forward_speed_mps)
            )
        )
        self.riser_face_strike_sample.copy_(riser_face_strike_sample)
        self.riser_strike_force_x.copy_(force_x)

        self.previous_foot_vx.copy_(current_vx)
        self.previous_foot_vx_initialized[:] = True

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

        if self.dynamic_alternating_rewards:
            preferred_gate = alternating_foot_gate(
                self.preferred_foot
            )
            swing_gate = (
                swing_gate
                * preferred_gate.to(swing_gate.dtype)
            )
        else:
            preferred_gate = torch.ones_like(
                valid_target,
                dtype=torch.bool,
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

        if suppress_riser_face_clearance_credit:
            clearance_delta = torch.where(
                riser_face_strike_sample,
                torch.zeros_like(clearance_delta),
                clearance_delta,
            )

        if escrow_clearance_until_riser_crossing:
            # Accumulate bounded monotonic progress without paying it yet.
            self.pending_clearance.add_(clearance_delta)

            # A target-riser strike invalidates all provisional clearance
            # accumulated for this foot/current target.
            self.pending_clearance.copy_(
                torch.where(
                    riser_face_strike_sample,
                    torch.zeros_like(self.pending_clearance),
                    self.pending_clearance,
                )
            )

            # Keep provisional clearance credit withheld. It is released
            # only after verified touchdown/support on the current target
            # tread below.
            clearance_delta = torch.zeros_like(clearance_delta)

        # Advance the potential even on a strike so credit cannot be paid later.
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

        self.insertion_reward.copy_(
            insertion_delta.sum(dim=1) / dt
        )

        self.first_touchdown_placement_reward.zero_()

        # Current complete-sole placement inside the safe tread.
        # The final tread transitions directly onto the platform, so its
        # forward edge does not need clearance from another riser.
        final_target = target_index == (step_count - 1)
        placement_good = (
            (rear_x >= safe_start)
            & (
                (front_x <= safe_end)
                | final_target
            )
        )

        if self.require_first_touchdown_placement:
            # A landing attempt begins at the first loaded top-surface contact,
            # before the speed gate can delay recognition until after a shuffle.
            # Broad tread overlap intentionally includes poor rear-edge landings.
            top_surface_contact = (
                valid_target
                & (front_x >= next_riser_x)
                & (rear_x <= next_riser_x + tread)
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
            )

            # Do not allow a brief unloading during a shuffle to create a new
            # attempt. A new attempt requires clear physical lift from the tread.
            lifted_off = (
                self.touchdown_attempt_active
                & (
                    vertical_force
                    <= float(unload_full_force_n)
                )
                & (
                    sole_bottom
                    > (
                        target_tread_height
                        + float(touchdown_height_tolerance_m)
                    )
                )
            )

            self.touchdown_attempt_active.copy_(
                self.touchdown_attempt_active & ~lifted_off
            )
            self.touchdown_attempt_good.copy_(
                self.touchdown_attempt_good & ~lifted_off
            )

            new_attempt = (
                top_surface_contact
                & ~self.touchdown_attempt_active
            )

            rear_margin = rear_x - next_riser_x
            front_margin = next_riser_x + tread - front_x

            placement_score = first_touchdown_placement_score(
                rear_margin,
                front_margin,
                shaping_start_margin_m=(
                    self.first_touchdown_shaping_start_margin_m
                ),
                full_rear_margin_m=(
                    float(tread_margin_m)
                    + float(insertion_target_m)
                ),
                front_safe_margin_m=float(tread_margin_m),
            )

            placement_reward_event = (
                new_attempt
                & ~self.touchdown_placement_scored
                & preferred_gate
            )

            self.first_touchdown_placement_reward.copy_(
                (
                    placement_score
                    * placement_reward_event.to(
                        placement_score.dtype
                    )
                ).sum(dim=1)
                / dt
            )

            self.touchdown_placement_scored.copy_(
                self.touchdown_placement_scored
                | placement_reward_event
            )

            self.touchdown_attempt_good.copy_(
                torch.where(
                    new_attempt,
                    placement_good,
                    self.touchdown_attempt_good,
                )
            )
            self.touchdown_attempt_active.copy_(
                self.touchdown_attempt_active | new_attempt
            )

            first_placement_gate = (
                self.touchdown_attempt_active
                & self.touchdown_attempt_good
            )
        else:
            first_placement_gate = torch.ones_like(valid_target)

        # Physical touchdown controls stair-target progression.
        # First-placement quality controls reward eligibility separately.
        physical_touchdown = (
            valid_target
            & placement_good
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

        if self.support_credit_dwell > 0:
            from .support_credit import sustained_support
            elapsed, physical_touchdown = sustained_support(
                self.support_elapsed,
                physical_touchdown,
                dt,
                self.support_credit_dwell,
            )
            self.support_elapsed.copy_(
                torch.where(
                    physical_touchdown,
                    torch.zeros_like(elapsed),
                    elapsed,
                )
            )

        self.physical_touchdown_event.copy_(physical_touchdown)

        rewarded_touchdown = (
            physical_touchdown
            & first_placement_gate
            & preferred_gate
        )

        if self.dynamic_alternating_rewards:
            nonpreferred_event = nonpreferred_acquisition_event(
                self.preferred_foot,
                physical_touchdown,
            )
            self.nonpreferred_acquisition_reward.copy_(
                nonpreferred_event.to(self.clearance_reward.dtype) / dt
            )

        if self.support_credit_dwell > 0:
            insertion_delta = rewarded_touchdown.to(
                self.insertion_reward.dtype
            )
            self.insertion_reward.copy_(
                insertion_delta.sum(dim=1) / dt
            )

        if escrow_clearance_until_riser_crossing:
            # Verified touchdown is the acquisition event. A foot that only
            # crosses the riser geometrically cannot cash provisional credit.
            clearance_delta = torch.where(
                rewarded_touchdown,
                self.pending_clearance,
                torch.zeros_like(clearance_delta),
            )

            self.pending_clearance.copy_(
                torch.where(
                    physical_touchdown,
                    torch.zeros_like(self.pending_clearance),
                    self.pending_clearance,
                )
            )

        self.clearance_reward.copy_(
            clearance_delta.sum(dim=1) / dt
        )

        if bool(physical_touchdown.any()):
            if self.global_stair_acquisition:
                (
                    old_target,
                    step_acquired,
                    next_target,
                ) = global_stair_target_update(
                    self.target_step,
                    physical_touchdown,
                    step_count,
                )

                self.step_acquired_event.copy_(step_acquired)
                self.acquired_target_index.copy_(
                    torch.where(
                        step_acquired,
                        old_target,
                        torch.full_like(old_target, -1),
                    )
                )

                if self.dynamic_alternating_rewards:
                    self.preferred_foot.copy_(
                        update_alternating_preferred_foot(
                            self.preferred_foot,
                            physical_touchdown,
                            step_acquired,
                        )
                    )

                # One stair acquisition per environment, even if both feet
                # satisfy touchdown on the same simulation step.
                for stair_index in range(step_count):
                    self.touchdowns[stair_index] += torch.count_nonzero(
                        step_acquired
                        & (old_target == stair_index)
                    )

                self.target_step.copy_(next_target)

                # A new stair becomes the target for BOTH feet.
                reset_mask = step_acquired[:, None].expand_as(
                    self.target_step
                )
            else:
                # Historical per-foot progression retained for reproducibility.
                old_target = self.target_step.clone()

                for stair_index in range(step_count):
                    self.touchdowns[stair_index] += torch.count_nonzero(
                        physical_touchdown
                        & (old_target == stair_index)
                    )

                self.target_step.copy_(
                    torch.where(
                        physical_touchdown,
                        self.target_step + 1,
                        self.target_step,
                    )
                )

                reset_mask = physical_touchdown

            self.touchdown_attempt_active.copy_(
                torch.where(
                    reset_mask,
                    torch.zeros_like(self.touchdown_attempt_active),
                    self.touchdown_attempt_active,
                )
            )
            self.touchdown_attempt_good.copy_(
                torch.where(
                    reset_mask,
                    torch.zeros_like(self.touchdown_attempt_good),
                    self.touchdown_attempt_good,
                )
            )
            self.touchdown_placement_scored.copy_(
                torch.where(
                    reset_mask,
                    torch.zeros_like(self.touchdown_placement_scored),
                    self.touchdown_placement_scored,
                )
            )
            self.support_elapsed.copy_(
                torch.where(
                    reset_mask,
                    torch.zeros_like(self.support_elapsed),
                    self.support_elapsed,
                )
            )

            # New stair = new non-recyclable potential.
            self.best_clearance.copy_(
                torch.where(
                    reset_mask,
                    torch.zeros_like(self.best_clearance),
                    self.best_clearance,
                )
            )
            self.best_insertion.copy_(
                torch.where(
                    reset_mask,
                    torch.zeros_like(self.best_insertion),
                    self.best_insertion,
                )
            )
            self.pending_clearance.copy_(
                torch.where(
                    reset_mask,
                    torch.zeros_like(self.pending_clearance),
                    self.pending_clearance,
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
    suppress_riser_face_clearance_credit=False,
    escrow_clearance_until_riser_crossing=False,
    riser_face_band_m=0.025,
    riser_below_top_margin_m=0.005,
    riser_minimum_horizontal_force_n=30.0,
    riser_minimum_previous_forward_speed_mps=0.10,
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
        suppress_riser_face_clearance_credit: bool = False,
        escrow_clearance_until_riser_crossing: bool = False,
        riser_face_band_m: float = 0.025,
        riser_below_top_margin_m: float = 0.005,
        riser_minimum_horizontal_force_n: float = 30.0,
        riser_minimum_previous_forward_speed_mps: float = 0.10,
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
            suppress_riser_face_clearance_credit=suppress_riser_face_clearance_credit,
            escrow_clearance_until_riser_crossing=escrow_clearance_until_riser_crossing,
            riser_face_band_m=riser_face_band_m,
            riser_below_top_margin_m=riser_below_top_margin_m,
            riser_minimum_horizontal_force_n=riser_minimum_horizontal_force_n,
            riser_minimum_previous_forward_speed_mps=riser_minimum_previous_forward_speed_mps,
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
        suppress_riser_face_clearance_credit: bool = False,
        escrow_clearance_until_riser_crossing: bool = False,
        riser_face_band_m: float = 0.025,
        riser_below_top_margin_m: float = 0.005,
        riser_minimum_horizontal_force_n: float = 30.0,
        riser_minimum_previous_forward_speed_mps: float = 0.10,
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
            suppress_riser_face_clearance_credit=suppress_riser_face_clearance_credit,
            escrow_clearance_until_riser_crossing=escrow_clearance_until_riser_crossing,
            riser_face_band_m=riser_face_band_m,
            riser_below_top_margin_m=riser_below_top_margin_m,
            riser_minimum_horizontal_force_n=riser_minimum_horizontal_force_n,
            riser_minimum_previous_forward_speed_mps=riser_minimum_previous_forward_speed_mps,
        )
        return self.runtime.insertion_reward


class first_touchdown_placement(ManagerTermBase):
    """One-shot graded reward at first loaded top-surface contact."""

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
    ) -> torch.Tensor:
        if self.runtime.last_step != int(env.common_step_counter):
            raise RuntimeError(
                "first_touchdown_placement must run after "
                "terrain_relative_clearance/forward_insertion"
            )
        return self.runtime.first_touchdown_placement_reward

class anticipatory_clearance_progress(ManagerTermBase):
    """Reward only bounded vertical progress before reaching the target riser.

    Horizontal foot motion only selects the local pre-riser window; it cannot
    create reward. Credit is capped by vertical speed and stops at the smooth
    geometry-relative clearance target, preventing giant-stride/high-kick
    exploitation while providing earlier swing timing than unload-gated credit.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)
        shape = (env.num_envs, 2)
        self._previous_height = torch.zeros(
            shape, dtype=torch.float32, device=env.device
        )
        self._previous_target = torch.full(
            shape, -1, dtype=torch.long, device=env.device
        )
        self._initialized = torch.zeros(
            shape, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_height[env_ids] = 0.0
        self._previous_target[env_ids] = -1
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        start_distance_m: float = 0.22,
        full_clearance_distance_m: float = 0.05,
        clearance_margin_m: float = 0.015,
        support_advantage_start_n: float = 0.0,
        support_advantage_full_n: float = 60.0,
        max_credit_vertical_speed_mps: float = 0.60,
    ) -> torch.Tensor:
        if self.runtime.last_step != int(env.common_step_counter):
            raise RuntimeError(
                "anticipatory_clearance_progress must run after "
                "terrain_relative_clearance/forward_insertion"
            )
        if clearance_margin_m < 0.0:
            raise ValueError("clearance_margin_m cannot be negative")
        if not support_advantage_full_n > support_advantage_start_n:
            raise ValueError(
                "support_advantage_full_n must exceed "
                "support_advantage_start_n"
            )

        runtime = self.runtime
        cfg = stair_cfg(env)
        tread = float(cfg.tread_depth)
        first_riser = float(cfg.start_clearance)
        step_count = int(cfg.step_count)
        riser_height = riser_height_per_env(env)
        data = runtime.robot.data

        foot_pos = (
            data.body_pos_w[:, runtime.robot_foot_ids]
            - env.scene.env_origins[:, None, :]
        )
        foot_quat = data.body_quat_w[:, runtime.robot_foot_ids]
        offsets = runtime.sole_centres.to(dtype=foot_pos.dtype)[
            None, None
        ].expand(env.num_envs, 2, -1, -1)
        quaternions = foot_quat.unsqueeze(2).expand(
            -1, -1, offsets.shape[2], -1
        )
        sole_points = (
            foot_pos.unsqueeze(2)
            + _quat_rotate_wxyz(quaternions, offsets)
        )
        radius = float(G1_SOLE_SPHERE_RADIUS_M)
        sole_bottom = (sole_points[..., 2] - radius).amin(dim=2)
        front_x = sole_points[..., 0].amax(dim=2) + radius

        target_index = runtime.target_step.clamp(
            min=0, max=step_count - 1
        )
        valid_target = (
            (runtime.target_step < step_count)
            & (riser_height[:, None] > 1.0e-8)
        )
        target_riser_x = (
            first_riser + target_index.to(foot_pos.dtype) * tread
        )
        support_height = (
            target_index.to(foot_pos.dtype) * riser_height[:, None]
        )
        relative_height = (sole_bottom - support_height).clamp_min(0.0)
        full_clearance = (
            riser_height[:, None]
            + float(clearance_margin_m)
        ).expand_as(relative_height).clamp_min(1.0e-6)

        distance_to_edge = target_riser_x - front_x
        target_height, in_window = anticipatory_clearance_target(
            distance_to_edge,
            full_clearance,
            start_distance_m=float(start_distance_m),
            full_distance_m=float(full_clearance_distance_m),
        )

        target_changed = runtime.target_step != self._previous_target
        previous_height = torch.where(
            self._initialized & ~target_changed,
            self._previous_height,
            relative_height,
        )
        credit = bounded_vertical_progress_credit(
            relative_height,
            previous_height,
            target_height,
            full_clearance,
            max_credit_vertical_speed_mps=(
                float(max_credit_vertical_speed_mps)
            ),
            dt_s=float(env.step_dt),
        )

        vertical_force = runtime.sensor.data.net_forces_w[
            :, runtime.sensor_foot_ids, 2
        ].clamp_min(0.0)
        opposite_force = torch.stack(
            (vertical_force[:, 1], vertical_force[:, 0]), dim=1
        )
        support_advantage = opposite_force - vertical_force
        transfer_gate = _smooth_rising(
            support_advantage,
            float(support_advantage_start_n),
            float(support_advantage_full_n),
        )

        if runtime.dynamic_alternating_rewards:
            preferred_gate = alternating_foot_gate(runtime.preferred_foot)
        else:
            preferred_gate = torch.ones_like(valid_target, dtype=torch.bool)

        credit = (
            credit
            * transfer_gate
            * in_window.to(credit.dtype)
            * valid_target.to(credit.dtype)
            * preferred_gate.to(credit.dtype)
        )

        self._previous_height.copy_(relative_height)
        self._previous_target.copy_(runtime.target_step)
        self._initialized[:] = True

        # RewardManager multiplies by env.step_dt. Normalized vertical progress
        # therefore integrates to bounded per-target credit.
        return credit.sum(dim=1) / float(env.step_dt)


class nonpreferred_stair_acquisition(ManagerTermBase):
    """One-shot event when the nonpreferred foot physically acquires a stair."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        del env_ids

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        if self.runtime.last_step != int(env.common_step_counter):
            raise RuntimeError(
                "nonpreferred_stair_acquisition must run after "
                "terrain_relative_clearance/forward_insertion"
            )
        return self.runtime.nonpreferred_acquisition_reward


class preferred_old_tread_replant(ManagerTermBase):
    """Emit one event when the preferred foot lifts then replants old tread.

    The event is armed only after clear unloading/lift following an acquisition.
    Static support is never penalized. A direct loaded contact on the next tread
    resolves the transition without penalty.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)
        self._armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._resolved = torch.zeros_like(self._armed)
        self._target = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self._preferred = torch.full_like(self._target, -1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._armed[env_ids] = False
        self._resolved[env_ids] = False
        self._target[env_ids] = -1
        self._preferred[env_ids] = -1

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        unload_force_n: float = 15.0,
        touchdown_force_n: float = 30.0,
        lift_clearance_m: float = 0.030,
        top_height_tolerance_m: float = 0.025,
    ) -> torch.Tensor:
        if self.runtime.last_step != int(env.common_step_counter):
            raise RuntimeError(
                "preferred_old_tread_replant must run after "
                "terrain_relative_clearance/forward_insertion"
            )
        if unload_force_n < 0.0:
            raise ValueError("unload_force_n cannot be negative")
        if touchdown_force_n <= unload_force_n:
            raise ValueError("touchdown_force_n must exceed unload_force_n")
        if lift_clearance_m <= 0.0:
            raise ValueError("lift_clearance_m must be positive")
        if top_height_tolerance_m <= 0.0:
            raise ValueError("top_height_tolerance_m must be positive")

        runtime = self.runtime
        cfg = stair_cfg(env)
        tread = float(cfg.tread_depth)
        first_riser = float(cfg.start_clearance)
        step_count = int(cfg.step_count)
        riser_height = riser_height_per_env(env)

        current_target = runtime.target_step[:, 0]
        preferred = runtime.preferred_foot
        valid_transition = (
            runtime.dynamic_alternating_rewards
            & (preferred >= 0)
            & (current_target > 0)
            & (current_target < step_count)
        )

        changed = (
            (current_target != self._target)
            | (preferred != self._preferred)
        )
        self._armed.copy_(self._armed & ~changed & valid_transition)
        self._resolved.copy_(self._resolved & ~changed & valid_transition)
        self._target.copy_(current_target)
        self._preferred.copy_(preferred)

        data = runtime.robot.data
        foot_pos = (
            data.body_pos_w[:, runtime.robot_foot_ids]
            - env.scene.env_origins[:, None, :]
        )
        foot_quat = data.body_quat_w[:, runtime.robot_foot_ids]
        offsets = runtime.sole_centres.to(dtype=foot_pos.dtype)[
            None, None
        ].expand(env.num_envs, 2, -1, -1)
        quaternions = foot_quat.unsqueeze(2).expand(
            -1, -1, offsets.shape[2], -1
        )
        sole_points = (
            foot_pos.unsqueeze(2)
            + _quat_rotate_wxyz(quaternions, offsets)
        )
        radius = float(G1_SOLE_SPHERE_RADIUS_M)
        sole_bottom_points = sole_points[..., 2] - radius
        rear_x = sole_points[..., 0].amin(dim=2) - radius
        front_x = sole_points[..., 0].amax(dim=2) + radius
        vertical_force = runtime.sensor.data.net_forces_w[
            :, runtime.sensor_foot_ids, 2
        ].clamp_min(0.0)

        pref_idx = preferred.clamp(0, 1)
        gather1 = pref_idx[:, None]
        pref_rear = rear_x.gather(1, gather1).squeeze(1)
        pref_front = front_x.gather(1, gather1).squeeze(1)
        pref_force = vertical_force.gather(1, gather1).squeeze(1)
        pref_bottom = sole_bottom_points.amin(dim=2).gather(
            1, gather1
        ).squeeze(1)

        target_float = current_target.to(foot_pos.dtype)
        old_start = first_riser + (target_float - 1.0) * tread
        old_end = first_riser + target_float * tread
        next_start = old_end
        next_end = next_start + tread
        old_top = target_float * riser_height
        next_top = (target_float + 1.0) * riser_height

        clear_lift = (
            valid_transition
            & (pref_force <= float(unload_force_n))
            & (pref_bottom >= old_top + float(lift_clearance_m))
        )

        loaded = pref_force >= float(touchdown_force_n)
        # Any loaded sole-bottom contact on the old tread counts as a replant;
        # requiring the entire sole to be flat would leave a pitched-toe/heel
        # loophole. Direct-next gets priority only when the loaded bottom is at
        # the next tread height.
        old_height = (
            (pref_bottom - old_top).abs()
            <= float(top_height_tolerance_m)
        )
        next_height = (
            (pref_bottom - next_top).abs()
            <= float(top_height_tolerance_m)
        )
        old_contact = (
            loaded
            & old_height
            & (pref_front >= old_start)
            & (pref_rear <= old_end)
        )
        next_contact = (
            loaded
            & next_height
            & (pref_front >= next_start)
            & (pref_rear <= next_end)
        )

        armed, resolved, replant, _direct = (
            preferred_transition_contact_update(
                self._armed,
                self._resolved,
                clear_lift,
                old_contact & valid_transition,
                next_contact & valid_transition,
            )
        )
        self._armed.copy_(armed)
        self._resolved.copy_(resolved)

        return replant.to(dtype=foot_pos.dtype) / float(env.step_dt)



class step_over_transition_contract(ManagerTermBase):
    """Reward only complete clean step-over transitions.

    Stair 1 is free-choice: the first clearly lifting foot inside a local
    pre-riser window becomes committed. After every verified stair acquisition,
    the opposite foot becomes committed automatically through the existing
    dynamic-alternation state. A transition succeeds only if the committed foot
    lifts, clears the target riser, makes its first loaded target-surface
    contact on the next tread, and then physically acquires that tread.

    Replant, riser strike, excessive airborne forward excursion, wrong-foot
    acquisition, or acquisition without the required sequence resolves that
    transition as a one-shot failure. Physical acquisition itself is never
    blocked or rewritten.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)
        n = env.num_envs
        device = env.device

        self._active = torch.zeros(n, dtype=torch.bool, device=device)
        self._target = torch.full(
            (n,), -1, dtype=torch.long, device=device
        )
        self._preferred = torch.full_like(self._target, -1)
        self._lifted = torch.zeros_like(self._active)
        self._cleared = torch.zeros_like(self._active)
        self._next_contact_seen = torch.zeros_like(self._active)
        self._resolved = torch.zeros_like(self._active)
        self._lift_front_x = torch.zeros(
            n, dtype=torch.float32, device=device
        )
        self._lift_front_valid = torch.zeros_like(self._active)
        self._milestone_credit_paid = torch.zeros(
            n, dtype=torch.float32, device=device
        )
        self.last_event_code = torch.zeros_like(self._target)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._active[env_ids] = False
        self._target[env_ids] = -1
        self._preferred[env_ids] = -1
        self._lifted[env_ids] = False
        self._cleared[env_ids] = False
        self._next_contact_seen[env_ids] = False
        self._resolved[env_ids] = False
        self._lift_front_x[env_ids] = 0.0
        self._lift_front_valid[env_ids] = False
        self._milestone_credit_paid[env_ids] = 0.0
        self.last_event_code[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        first_commit_distance_m: float = 0.30,
        unload_force_n: float = 15.0,
        opposite_support_force_n: float = 30.0,
        touchdown_force_n: float = 30.0,
        lift_clearance_m: float = 0.030,
        riser_clearance_margin_m: float = 0.015,
        clearance_window_m: float = 0.050,
        top_height_tolerance_m: float = 0.025,
        max_forward_excursion_m: float = 0.40,
        strike_minimum_force_n: float = 30.0,
        strike_full_severity_force_n: float = 600.0,
        strike_minimum_event_fraction: float = 0.10,
        success_value: float = 1.0,
        failure_value: float = -1.0,
        lift_milestone_value: float = 0.20,
        clearance_milestone_value: float = 0.30,
        strike_failure_base: float = -1.0,
        strike_failure_severity_scale: float = -0.5,
    ) -> torch.Tensor:
        if self.runtime.last_step != int(env.common_step_counter):
            raise RuntimeError(
                "step_over_transition_contract must run after "
                "terrain_relative_clearance/forward_insertion"
            )
        if not self.runtime.global_stair_acquisition:
            raise RuntimeError(
                "step_over_transition_contract requires global acquisition"
            )
        if not self.runtime.dynamic_alternating_rewards:
            raise RuntimeError(
                "step_over_transition_contract requires dynamic alternation"
            )
        if first_commit_distance_m <= 0.0:
            raise ValueError("first_commit_distance_m must be positive")
        if unload_force_n < 0.0:
            raise ValueError("unload_force_n cannot be negative")
        if opposite_support_force_n <= unload_force_n:
            raise ValueError(
                "opposite_support_force_n must exceed unload_force_n"
            )
        if touchdown_force_n <= unload_force_n:
            raise ValueError("touchdown_force_n must exceed unload_force_n")
        if lift_clearance_m <= 0.0:
            raise ValueError("lift_clearance_m must be positive")
        if riser_clearance_margin_m < 0.0:
            raise ValueError("riser_clearance_margin_m cannot be negative")
        if clearance_window_m <= 0.0:
            raise ValueError("clearance_window_m must be positive")
        if top_height_tolerance_m <= 0.0:
            raise ValueError("top_height_tolerance_m must be positive")
        if max_forward_excursion_m <= 0.0:
            raise ValueError("max_forward_excursion_m must be positive")

        runtime = self.runtime
        cfg = stair_cfg(env)
        tread = float(cfg.tread_depth)
        first_riser = float(cfg.start_clearance)
        step_count = int(cfg.step_count)
        riser_height = riser_height_per_env(env)
        current_target = runtime.target_step[:, 0]

        # Start the first-riser contract without prescribing left or right.
        first_init = (
            ~self._active
            & (self._target < 0)
            & (current_target == 0)
            & ~runtime.step_acquired_event
        )
        self._active.copy_(self._active | first_init)
        self._target.copy_(
            torch.where(first_init, torch.zeros_like(self._target), self._target)
        )

        data = runtime.robot.data
        foot_pos = (
            data.body_pos_w[:, runtime.robot_foot_ids]
            - env.scene.env_origins[:, None, :]
        )
        foot_quat = data.body_quat_w[:, runtime.robot_foot_ids]
        offsets = runtime.sole_centres.to(dtype=foot_pos.dtype)[
            None, None
        ].expand(env.num_envs, 2, -1, -1)
        quaternions = foot_quat.unsqueeze(2).expand(
            -1, -1, offsets.shape[2], -1
        )
        sole_points = (
            foot_pos.unsqueeze(2)
            + _quat_rotate_wxyz(quaternions, offsets)
        )
        radius = float(G1_SOLE_SPHERE_RADIUS_M)
        sole_bottom_points = sole_points[..., 2] - radius
        sole_bottom = sole_bottom_points.amin(dim=2)
        rear_x = sole_points[..., 0].amin(dim=2) - radius
        front_x = sole_points[..., 0].amax(dim=2) + radius
        vertical_force = runtime.sensor.data.net_forces_w[
            :, runtime.sensor_foot_ids, 2
        ].clamp_min(0.0)

        target = self._target.clamp(min=0, max=step_count - 1)
        target_f = target.to(foot_pos.dtype)
        old_top = target_f * riser_height
        next_top = (target_f + 1.0) * riser_height
        next_start = first_riser + target_f * tread
        next_end = next_start + tread
        historical_old_start = first_riser + (target_f - 1.0) * tread
        old_start = torch.where(
            target == 0,
            next_start - float(first_commit_distance_m),
            historical_old_start,
        )
        old_end = next_start

        # For stair 1 only, commit whichever foot naturally initiates a local
        # supported lift. No reward is emitted for commitment itself.
        open_first = (
            self._active
            & ~self._resolved
            & (self._target == 0)
            & (self._preferred < 0)
        )
        distance_to_first = next_start[:, None] - front_x
        opposite_force = torch.stack(
            (vertical_force[:, 1], vertical_force[:, 0]), dim=1
        )
        first_candidate = (
            open_first[:, None]
            & (distance_to_first >= 0.0)
            & (distance_to_first <= float(first_commit_distance_m))
            & (vertical_force <= float(unload_force_n))
            & (opposite_force >= float(opposite_support_force_n))
            & (
                sole_bottom
                >= old_top[:, None] + float(lift_clearance_m)
            )
        )
        candidate_score = torch.where(
            first_candidate,
            sole_bottom - old_top[:, None],
            torch.full_like(sole_bottom, -1.0),
        )
        has_candidate = first_candidate.any(dim=1)
        chosen = candidate_score.argmax(dim=1)
        commit_now = open_first & has_candidate
        self._preferred.copy_(
            torch.where(commit_now, chosen, self._preferred)
        )

        preferred_idx = self._preferred.clamp(0, 1)
        gather1 = preferred_idx[:, None]
        committed = self._preferred >= 0
        pref_rear = rear_x.gather(1, gather1).squeeze(1)
        pref_front = front_x.gather(1, gather1).squeeze(1)
        pref_force = vertical_force.gather(1, gather1).squeeze(1)
        pref_bottom = sole_bottom.gather(1, gather1).squeeze(1)

        clear_lift = (
            self._active
            & committed
            & ~self._resolved
            & (pref_force <= float(unload_force_n))
            & (pref_bottom >= old_top + float(lift_clearance_m))
        )
        new_lift = clear_lift & ~self._lifted
        self._lift_front_x.copy_(
            torch.where(new_lift, pref_front, self._lift_front_x)
        )
        self._lift_front_valid.copy_(
            self._lift_front_valid | new_lift
        )

        clear_riser = (
            self._active
            & committed
            & (pref_bottom >= next_top + float(riser_clearance_margin_m))
            & (pref_front >= next_start - float(clearance_window_m))
            & (pref_rear <= next_end)
        )

        loaded = pref_force >= float(touchdown_force_n)
        old_contact = (
            self._active
            & committed
            & loaded
            & (
                (pref_bottom - old_top).abs()
                <= float(top_height_tolerance_m)
            )
            & (pref_front >= old_start)
            & (pref_rear <= old_end)
        )
        next_contact = (
            self._active
            & committed
            & loaded
            & (
                (pref_bottom - next_top).abs()
                <= float(top_height_tolerance_m)
            )
            & (pref_front >= next_start)
            & (pref_rear <= next_end)
        )

        excursion = pref_front - self._lift_front_x
        excursion_violation = (
            self._active
            & committed
            & self._lift_front_valid
            & (excursion > float(max_forward_excursion_m))
        )

        sample_matches = runtime.sample_target_index == self._target
        strike_all = runtime.riser_face_strike_sample & sample_matches[:, None]
        severity_all = bounded_impact_severity(
            runtime.riser_strike_force_x,
            strike_all,
            minimum_force_n=float(strike_minimum_force_n),
            full_severity_force_n=float(strike_full_severity_force_n),
            minimum_event_fraction=float(strike_minimum_event_fraction),
        )
        preferred_strike = strike_all.gather(1, gather1).squeeze(1)
        preferred_severity = severity_all.gather(1, gather1).squeeze(1)
        open_strike = strike_all.any(dim=1)
        open_severity = severity_all.amax(dim=1)
        strike_event = torch.where(
            committed, preferred_strike, open_strike
        )
        strike_severity = torch.where(
            committed, preferred_severity, open_severity
        )

        acquisition_event = (
            runtime.step_acquired_event
            & (runtime.acquired_target_index == self._target)
        )
        physical = runtime.physical_touchdown_event
        preferred_acquisition = (
            committed
            & physical.gather(1, gather1).squeeze(1)
        )
        single_acquisition = physical.sum(dim=1) == 1

        (
            lifted,
            cleared,
            next_seen,
            resolved,
            reward,
            event_code,
        ) = step_over_transition_update(
            self._active,
            self._lifted,
            self._cleared,
            self._next_contact_seen,
            self._resolved,
            clear_lift=clear_lift,
            clear_riser=clear_riser,
            old_tread_contact=old_contact,
            next_tread_contact=next_contact,
            strike_event=strike_event,
            forward_excursion_violation=excursion_violation,
            acquisition_event=acquisition_event,
            preferred_acquisition=preferred_acquisition,
            single_acquisition=single_acquisition,
            strike_severity=strike_severity,
            success_value=float(success_value),
            failure_value=float(failure_value),
            strike_failure_base=float(strike_failure_base),
            strike_failure_severity_scale=(
                float(strike_failure_severity_scale)
            ),
        )
        new_lift_milestone = lifted & ~self._lifted
        new_clearance_milestone = cleared & ~self._cleared
        reward, milestone_credit_paid = escrowed_transition_reward(
            reward,
            event_code,
            new_lift_milestone,
            new_clearance_milestone,
            self._milestone_credit_paid,
            lift_milestone_value=float(lift_milestone_value),
            clearance_milestone_value=float(clearance_milestone_value),
        )

        self._lifted.copy_(lifted)
        self._cleared.copy_(cleared)
        self._next_contact_seen.copy_(next_seen)
        self._resolved.copy_(resolved)
        self._milestone_credit_paid.copy_(milestone_credit_paid)
        self.last_event_code.copy_(event_code)

        # After any physical acquisition, begin the next transition from the
        # newly synchronized target. The existing runtime chooses the opposite
        # foot; simultaneous acquisition leaves preferred=-1 and therefore does
        # not silently invent an alternating foot.
        acquired = runtime.step_acquired_event
        self._active.copy_(torch.where(acquired, torch.zeros_like(self._active), self._active))
        self._target.copy_(torch.where(acquired, torch.full_like(self._target, -1), self._target))
        self._preferred.copy_(torch.where(acquired, torch.full_like(self._preferred, -1), self._preferred))
        self._lifted.copy_(torch.where(acquired, torch.zeros_like(self._lifted), self._lifted))
        self._cleared.copy_(torch.where(acquired, torch.zeros_like(self._cleared), self._cleared))
        self._next_contact_seen.copy_(torch.where(acquired, torch.zeros_like(self._next_contact_seen), self._next_contact_seen))
        self._resolved.copy_(torch.where(acquired, torch.zeros_like(self._resolved), self._resolved))
        self._lift_front_valid.copy_(torch.where(acquired, torch.zeros_like(self._lift_front_valid), self._lift_front_valid))
        self._milestone_credit_paid.copy_(
            torch.where(
                acquired,
                torch.zeros_like(self._milestone_credit_paid),
                self._milestone_credit_paid,
            )
        )

        next_transition = (
            acquired
            & (current_target > 0)
            & (current_target < step_count)
            & (runtime.preferred_foot >= 0)
        )
        self._active.copy_(self._active | next_transition)
        self._target.copy_(
            torch.where(next_transition, current_target, self._target)
        )
        self._preferred.copy_(
            torch.where(
                next_transition,
                runtime.preferred_foot,
                self._preferred,
            )
        )

        # RewardManager multiplies by step_dt; convert one-shot transition
        # values into exact event returns.
        return reward / float(env.step_dt)



class riser_face_strike(ManagerTermBase):
    """Emit one event for a low forward impact on the current target riser."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)

        shape = (env.num_envs, 2)

        self._active = torch.zeros(
            shape,
            dtype=torch.bool,
            device=env.device,
        )
        self._previous_vx = torch.zeros(
            shape,
            dtype=torch.float32,
            device=env.device,
        )
        self._initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._active[env_ids] = False
        self._previous_vx[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        face_band_m: float = 0.025,
        below_top_margin_m: float = 0.005,
        minimum_horizontal_force_n: float = 30.0,
        minimum_previous_forward_speed_mps: float = 0.10,
        severity_enabled: bool = False,
        full_severity_force_n: float = 250.0,
        minimum_event_fraction: float = 0.25,
    ) -> torch.Tensor:
        if face_band_m <= 0.0:
            raise ValueError("face_band_m must be positive")
        if below_top_margin_m < 0.0:
            raise ValueError("below_top_margin_m cannot be negative")
        if minimum_horizontal_force_n <= 0.0:
            raise ValueError("minimum_horizontal_force_n must be positive")
        if minimum_previous_forward_speed_mps < 0.0:
            raise ValueError(
                "minimum_previous_forward_speed_mps cannot be negative"
            )
        if severity_enabled:
            if full_severity_force_n <= minimum_horizontal_force_n:
                raise ValueError(
                    "full_severity_force_n must exceed "
                    "minimum_horizontal_force_n"
                )
            if not 0.0 <= minimum_event_fraction <= 1.0:
                raise ValueError(
                    "minimum_event_fraction must be in [0, 1]"
                )

        runtime = self.runtime
        cfg = stair_cfg(env)

        tread = float(cfg.tread_depth)
        first_riser = float(cfg.start_clearance)
        step_count = int(cfg.step_count)

        riser_height = riser_height_per_env(env)

        data = runtime.robot.data

        foot_pos = (
            data.body_pos_w[:, runtime.robot_foot_ids]
            - env.scene.env_origins[:, None, :]
        )
        foot_quat = data.body_quat_w[:, runtime.robot_foot_ids]

        offsets = runtime.sole_centres.to(
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

        sole_bottom = (
            sole_points[..., 2] - radius
        ).amin(dim=2)

        rear_x = (
            sole_points[..., 0].amin(dim=2) - radius
        )
        front_x = (
            sole_points[..., 0].amax(dim=2) + radius
        )

        target_index = runtime.target_step.clamp(
            min=0,
            max=step_count - 1,
        )

        valid_target = (
            (runtime.target_step < step_count)
            & (riser_height[:, None] > 1.0e-8)
        )

        target_riser_x = (
            first_riser
            + target_index.to(foot_pos.dtype) * tread
        )

        target_tread_height = (
            (target_index + 1).to(foot_pos.dtype)
            * riser_height[:, None]
        )

        force_x = runtime.sensor.data.net_forces_w[
            :,
            runtime.sensor_foot_ids,
            0,
        ]

        current_vx = data.body_lin_vel_w[
            :,
            runtime.robot_foot_ids,
            0,
        ]

        previous_vx = torch.where(
            self._initialized[:, None],
            self._previous_vx,
            current_vx,
        )

        near_face = (
            (front_x >= target_riser_x - float(face_band_m))
            & (rear_x <= target_riser_x + float(face_band_m))
        )

        below_top = (
            sole_bottom
            <= target_tread_height - float(below_top_margin_m)
        )

        opposing_impact = (
            force_x <= -float(minimum_horizontal_force_n)
        )

        moving_forward_before_impact = (
            previous_vx
            >= float(minimum_previous_forward_speed_mps)
        )

        strike = (
            valid_target
            & near_face
            & below_top
            & opposing_impact
            & moving_forward_before_impact
        )

        event = strike & ~self._active

        self._active.copy_(strike)
        self._previous_vx.copy_(current_vx)
        self._initialized[:] = True

        if severity_enabled:
            value = bounded_impact_severity(
                force_x,
                event,
                minimum_force_n=float(minimum_horizontal_force_n),
                full_severity_force_n=float(full_severity_force_n),
                minimum_event_fraction=float(minimum_event_fraction),
            )
        else:
            value = event.to(dtype=foot_pos.dtype)

        # RewardManager multiplies by env.step_dt.
        return value.sum(dim=1) / float(env.step_dt)


class knee_forward_velocity_guard(ManagerTermBase):
    """Penalize excessive forward knee speed during tall-riser acquisition.

    The gate is geometry-relative and foot-specific. It uses the same per-foot
    target_step state as the V04 acquisition rewards, but does not prescribe a
    left/right gait sequence or a knee position trajectory. The term is zero
    on low stairs, outside the local riser-passage window, and for a loaded
    stance foot without opposite-foot support.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.runtime = _runtime(env)
        self.knee_ids = _required_indices(
            self.runtime.robot.body_names,
            KNEE_BODY_NAMES,
        )

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        self.runtime.reset(env_ids)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
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
        post_riser_distance_m: float = 0.12,
        high_riser_start_m: float = 0.14,
        high_riser_full_m: float = 0.18,
        free_forward_velocity_mps: float = 0.20,
        full_penalty_forward_velocity_mps: float = 0.70,
    ) -> torch.Tensor:
        del asset_cfg, contact_sensor_cfg

        # Keep shared target-step state current even if manager term ordering
        # changes. With the same acquisition parameters this is a no-op after
        # terrain_relative_clearance/forward_insertion have updated the step.
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

        if activation_distance_m <= 0.0:
            raise ValueError("activation_distance_m must be positive")
        if post_riser_distance_m <= 0.0:
            raise ValueError("post_riser_distance_m must be positive")
        if high_riser_start_m < 0.0:
            raise ValueError("high_riser_start_m cannot be negative")
        if high_riser_full_m <= high_riser_start_m:
            raise ValueError("high_riser_full_m must exceed high_riser_start_m")

        cfg = stair_cfg(env)
        riser_height = riser_height_per_env(env)
        tread = float(cfg.tread_depth)
        first_riser = float(cfg.start_clearance)
        step_count = int(cfg.step_count)
        data = self.runtime.robot.data

        foot_pos = (
            data.body_pos_w[:, self.runtime.robot_foot_ids]
            - env.scene.env_origins[:, None, :]
        )
        foot_quat = data.body_quat_w[:, self.runtime.robot_foot_ids]
        offsets = self.runtime.sole_centres.to(
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
        front_x = (
            sole_points[..., 0].amax(dim=2)
            + float(G1_SOLE_SPHERE_RADIUS_M)
        )

        valid_target = (
            (self.runtime.target_step < step_count)
            & (riser_height[:, None] > 1.0e-8)
        )
        target_index = self.runtime.target_step.clamp(
            min=0,
            max=step_count - 1,
        )
        next_riser_x = (
            first_riser
            + target_index.to(foot_pos.dtype) * tread
        )

        distance_to_edge = next_riser_x - front_x
        approach_gate = (
            1.0
            - distance_to_edge / float(activation_distance_m)
        ).clamp(0.0, 1.0)
        post_gate = (
            (
                next_riser_x
                + float(post_riser_distance_m)
                - front_x
            )
            / float(post_riser_distance_m)
        ).clamp(0.0, 1.0)

        vertical_force = (
            self.runtime.sensor.data.net_forces_w[
                :,
                self.runtime.sensor_foot_ids,
                2,
            ].clamp_min(0.0)
        )
        own_unload = _smooth_falling(
            vertical_force,
            float(unload_full_force_n),
            float(unload_zero_force_n),
        )
        opposite_force = torch.stack(
            (vertical_force[:, 1], vertical_force[:, 0]),
            dim=1,
        )
        opposite_support = _smooth_rising(
            opposite_force,
            float(opposite_support_start_n),
            float(opposite_support_full_n),
        )

        height_gate = (
            (riser_height - float(high_riser_start_m))
            / float(high_riser_full_m - high_riser_start_m)
        ).clamp(0.0, 1.0)[:, None]

        knee_forward_velocity = data.body_lin_vel_w[
            :,
            self.knee_ids,
            0,
        ]
        severity = normalized_excess_forward_velocity(
            knee_forward_velocity,
            free_velocity_mps=free_forward_velocity_mps,
            full_penalty_velocity_mps=full_penalty_forward_velocity_mps,
        )

        passage_gate = (
            approach_gate
            * post_gate
            * own_unload
            * opposite_support
            * height_gate
            * valid_target.to(foot_pos.dtype)
        )

        # At most one bounded unit of severity per environment. This prevents
        # accidental double-penalty if both legs momentarily satisfy the gate.
        return (passage_gate * severity).amax(dim=1)
