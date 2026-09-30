"""Geometry-driven reward terms for the active G1 stair specialist.

The active specialist intentionally avoids experiment-number-specific state
machines.  Each term below is computed from current physical geometry,
contact, and body state.  Stateful terms are limited to potential differences,
contact-onset events, and terminal dwell detection.

Isaac Lab multiplies every reward term by ``weight * env.step_dt``.  Potential
terms therefore return ``delta_potential / env.step_dt`` so a full potential
increase from 0 to 1 contributes exactly its configured weight over an
episode, independent of the 50 Hz control rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .contact_sensor_contract import (
    FOOT_BODY_NAMES,
    FORBIDDEN_CONTACT_BODY_NAMES,
    resolve_available_body_indices,
)
from .contracts import (
    BODY_JOINTS,
    G1_SOLE_SPHERE_CENTRES_M,
    G1_SOLE_SPHERE_RADIUS_M,
)
from .reward_math import (
    body_advance_gate,
    bottleneck_score,
    continuous_touchdown_potential,
    coupled_swing_score,
    deadzone_score,
    support_anchor_slip_penalty,
    support_quality,
    supported_single_foot_lift_score,
    swing_arc_waypoints,
    support_relative_forward_progress,
    swing_phase_potential,
    target_band_score,
    tread_containment_score,
)
from .terrain_state import riser_height_per_env, stair_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _required_indices(available: Sequence[str], requested: Sequence[str]) -> tuple[int, ...]:
    index_by_name = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in index_by_name]
    if missing:
        raise RuntimeError(
            "required stair-reward bodies missing: " + ", ".join(missing)
        )
    return tuple(index_by_name[name] for name in requested)


def _body_linear_velocity(data):
    if hasattr(data, "body_link_lin_vel_w"):
        return data.body_link_lin_vel_w
    if hasattr(data, "body_lin_vel_w"):
        return data.body_lin_vel_w
    raise RuntimeError(
        "robot articulation data has no body linear velocity tensor"
    )


def _quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by broadcast-compatible WXYZ quaternions."""

    q_w = quaternion[..., :1]
    q_v = quaternion[..., 1:]
    uv = torch.cross(q_v, vector, dim=-1)
    uuv = torch.cross(q_v, uv, dim=-1)
    return vector + 2.0 * (q_w * uv + uuv)


def _quat_to_euler_wxyz(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return roll, pitch, yaw for broadcast-compatible WXYZ quaternions."""

    w, x, y, z = quaternion.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = torch.asin(
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    )
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


class _SingleStepRewardBase(ManagerTermBase):
    """Shared physical measurements for one-step specialist rewards."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        robot = env.scene["robot"]
        sensor = env.scene.sensors.get("contact_forces")
        if sensor is None:
            raise RuntimeError(
                "specialist stair rewards require contact_forces sensor"
            )

        self._robot_foot_ids = _required_indices(
            tuple(robot.body_names),
            FOOT_BODY_NAMES,
        )
        self._sensor_foot_ids = _required_indices(
            tuple(sensor.body_names),
            FOOT_BODY_NAMES,
        )
        self._forbidden_sensor_ids = tuple(
            resolve_available_body_indices(
                tuple(sensor.body_names),
                FORBIDDEN_CONTACT_BODY_NAMES,
            ).values()
        )

        self._sole_centres = torch.tensor(
            G1_SOLE_SPHERE_CENTRES_M,
            dtype=torch.float32,
            device=env.device,
        )

        # Resolve bodyweight lazily on the first reward call. Reward-manager
        # class terms can be constructed before every PhysX view has reached
        # the same initialization point, while the first compute happens only
        # after the simulation is ready.
        self._bodyweight_n = None

    def _measure(
        self,
        env: "ManagerBasedRLEnv",
        *,
        asset_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        tread_margin_m: float,
        foot_height_tolerance_m: float,
        support_force_bodyweight_fraction: float,
        maximum_supported_foot_speed_mps: float,
        minimum_upright_gravity_z: float,
    ) -> dict[str, torch.Tensor | float]:
        robot = env.scene[asset_cfg.name]
        sensor = env.scene.sensors[contact_sensor_cfg.name]
        data = robot.data

        foot_ids = list(self._robot_foot_ids)
        sensor_ids = list(self._sensor_foot_ids)

        foot_pos_w = data.body_pos_w[:, foot_ids]
        foot_quat_w = data.body_quat_w[:, foot_ids]
        foot_lin_vel_w = _body_linear_velocity(data)[:, foot_ids]

        num_envs = foot_pos_w.shape[0]
        local_centres = self._sole_centres.to(
            dtype=foot_pos_w.dtype,
            device=foot_pos_w.device,
        )
        local_centres = local_centres.view(1, 1, 4, 3).expand(
            num_envs, 2, 4, 3
        )
        quaternions = foot_quat_w.unsqueeze(2).expand(
            num_envs, 2, 4, 4
        )
        sole_centres_w = foot_pos_w.unsqueeze(2) + _quat_rotate_wxyz(
            quaternions,
            local_centres,
        )

        origins = env.scene.env_origins
        sole_centres_local = sole_centres_w - origins[:, None, None, :]

        rear_x = sole_centres_local[..., 0].amin(dim=2)
        front_x = sole_centres_local[..., 0].amax(dim=2)
        sole_center = sole_centres_local.mean(dim=2)
        sole_bottom = (
            sole_centres_local[..., 2] - G1_SOLE_SPHERE_RADIUS_M
        ).amin(dim=2)
        foot_roll, foot_pitch, foot_yaw = _quat_to_euler_wxyz(foot_quat_w)

        foot_speed = torch.linalg.vector_norm(
            foot_lin_vel_w,
            dim=2,
        )
        foot_planar_speed = torch.linalg.vector_norm(
            foot_lin_vel_w[..., :2],
            dim=2,
        )

        forces = sensor.data.net_forces_w[:, sensor_ids]
        vertical_force = forces[..., 2].clamp_min(0.0)
        horizontal_force_x = forces[..., 0].abs()

        terrain = stair_cfg(env)
        riser_x = float(terrain.start_clearance)
        tread_depth = float(terrain.tread_depth)
        riser_height = riser_height_per_env(env)

        containment = tread_containment_score(
            rear_x,
            front_x,
            riser_x_m=riser_x,
            tread_depth_m=tread_depth,
            margin_m=tread_margin_m,
        )

        height_score = (
            1.0
            - torch.abs(
                sole_bottom - riser_height.unsqueeze(1)
            )
            / float(foot_height_tolerance_m)
        ).clamp(0.0, 1.0)

        if not 0.0 < float(support_force_bodyweight_fraction) <= 1.0:
            raise ValueError(
                "support_force_bodyweight_fraction must be in (0, 1]"
            )
        if self._bodyweight_n is None:
            masses = robot.root_physx_view.get_masses().to(device=env.device)
            self._bodyweight_n = masses.sum(dim=1) * 9.81
            if bool((self._bodyweight_n <= 0.0).any()):
                raise RuntimeError(
                    "specialist robot bodyweight must be positive"
                )
        full_support_force_n = (
            self._bodyweight_n.unsqueeze(1)
            * float(support_force_bodyweight_fraction)
        )
        physical_support = support_quality(
            containment=containment,
            height_score=height_score,
            vertical_force_n=vertical_force,
            speed_mps=foot_speed,
            full_support_force_n=full_support_force_n,
            maximum_supported_speed_mps=maximum_supported_foot_speed_mps,
        )

        upright = -data.projected_gravity_b[:, 2]

        local_root = data.root_pos_w - origins
        root_roll, root_pitch, root_yaw = _quat_to_euler_wxyz(data.root_quat_w)

        forbidden_contact = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        if self._forbidden_sensor_ids:
            forbidden_forces = sensor.data.net_forces_w[
                :, list(self._forbidden_sensor_ids)
            ]
            forbidden_contact = (
                torch.linalg.vector_norm(forbidden_forces, dim=2)
                >= 5.0
            ).any(dim=1)

        # Keep support a contact-quality signal rather than another posture
        # shaping term.  A loose upright/safety gate only invalidates support
        # after the robot is already in a clearly unsafe configuration.
        task_safe = (
            (upright >= float(minimum_upright_gravity_z))
            & ~forbidden_contact
        )
        support = (
            physical_support
            * task_safe.to(physical_support.dtype).unsqueeze(1)
        )

        return {
            "rear_x": rear_x,
            "front_x": front_x,
            "sole_center_x": sole_center[..., 0],
            "sole_center_y": sole_center[..., 1],
            "sole_bottom": sole_bottom,
            "foot_roll": foot_roll,
            "foot_pitch": foot_pitch,
            "foot_yaw": foot_yaw,
            "foot_speed": foot_speed,
            "foot_planar_speed": foot_planar_speed,
            "foot_vertical_velocity": foot_lin_vel_w[..., 2],
            "vertical_force": vertical_force,
            "bodyweight_n": self._bodyweight_n,
            "horizontal_force_x": horizontal_force_x,
            "support": support,
            "physical_support": physical_support,
            "root_x": local_root[:, 0],
            "root_y": local_root[:, 1],
            "root_z": local_root[:, 2],
            "root_roll": root_roll,
            "root_pitch": root_pitch,
            "root_yaw": root_yaw,
            "root_lin_speed": torch.linalg.vector_norm(
                data.root_lin_vel_w,
                dim=1,
            ),
            "root_ang_speed": torch.linalg.vector_norm(
                data.root_ang_vel_w,
                dim=1,
            ),
            "upright": upright,
            "task_safe": task_safe,
            "forbidden_contact": forbidden_contact,
            "riser_x": riser_x,
            "riser_height": riser_height,
            "tread_depth": tread_depth,
        }


class _PotentialDifferenceTerm(_SingleStepRewardBase):
    """Return a time-normalized change in a bounded physical potential."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self._initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous[env_ids] = 0.0
        self._initialized[env_ids] = False

    def _potential_rate(
        self,
        potential: torch.Tensor,
        *,
        dt: float,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dt <= 0.0:
            raise ValueError("reward step dt must be positive")

        potential = potential.clamp(0.0, 1.0)
        result = torch.zeros_like(potential)
        active = self._initialized

        if valid_mask is not None:
            if valid_mask.shape != potential.shape:
                raise ValueError(
                    "valid_mask must match the potential shape"
                )
            active = active & valid_mask

        result[active] = (
            potential[active] - self._previous[active]
        ) / float(dt)
        self._previous.copy_(potential.detach())
        self._initialized[:] = True
        return result


def failure_termination_event(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Emit one exact event for a native non-timeout episode termination."""

    terminated = getattr(env, "reset_terminated", None)
    if terminated is None:
        raise RuntimeError(
            "failure_termination_event requires env.reset_terminated"
        )
    if float(env.step_dt) <= 0.0:
        raise ValueError("reward step dt must be positive")
    return terminated.to(dtype=torch.float32) / float(env.step_dt)


class approach_progress(_PotentialDifferenceTerm):
    """Reward actual foot approach toward the riser, not pelvis-only lean."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._initial_foot_distance = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self._initial_root_distance = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self._distance_initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initial_foot_distance[env_ids] = 0.0
        self._initial_root_distance[env_ids] = 0.0
        self._distance_initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        preparation_distance_m: float = 0.18,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if preparation_distance_m <= 0.0:
            raise ValueError("preparation_distance_m must be positive")

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )
        # Genuine locomotor approach must move both the support geometry and
        # the body toward the riser. Root-only progress can be earned by
        # leaning; foot-only progress can be earned by sliding the feet out
        # from under the body. Couple both coordinates through the same
        # physical advance target so neither proxy can earn the reward alone.
        foot_distance = (
            float(m["riser_x"])
            - m["front_x"].mean(dim=1)
        )
        root_distance = float(m["riser_x"]) - m["root_x"]

        new = ~self._distance_initialized
        self._initial_foot_distance[new] = foot_distance[new].detach()
        self._initial_root_distance[new] = root_distance[new].detach()
        self._distance_initialized[new] = True

        advance_target = (
            self._initial_foot_distance - float(preparation_distance_m)
        ).clamp_min(0.05)
        foot_progress = (
            (self._initial_foot_distance - foot_distance) / advance_target
        ).clamp(0.0, 1.0)
        root_progress = (
            (self._initial_root_distance - root_distance) / advance_target
        ).clamp(0.0, 1.0)
        potential = torch.minimum(foot_progress, root_progress)

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class swing_waypoint_progress(_PotentialDifferenceTerm):
    """Five-phase swing arc that couples forward motion and lift from toe-off.

    Acquisition fixes one swing foot and shapes a physically ordered arc:

    0. load the support foot and unload the swing foot;
    1. move forward *and* upward to an early-lift waypoint;
    2. move forward *and* upward through a riser-clearance waypoint;
    3. advance over the tread while maintaining clearance;
    4. descend into a low-speed, stable tread touchdown.

    The lower support foot is anchored to its reset location throughout the
    airborne phases so whole-body shuffling cannot masquerade as swing-foot
    progress.  Positive swing progress also requires the swing foot to remain
    unloaded, inside its lateral lane, reasonably level in roll/yaw, and the
    torso to remain inside its orientation envelope.
    """

    _PHASE_COUNT = 5

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._dwell = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._lane_y = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._start_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._support_start_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._support_start_y = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._touchdown_entry_potential = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._phase[env_ids] = 0
        self._dwell[env_ids] = 0
        self._lane_y[env_ids] = 0.0
        self._start_x[env_ids] = 0.0
        self._support_start_x[env_ids] = 0.0
        self._support_start_y[env_ids] = 0.0
        self._touchdown_entry_potential[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        swing_left: bool = True,
        support_force_bodyweight_fraction: float = 0.40,
        swing_unload_bodyweight_fraction: float = 0.20,
        phase0_bottleneck: bool = False,
        phase0_require_grounded_swing: bool = False,
        phase0_swing_height_free_m: float = 0.010,
        phase0_swing_height_zero_m: float = 0.030,
        pelvis_support_tolerance_m: float = 0.09,
        pelvis_support_zero_m: float = 0.20,
        support_anchor_tolerance_m: float = 0.015,
        support_anchor_zero_m: float = 0.070,
        early_x_fraction: float = 0.35,
        clearance_x_fraction: float = 0.60,
        early_lift_riser_fraction: float = 0.55,
        clearance_margin_m: float = 0.025,
        hover_margin_m: float = 0.025,
        waypoint_x_tolerance_m: float = 0.030,
        waypoint_x_zero_m: float = 0.120,
        waypoint_z_tolerance_m: float = 0.015,
        waypoint_z_zero_m: float = 0.055,
        lateral_lane_tolerance_m: float = 0.030,
        lateral_lane_zero_m: float = 0.080,
        foot_roll_tolerance_rad: float = 0.12,
        foot_roll_zero_rad: float = 0.35,
        foot_yaw_tolerance_rad: float = 0.12,
        foot_yaw_zero_rad: float = 0.35,
        root_roll_tolerance_rad: float = 0.10,
        root_roll_zero_rad: float = 0.30,
        root_pitch_tolerance_rad: float = 0.20,
        root_pitch_zero_rad: float = 0.45,
        root_yaw_tolerance_rad: float = 0.10,
        root_yaw_zero_rad: float = 0.30,
        maximum_lower_support_foot_speed_mps: float = 0.20,
        transition_dwell_s: float = 0.06,
        touchdown_dwell_s: float = 0.08,
        maximum_touchdown_vertical_speed_mps: float = 0.25,
        reversible_touchdown_credit: bool = False,
        continuous_touchdown_credit: bool = False,
        touchdown_commitment_full_score: float = 0.75,
        touchdown_commitment_zero_score: float = 0.25,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        upper_support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if not 0.0 < support_force_bodyweight_fraction <= 1.0:
            raise ValueError("support_force_bodyweight_fraction must be in (0, 1]")
        if not 0.0 <= swing_unload_bodyweight_fraction < support_force_bodyweight_fraction:
            raise ValueError("swing_unload_bodyweight_fraction must be below support threshold")
        if not 0.0 < early_x_fraction < clearance_x_fraction < 1.0:
            raise ValueError("waypoint x fractions must satisfy 0 < early < clearance < 1")
        if transition_dwell_s <= 0.0 or touchdown_dwell_s <= 0.0:
            raise ValueError("phase dwell times must be positive")
        if not 0.0 <= touchdown_commitment_zero_score < touchdown_commitment_full_score <= 1.0:
            raise ValueError(
                "touchdown commitment thresholds must satisfy 0 <= zero < full <= 1"
            )
        if phase0_swing_height_free_m < 0.0:
            raise ValueError("phase0_swing_height_free_m cannot be negative")
        if phase0_swing_height_zero_m <= phase0_swing_height_free_m:
            raise ValueError(
                "phase0_swing_height_zero_m must exceed phase0_swing_height_free_m"
            )

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=upper_support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )
        swing = 0 if bool(swing_left) else 1
        support = 1 - swing
        new = ~self._initialized
        self._lane_y[new] = m["sole_center_y"][new, swing].detach()
        self._start_x[new] = m["sole_center_x"][new, swing].detach()
        self._support_start_x[new] = m["sole_center_x"][new, support].detach()
        self._support_start_y[new] = m["sole_center_y"][new, support].detach()
        self._initialized[new] = True

        bodyweight = m["bodyweight_n"].clamp_min(1.0)
        support_force_fraction = m["vertical_force"][:, support] / bodyweight
        swing_force_fraction = m["vertical_force"][:, swing] / bodyweight
        support_force_score = (
            (support_force_fraction - 0.25)
            / max(float(support_force_bodyweight_fraction) - 0.25, 1.0e-6)
        ).clamp(0.0, 1.0)
        swing_unload_score = (
            (0.35 - swing_force_fraction)
            / max(0.35 - float(swing_unload_bodyweight_fraction), 1.0e-6)
        ).clamp(0.0, 1.0)
        support_height_score = deadzone_score(
            m["sole_bottom"][:, support].abs(), free_tolerance=0.010, zero_at=0.040
        )
        support_speed_score = deadzone_score(
            m["foot_speed"][:, support],
            free_tolerance=maximum_lower_support_foot_speed_mps,
            zero_at=max(0.50, maximum_lower_support_foot_speed_mps + 0.20),
        )
        support_anchor_error = torch.sqrt(
            (m["sole_center_x"][:, support] - self._support_start_x) ** 2
            + (m["sole_center_y"][:, support] - self._support_start_y) ** 2
        )
        support_anchor_score = deadzone_score(
            support_anchor_error,
            free_tolerance=support_anchor_tolerance_m,
            zero_at=support_anchor_zero_m,
        )
        support_quality_lower = torch.minimum(
            torch.minimum(support_force_score, support_height_score),
            torch.minimum(support_speed_score, support_anchor_score),
        )
        pelvis_score = deadzone_score(
            (m["root_y"] - m["sole_center_y"][:, support]).abs(),
            free_tolerance=pelvis_support_tolerance_m,
            zero_at=pelvis_support_zero_m,
        )
        phase0_ground_score = deadzone_score(
            m["sole_bottom"][:, swing].clamp_min(0.0),
            free_tolerance=phase0_swing_height_free_m,
            zero_at=phase0_swing_height_zero_m,
        )
        lateral_score = deadzone_score(
            (m["sole_center_y"][:, swing] - self._lane_y).abs(),
            free_tolerance=lateral_lane_tolerance_m,
            zero_at=lateral_lane_zero_m,
        )
        foot_orientation_score = torch.minimum(
            deadzone_score(
                m["foot_roll"][:, swing].abs(),
                free_tolerance=foot_roll_tolerance_rad,
                zero_at=foot_roll_zero_rad,
            ),
            deadzone_score(
                m["foot_yaw"][:, swing].abs(),
                free_tolerance=foot_yaw_tolerance_rad,
                zero_at=foot_yaw_zero_rad,
            ),
        )
        body_orientation_score = torch.minimum(
            torch.minimum(
                deadzone_score(
                    m["root_roll"].abs(),
                    free_tolerance=root_roll_tolerance_rad,
                    zero_at=root_roll_zero_rad,
                ),
                deadzone_score(
                    m["root_pitch"].abs(),
                    free_tolerance=root_pitch_tolerance_rad,
                    zero_at=root_pitch_zero_rad,
                ),
            ),
            deadzone_score(
                m["root_yaw"].abs(),
                free_tolerance=root_yaw_tolerance_rad,
                zero_at=root_yaw_zero_rad,
            ),
        )
        airborne_gate = torch.minimum(
            swing_unload_score,
            torch.minimum(
                lateral_score,
                torch.minimum(
                    foot_orientation_score,
                    torch.minimum(support_quality_lower, body_orientation_score),
                ),
            ),
        )

        waypoints = swing_arc_waypoints(
            start_x_m=self._start_x,
            riser_x_m=float(m["riser_x"]),
            tread_depth_m=float(m["tread_depth"]),
            riser_height_m=m["riser_height"],
            early_x_fraction=early_x_fraction,
            clearance_x_fraction=clearance_x_fraction,
            early_lift_riser_fraction=early_lift_riser_fraction,
            clearance_margin_m=clearance_margin_m,
            hover_margin_m=hover_margin_m,
        )
        landing_x_tensor = waypoints["landing_x"]
        phase_targets_x = torch.stack(
            (self._start_x, waypoints["early_x"], waypoints["clearance_x"], waypoints["hover_x"], waypoints["hover_x"]),
            dim=1,
        )
        phase_targets_z = torch.stack(
            (torch.zeros_like(m["riser_height"]), waypoints["early_z"], waypoints["clearance_z"], waypoints["hover_z"], waypoints["touchdown_z"]),
            dim=1,
        )
        active_phase = self._phase.clamp(max=self._PHASE_COUNT - 1)
        target_x = phase_targets_x.gather(1, active_phase.unsqueeze(1)).squeeze(1)
        target_z = phase_targets_z.gather(1, active_phase.unsqueeze(1)).squeeze(1)
        x_error = (m["sole_center_x"][:, swing] - target_x).abs()
        z_error = (m["sole_bottom"][:, swing] - target_z).abs()
        x_score = deadzone_score(
            x_error,
            free_tolerance=waypoint_x_tolerance_m,
            zero_at=waypoint_x_zero_m,
        )
        z_score = deadzone_score(
            z_error,
            free_tolerance=waypoint_z_tolerance_m,
            zero_at=waypoint_z_zero_m,
        )
        waypoint_score = 0.5 * (x_score + z_score) * airborne_gate

        legacy_phase0 = (
            support_quality_lower + swing_unload_score + pelvis_score + body_orientation_score
        ) / 4.0
        phase0_components = [
            support_quality_lower,
            swing_unload_score,
            pelvis_score,
            body_orientation_score,
        ]
        if bool(phase0_require_grounded_swing):
            phase0_components.append(phase0_ground_score)
        phase0_locked = bottleneck_score(*phase0_components)
        phase0 = torch.where(
            torch.full_like(legacy_phase0, bool(phase0_bottleneck), dtype=torch.bool),
            phase0_locked,
            legacy_phase0,
        )
        phase1 = waypoint_score
        phase2 = waypoint_score
        phase3 = waypoint_score
        touchdown_speed_score = deadzone_score(
            m["foot_vertical_velocity"][:, swing].abs(),
            free_tolerance=maximum_touchdown_vertical_speed_mps,
            zero_at=max(0.80, maximum_touchdown_vertical_speed_mps + 0.40),
        )
        touchdown_gate = torch.minimum(
            lateral_score,
            torch.minimum(
                foot_orientation_score,
                torch.minimum(support_anchor_score, body_orientation_score),
            ),
        )
        phase4 = (
            0.45 * (0.5 * (x_score + z_score))
            + 0.35 * m["physical_support"][:, swing]
            + 0.20 * touchdown_speed_score
        ) * touchdown_gate
        phase_scores = torch.stack((phase0, phase1, phase2, phase3, phase4), dim=1)
        active_score = phase_scores.gather(1, active_phase.unsqueeze(1)).squeeze(1)
        complete = self._phase >= self._PHASE_COUNT
        active_score = torch.where(complete, torch.ones_like(active_score), active_score)

        common_airborne_transition = (
            (swing_force_fraction <= float(swing_unload_bodyweight_fraction))
            & (support_quality_lower >= 0.70)
            & (support_anchor_score >= 0.75)
            & (lateral_score >= 0.75)
            & (foot_orientation_score >= 0.75)
            & (body_orientation_score >= 0.75)
        )
        transition0 = (
            (support_force_fraction >= float(support_force_bodyweight_fraction))
            & (swing_force_fraction <= float(swing_unload_bodyweight_fraction))
            & (pelvis_score >= 0.65)
            & (support_anchor_score >= 0.85)
            & (body_orientation_score >= 0.75)
            & (
                (phase0_ground_score >= 0.75)
                if bool(phase0_require_grounded_swing)
                else torch.ones_like(phase0_ground_score, dtype=torch.bool)
            )
        )
        transition1 = common_airborne_transition & (x_score >= 0.75) & (z_score >= 0.75)
        transition2 = common_airborne_transition & (x_score >= 0.75) & (z_score >= 0.75)
        transition3 = common_airborne_transition & (x_score >= 0.75) & (z_score >= 0.75)
        transition4 = (
            (m["physical_support"][:, swing] >= 0.60)
            & (x_score >= 0.75)
            & (z_score >= 0.75)
            & (m["foot_vertical_velocity"][:, swing].abs() <= float(maximum_touchdown_vertical_speed_mps))
            & (lateral_score >= 0.75)
            & (foot_orientation_score >= 0.75)
            & (body_orientation_score >= 0.75)
        )
        transition_by_phase = torch.stack(
            (transition0, transition1, transition2, transition3, transition4), dim=1
        )
        transition = transition_by_phase.gather(
            1, active_phase.unsqueeze(1)
        ).squeeze(1) & ~complete

        potential = swing_phase_potential(
            phase=self._phase,
            active_score=active_score,
            phase_count=self._PHASE_COUNT,
            reversible_touchdown_credit=bool(reversible_touchdown_credit),
        )
        if bool(continuous_touchdown_credit):
            touchdown_phase = self._phase == (self._PHASE_COUNT - 1)
            current_commitment_score = bottleneck_score(
                x_score,
                lateral_score,
                foot_orientation_score,
                support_anchor_score,
                body_orientation_score,
            )
            touchdown_potential = continuous_touchdown_potential(
                entry_potential=self._touchdown_entry_potential,
                commitment_score=current_commitment_score,
                upper_support_score=m["physical_support"][:, swing],
                commitment_full_at=float(touchdown_commitment_full_score),
                commitment_zero_at=float(touchdown_commitment_zero_score),
            )
            potential = torch.where(
                touchdown_phase,
                touchdown_potential,
                potential,
            )

        self._dwell = torch.where(transition, self._dwell + 1, torch.zeros_like(self._dwell))
        normal_steps = max(1, int(round(float(transition_dwell_s) / float(env.step_dt))))
        touchdown_steps = max(1, int(round(float(touchdown_dwell_s) / float(env.step_dt))))
        required_steps = torch.where(
            active_phase == 4,
            torch.full_like(self._dwell, touchdown_steps),
            torch.full_like(self._dwell, normal_steps),
        )
        advance_phase = (self._dwell >= required_steps) & ~complete
        entering_touchdown = advance_phase & (active_phase == (self._PHASE_COUNT - 2))
        if bool(continuous_touchdown_credit):
            self._touchdown_entry_potential[entering_touchdown] = potential[
                entering_touchdown
            ].detach()
        self._phase[advance_phase] += 1
        self._dwell[advance_phase] = 0

        env._g1_stair_motion_phase = self._phase.detach().clone()
        env._g1_stair_motion_phase_progress = active_score.detach().clone()
        env._g1_stair_swing_lane_y = self._lane_y.detach().clone()
        env._g1_stair_swing_start_x = self._start_x.detach().clone()
        env._g1_stair_swing_target_x = target_x.detach().clone()
        env._g1_stair_swing_target_z = target_z.detach().clone()
        env._g1_stair_landing_target_x = landing_x_tensor.detach().clone()
        env._g1_stair_support_anchor_error = support_anchor_error.detach().clone()
        env._g1_stair_phase0_support_quality = support_quality_lower.detach().clone()
        env._g1_stair_phase0_swing_unload_score = swing_unload_score.detach().clone()
        env._g1_stair_phase0_pelvis_score = pelvis_score.detach().clone()
        env._g1_stair_phase0_ground_score = phase0_ground_score.detach().clone()
        env._g1_stair_phase0_bottleneck_score = phase0_locked.detach().clone()
        env._g1_stair_swing_left = bool(swing_left)

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class swing_body_advance_progress(_PotentialDifferenceTerm):
    """Reward controlled sagittal body commitment after riser clearance.

    This term is deliberately narrower than a forward-velocity command.  It
    stays dormant until the learned swing has reached the phase-2/3 clearance
    region with the opposite foot still planted.  From that point it rewards
    only new root advance toward a target that remains inside/near the support
    footprint.  If the swing retracts below the live clearance gate before
    touchdown, the potential is lost and the same term produces the matching
    negative shaping signal.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._swing_start_x = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._support_anchor_x = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._support_anchor_y = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._activation_root_x = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._geometry_initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._advance_initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._swing_start_x[env_ids] = 0.0
        self._support_anchor_x[env_ids] = 0.0
        self._support_anchor_y[env_ids] = 0.0
        self._activation_root_x[env_ids] = 0.0
        self._geometry_initialized[env_ids] = False
        self._advance_initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        swing_left: bool = True,
        target_forward_from_support_m: float = 0.080,
        clearance_x_fraction: float = 0.60,
        clearance_x_tolerance_m: float = 0.040,
        minimum_clearance_above_riser_m: float = 0.010,
        support_anchor_limit_m: float = 0.050,
        support_anchor_xy: bool = False,
        support_force_bodyweight_fraction: float = 0.25,
        swing_unload_bodyweight_fraction: float = 0.25,
        require_live_touchdown_commitment: bool = False,
        touchdown_containment_minimum: float = 0.75,
        upper_support_minimum: float = 0.60,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        upper_support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if target_forward_from_support_m <= 0.0:
            raise ValueError("target_forward_from_support_m must be positive")
        if not 0.0 < clearance_x_fraction < 1.0:
            raise ValueError("clearance_x_fraction must be in (0, 1)")
        if clearance_x_tolerance_m < 0.0:
            raise ValueError("clearance_x_tolerance_m cannot be negative")
        if minimum_clearance_above_riser_m < 0.0:
            raise ValueError("minimum_clearance_above_riser_m cannot be negative")
        if support_anchor_limit_m <= 0.0:
            raise ValueError("support_anchor_limit_m must be positive")
        if not 0.0 < support_force_bodyweight_fraction <= 1.0:
            raise ValueError("support_force_bodyweight_fraction must be in (0, 1]")
        if not 0.0 <= swing_unload_bodyweight_fraction < 1.0:
            raise ValueError("swing_unload_bodyweight_fraction must be in [0, 1)")
        if not 0.0 <= touchdown_containment_minimum <= 1.0:
            raise ValueError("touchdown_containment_minimum must be in [0, 1]")
        if not 0.0 <= upper_support_minimum <= 1.0:
            raise ValueError("upper_support_minimum must be in [0, 1]")

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=upper_support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )
        swing = 0 if bool(swing_left) else 1
        support = 1 - swing

        new_geometry = ~self._geometry_initialized
        self._swing_start_x[new_geometry] = (
            m["sole_center_x"][new_geometry, swing].detach()
        )
        self._support_anchor_x[new_geometry] = (
            m["sole_center_x"][new_geometry, support].detach()
        )
        self._support_anchor_y[new_geometry] = (
            m["sole_center_y"][new_geometry, support].detach()
        )
        self._geometry_initialized[new_geometry] = True

        phase = getattr(env, "_g1_stair_motion_phase", None)
        if phase is None:
            phase = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )

        bodyweight = m["bodyweight_n"].clamp_min(1.0)
        support_force_fraction = m["vertical_force"][:, support] / bodyweight
        swing_force_fraction = m["vertical_force"][:, swing] / bodyweight
        support_anchor_error_x = (
            m["sole_center_x"][:, support] - self._support_anchor_x
        ).abs()
        support_anchor_error_xy = torch.sqrt(
            (m["sole_center_x"][:, support] - self._support_anchor_x) ** 2
            + (m["sole_center_y"][:, support] - self._support_anchor_y) ** 2
        )
        support_anchor_error = torch.where(
            torch.full_like(
                support_anchor_error_x, bool(support_anchor_xy), dtype=torch.bool
            ),
            support_anchor_error_xy,
            support_anchor_error_x,
        )

        landing_x = float(m["riser_x"]) + 0.5 * float(m["tread_depth"])
        path = (landing_x - self._swing_start_x).clamp_min(0.05)
        clearance_x = (
            self._swing_start_x + float(clearance_x_fraction) * path
        )
        swing_clear_now = (
            m["sole_center_x"][:, swing]
            >= clearance_x - float(clearance_x_tolerance_m)
        ) & (
            m["sole_bottom"][:, swing]
            >= m["riser_height"] + float(minimum_clearance_above_riser_m)
        )
        support_valid_now = (
            support_anchor_error <= float(support_anchor_limit_m)
        ) & (
            support_force_fraction >= float(support_force_bodyweight_fraction)
        ) & (
            swing_force_fraction <= float(swing_unload_bodyweight_fraction)
        )
        clearance_phase = (phase >= 2) & (phase <= 3)
        live_gate = clearance_phase & swing_clear_now & support_valid_now & m["task_safe"]

        activate = live_gate & ~self._advance_initialized
        self._activation_root_x[activate] = m["root_x"][activate].detach()
        self._advance_initialized[activate] = True

        progress = support_relative_forward_progress(
            root_x_m=m["root_x"],
            support_anchor_x_m=self._support_anchor_x,
            activation_root_x_m=self._activation_root_x,
            target_forward_from_support_m=target_forward_from_support_m,
        )
        touchdown_containment = tread_containment_score(
            m["rear_x"][:, swing],
            m["front_x"][:, swing],
            riser_x_m=float(m["riser_x"]),
            tread_depth_m=float(m["tread_depth"]),
            margin_m=float(tread_margin_m),
        )
        touchdown_placement_now = (
            (touchdown_containment >= float(touchdown_containment_minimum))
            & (
                m["sole_bottom"][:, swing]
                >= m["riser_height"] - float(foot_height_tolerance_m)
            )
            & support_valid_now
            & m["task_safe"]
        )
        upper_support_now = (
            m["physical_support"][:, swing] >= float(upper_support_minimum)
        ) & m["task_safe"]

        # Historical v06 keeps body progress live solely because phase 4 was
        # reached.  v07 opts into a stricter contract: phase 4 must still have
        # a physically committed tread placement or genuine upper support.
        # Swing retraction therefore removes the potential and yields the
        # intended negative potential difference.
        physical_gate = body_advance_gate(
            phase=phase,
            live_clearance_gate=live_gate,
            touchdown_placement_gate=touchdown_placement_now,
            upper_support_gate=upper_support_now,
            require_live_touchdown_commitment=bool(
                require_live_touchdown_commitment
            ),
            touchdown_phase_index=4,
        )
        potential = torch.where(
            self._advance_initialized & physical_gate,
            progress,
            torch.zeros_like(progress),
        )

        env._g1_stair_root_to_support_x = (
            m["root_x"] - self._support_anchor_x
        ).detach().clone()
        env._g1_stair_body_advance_target_x = (
            self._support_anchor_x + float(target_forward_from_support_m)
        ).detach().clone()
        env._g1_stair_body_advance_progress = progress.detach().clone()
        env._g1_stair_body_advance_gate = physical_gate.detach().clone()
        env._g1_stair_body_support_anchor_error_xy = (
            support_anchor_error_xy.detach().clone()
        )

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class support_foot_anchor_guard(_SingleStepRewardBase):
    """Penalize XY drift and planar slip of the loaded lower support foot.

    This term is deliberately independent of the broad swing-waypoint guard.
    The latter uses a maximum over many failure modes and can saturate, making
    additional support-foot sliding effectively free.  This guard remains
    dedicated to the support contact during the airborne/touchdown phases.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._support_start_x = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._support_start_y = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._support_start_x[env_ids] = 0.0
        self._support_start_y[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        swing_left: bool = True,
        support_force_bodyweight_fraction: float = 0.25,
        swing_unload_bodyweight_fraction: float = 0.30,
        anchor_free_m: float = 0.015,
        anchor_zero_m: float = 0.060,
        speed_free_mps: float = 0.020,
        speed_zero_mps: float = 0.20,
        active_phase_min: int = 1,
        active_phase_max: int = 4,
        anchor_phase_gate_only: bool = False,
        latch_anchor_on_active_phase: bool = False,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        upper_support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if not 0.0 < support_force_bodyweight_fraction <= 1.0:
            raise ValueError("support_force_bodyweight_fraction must be in (0, 1]")
        if not 0.0 <= swing_unload_bodyweight_fraction < 1.0:
            raise ValueError("swing_unload_bodyweight_fraction must be in [0, 1)")
        if active_phase_min < 0 or active_phase_max < active_phase_min:
            raise ValueError("invalid active phase range")

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=upper_support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )
        swing = 0 if bool(swing_left) else 1
        support = 1 - swing

        phase = getattr(env, "_g1_stair_motion_phase", None)
        if phase is None:
            phase = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )

        phase_gate = (
            (phase >= int(active_phase_min))
            & (phase <= int(active_phase_max))
            & m["task_safe"]
        )
        new = ~self._initialized
        if bool(latch_anchor_on_active_phase):
            new = new & phase_gate
        self._support_start_x[new] = m["sole_center_x"][new, support].detach()
        self._support_start_y[new] = m["sole_center_y"][new, support].detach()
        self._initialized[new] = True

        bodyweight = m["bodyweight_n"].clamp_min(1.0)
        support_force_fraction = m["vertical_force"][:, support] / bodyweight
        swing_force_fraction = m["vertical_force"][:, swing] / bodyweight
        loaded_support_gate = (
            phase_gate
            & (support_force_fraction >= float(support_force_bodyweight_fraction))
        )
        legacy_gate = (
            loaded_support_gate
            & (swing_force_fraction <= float(swing_unload_bodyweight_fraction))
        )
        anchor_gate = phase_gate if bool(anchor_phase_gate_only) else legacy_gate
        speed_gate = loaded_support_gate if bool(anchor_phase_gate_only) else legacy_gate
        active_gate = anchor_gate

        anchor_error = torch.sqrt(
            (m["sole_center_x"][:, support] - self._support_start_x) ** 2
            + (m["sole_center_y"][:, support] - self._support_start_y) ** 2
        )
        penalty = support_anchor_slip_penalty(
            anchor_error_m=anchor_error,
            planar_speed_mps=m["foot_planar_speed"][:, support],
            active_gate=legacy_gate,
            anchor_gate=anchor_gate,
            speed_gate=speed_gate,
            anchor_free_m=float(anchor_free_m),
            anchor_zero_m=float(anchor_zero_m),
            speed_free_mps=float(speed_free_mps),
            speed_zero_mps=float(speed_zero_mps),
        )

        env._g1_stair_support_lock_anchor_error_xy = anchor_error.detach().clone()
        env._g1_stair_support_lock_planar_speed = (
            m["foot_planar_speed"][:, support].detach().clone()
        )
        env._g1_stair_support_lock_gate = active_gate.detach().clone()
        return penalty


class swing_waypoint_guard(_SingleStepRewardBase):
    """Penalty envelope for v04 swing-arc failure modes."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._lane_y = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._start_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._support_start_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._support_start_y = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lane_y[env_ids] = 0.0
        self._start_x[env_ids] = 0.0
        self._support_start_x[env_ids] = 0.0
        self._support_start_y[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        swing_left: bool = True,
        enforce_phase0_support: bool = False,
        phase0_swing_load_free_bodyweight_fraction: float = 0.30,
        phase0_swing_load_zero_bodyweight_fraction: float = 0.50,
        phase0_swing_height_free_m: float = 0.012,
        phase0_swing_height_zero_m: float = 0.035,
        phase0_support_speed_free_mps: float = 0.15,
        phase0_support_speed_zero_mps: float = 0.50,
        lateral_free_m: float = 0.030,
        lateral_zero_m: float = 0.100,
        support_anchor_free_m: float = 0.020,
        support_anchor_zero_m: float = 0.080,
        maximum_height_above_riser_m: float = 0.070,
        overheight_zero_m: float = 0.180,
        backward_free_m: float = 0.020,
        backward_zero_m: float = 0.100,
        foot_roll_free_rad: float = 0.15,
        foot_roll_zero_rad: float = 0.45,
        foot_yaw_free_rad: float = 0.15,
        foot_yaw_zero_rad: float = 0.45,
        root_roll_free_rad: float = 0.12,
        root_roll_zero_rad: float = 0.35,
        root_pitch_free_rad: float = 0.25,
        root_pitch_zero_rad: float = 0.55,
        pelvis_support_free_m: float = 0.11,
        pelvis_support_zero_m: float = 0.24,
        support_force_bodyweight_fraction: float = 0.25,
        lower_floor_recontact_force_bodyweight_fraction: float = 0.12,
        maximum_swing_speed_mps: float = 1.20,
        swing_speed_zero_mps: float = 2.20,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        upper_support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if not 0.0 <= phase0_swing_load_free_bodyweight_fraction < phase0_swing_load_zero_bodyweight_fraction:
            raise ValueError("phase0 swing-load thresholds must be ordered")
        if phase0_swing_height_free_m < 0.0 or phase0_swing_height_zero_m <= phase0_swing_height_free_m:
            raise ValueError("phase0 swing-height thresholds must be ordered")
        if phase0_support_speed_free_mps < 0.0 or phase0_support_speed_zero_mps <= phase0_support_speed_free_mps:
            raise ValueError("phase0 support-speed thresholds must be ordered")
        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=upper_support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )
        swing = 0 if bool(swing_left) else 1
        support = 1 - swing
        new = ~self._initialized
        self._lane_y[new] = m["sole_center_y"][new, swing].detach()
        self._start_x[new] = m["sole_center_x"][new, swing].detach()
        self._support_start_x[new] = m["sole_center_x"][new, support].detach()
        self._support_start_y[new] = m["sole_center_y"][new, support].detach()
        self._initialized[new] = True

        phase = getattr(env, "_g1_stair_motion_phase", None)
        if phase is None:
            phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        bodyweight = m["bodyweight_n"].clamp_min(1.0)
        support_force_fraction = m["vertical_force"][:, support] / bodyweight
        swing_force_fraction = m["vertical_force"][:, swing] / bodyweight
        landing_x = float(m["riser_x"]) + 0.5 * float(m["tread_depth"])
        support_anchor_error = torch.sqrt(
            (m["sole_center_x"][:, support] - self._support_start_x) ** 2
            + (m["sole_center_y"][:, support] - self._support_start_y) ** 2
        )

        lateral_bad = 1.0 - deadzone_score(
            (m["sole_center_y"][:, swing] - self._lane_y).abs(),
            free_tolerance=lateral_free_m,
            zero_at=lateral_zero_m,
        )
        support_anchor_bad = 1.0 - deadzone_score(
            support_anchor_error,
            free_tolerance=support_anchor_free_m,
            zero_at=support_anchor_zero_m,
        )
        max_height = m["riser_height"] + float(maximum_height_above_riser_m)
        overheight_bad = (
            (m["sole_bottom"][:, swing] - max_height).clamp_min(0.0)
            / max(float(overheight_zero_m) - float(maximum_height_above_riser_m), 1.0e-6)
        ).clamp(0.0, 1.0)
        backward_bad = (
            (self._start_x - m["sole_center_x"][:, swing] - float(backward_free_m)).clamp_min(0.0)
            / max(float(backward_zero_m) - float(backward_free_m), 1.0e-6)
        ).clamp(0.0, 1.0)
        forward_overshoot_bad = (
            (m["sole_center_x"][:, swing] - (landing_x + 0.050)).clamp_min(0.0) / 0.150
        ).clamp(0.0, 1.0)
        foot_roll_bad = 1.0 - deadzone_score(
            m["foot_roll"][:, swing].abs(),
            free_tolerance=foot_roll_free_rad,
            zero_at=foot_roll_zero_rad,
        )
        foot_yaw_bad = 1.0 - deadzone_score(
            m["foot_yaw"][:, swing].abs(),
            free_tolerance=foot_yaw_free_rad,
            zero_at=foot_yaw_zero_rad,
        )
        root_roll_bad = 1.0 - deadzone_score(
            m["root_roll"].abs(),
            free_tolerance=root_roll_free_rad,
            zero_at=root_roll_zero_rad,
        )
        root_pitch_bad = 1.0 - deadzone_score(
            m["root_pitch"].abs(),
            free_tolerance=root_pitch_free_rad,
            zero_at=root_pitch_zero_rad,
        )
        pelvis_bad = 1.0 - deadzone_score(
            (m["root_y"] - m["sole_center_y"][:, support]).abs(),
            free_tolerance=pelvis_support_free_m,
            zero_at=pelvis_support_zero_m,
        )
        support_loss = (
            1.0 - support_force_fraction / max(float(support_force_bodyweight_fraction), 1.0e-6)
        ).clamp(0.0, 1.0)
        support_lift_bad = (
            (m["sole_bottom"][:, support] - 0.010).clamp_min(0.0) / 0.050
        ).clamp(0.0, 1.0)
        both_airborne = (
            (m["vertical_force"][:, 0] < 5.0)
            & (m["vertical_force"][:, 1] < 5.0)
            & (m["sole_bottom"][:, 0] > 0.010)
            & (m["sole_bottom"][:, 1] > 0.010)
        ).to(dtype=lateral_bad.dtype)
        floor_recontact_bad = (
            (phase >= 1)
            & (phase <= 3)
            & (m["sole_bottom"][:, swing] <= 0.025)
            & (swing_force_fraction >= float(lower_floor_recontact_force_bodyweight_fraction))
        ).to(dtype=lateral_bad.dtype)
        swing_speed_bad = (
            (m["foot_speed"][:, swing] - float(maximum_swing_speed_mps)).clamp_min(0.0)
            / max(float(swing_speed_zero_mps) - float(maximum_swing_speed_mps), 1.0e-6)
        ).clamp(0.0, 1.0)
        phase0_swing_loaded_bad = (
            (swing_force_fraction - float(phase0_swing_load_free_bodyweight_fraction)).clamp_min(0.0)
            / max(
                float(phase0_swing_load_zero_bodyweight_fraction)
                - float(phase0_swing_load_free_bodyweight_fraction),
                1.0e-6,
            )
        ).clamp(0.0, 1.0)
        phase0_swing_lift_bad = (
            (m["sole_bottom"][:, swing].clamp_min(0.0) - float(phase0_swing_height_free_m)).clamp_min(0.0)
            / max(
                float(phase0_swing_height_zero_m) - float(phase0_swing_height_free_m),
                1.0e-6,
            )
        ).clamp(0.0, 1.0)
        phase0_support_speed_bad = (
            (m["foot_speed"][:, support] - float(phase0_support_speed_free_mps)).clamp_min(0.0)
            / max(
                float(phase0_support_speed_zero_mps) - float(phase0_support_speed_free_mps),
                1.0e-6,
            )
        ).clamp(0.0, 1.0)

        active = (phase >= 1).to(dtype=lateral_bad.dtype)
        severity = torch.stack(
            (
                lateral_bad,
                support_anchor_bad,
                overheight_bad,
                backward_bad,
                forward_overshoot_bad,
                foot_roll_bad,
                foot_yaw_bad,
                root_roll_bad,
                root_pitch_bad,
                pelvis_bad,
                support_loss,
                support_lift_bad,
                both_airborne,
                floor_recontact_bad,
                swing_speed_bad,
            ),
            dim=1,
        ).amax(dim=1)
        phase0_severity = torch.stack(
            (
                support_anchor_bad,
                support_loss,
                support_lift_bad,
                phase0_support_speed_bad,
                pelvis_bad,
                root_roll_bad,
                root_pitch_bad,
                phase0_swing_loaded_bad,
                phase0_swing_lift_bad,
            ),
            dim=1,
        ).amax(dim=1)
        if bool(enforce_phase0_support):
            return torch.where(phase == 0, phase0_severity, severity)
        return severity * active


class posture_envelope_guard(ManagerTermBase):
    """Soft full-body posture envelope without freezing balance motions."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        index_by_name = {name: i for i, name in enumerate(robot.joint_names)}
        missing = [name for name in BODY_JOINTS if name not in index_by_name]
        if missing:
            raise RuntimeError("posture envelope missing G1 joints: " + ", ".join(missing))
        self._arm_ids = tuple(
            index_by_name[name]
            for name in BODY_JOINTS
            if any(token in name for token in ("_shoulder_", "_elbow_", "_wrist_"))
        )

        # Opt-in V08-A3 groups. Historical posture-envelope behavior below
        # remains unchanged unless joint_specific_excess_squared=True.
        self._shoulder_pitch_ids = tuple(
            index_by_name[name]
            for name in BODY_JOINTS
            if "_shoulder_pitch_" in name
        )
        self._shoulder_roll_ids = tuple(
            index_by_name[name]
            for name in BODY_JOINTS
            if "_shoulder_roll_" in name
        )
        self._shoulder_yaw_ids = tuple(
            index_by_name[name]
            for name in BODY_JOINTS
            if "_shoulder_yaw_" in name
        )
        self._elbow_ids = tuple(
            index_by_name[name]
            for name in BODY_JOINTS
            if "_elbow_" in name
        )
        self._wrist_ids = tuple(
            index_by_name[name]
            for name in BODY_JOINTS
            if "_wrist_" in name
        )

        self._waist_ids = tuple(
            index_by_name[name] for name in BODY_JOINTS if name.startswith("waist_")
        )
        self._lateral_leg_ids = tuple(
            index_by_name[name]
            for name in BODY_JOINTS
            if any(token in name for token in ("_hip_roll_", "_hip_yaw_", "_ankle_roll_"))
        )

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        arms_free_rms_rad: float = 0.25,
        arms_zero_rms_rad: float = 0.70,
        waist_free_rms_rad: float = 0.15,
        waist_zero_rms_rad: float = 0.45,
        lateral_legs_free_rms_rad: float = 0.22,
        lateral_legs_zero_rms_rad: float = 0.55,
        arms_only: bool = False,
        joint_specific_excess_squared: bool = False,
        joint_specific_target_excess_squared: bool = False,
        shoulder_pitch_tolerance_rad: float = 0.45,
        shoulder_pitch_relative_weight: float = 0.25,
        shoulder_roll_tolerance_rad: float = 0.40,
        shoulder_roll_relative_weight: float = 0.50,
        shoulder_yaw_tolerance_rad: float = 0.25,
        shoulder_yaw_relative_weight: float = 1.50,
        elbow_tolerance_rad: float = 0.45,
        elbow_relative_weight: float = 0.25,
        wrist_tolerance_rad: float = 0.25,
        wrist_relative_weight: float = 2.00,
    ) -> torch.Tensor:
        data = env.scene[asset_cfg.name].data
        error = data.joint_pos - data.default_joint_pos

        def rms(ids):
            return torch.sqrt(torch.mean(error[:, list(ids)] ** 2, dim=1))

        if joint_specific_excess_squared or joint_specific_target_excess_squared:
            if joint_specific_excess_squared and joint_specific_target_excess_squared:
                raise ValueError(
                    "select only one joint-specific arm error source"
                )

            if joint_specific_target_excess_squared:
                if data.joint_pos_target is None:
                    raise RuntimeError(
                        "joint-specific target shaping requires joint_pos_target"
                    )
                joint_specific_error = (
                    data.joint_pos_target - data.default_joint_pos
                )
            else:
                joint_specific_error = error

            groups = (
                (
                    "shoulder_pitch",
                    self._shoulder_pitch_ids,
                    2,
                    shoulder_pitch_tolerance_rad,
                    shoulder_pitch_relative_weight,
                ),
                (
                    "shoulder_roll",
                    self._shoulder_roll_ids,
                    2,
                    shoulder_roll_tolerance_rad,
                    shoulder_roll_relative_weight,
                ),
                (
                    "shoulder_yaw",
                    self._shoulder_yaw_ids,
                    2,
                    shoulder_yaw_tolerance_rad,
                    shoulder_yaw_relative_weight,
                ),
                (
                    "elbow",
                    self._elbow_ids,
                    2,
                    elbow_tolerance_rad,
                    elbow_relative_weight,
                ),
                (
                    "wrist",
                    self._wrist_ids,
                    6,
                    wrist_tolerance_rad,
                    wrist_relative_weight,
                ),
            )

            numerator = torch.zeros(
                env.num_envs,
                device=joint_specific_error.device,
                dtype=joint_specific_error.dtype,
            )
            denominator = 0.0

            for name, ids, expected_count, tolerance, relative_weight in groups:
                if len(ids) != expected_count:
                    raise RuntimeError(
                        "unexpected G1 joint count for "
                        f"{name}: {len(ids)} != {expected_count}"
                    )
                if tolerance < 0.0:
                    raise ValueError(
                        f"{name} tolerance must be non-negative"
                    )
                if relative_weight < 0.0:
                    raise ValueError(
                        f"{name} relative weight must be non-negative"
                    )

                excess = (
                    joint_specific_error[:, list(ids)].abs()
                    - float(tolerance)
                ).clamp_min(0.0)

                numerator = numerator + float(relative_weight) * torch.sum(
                    excess.square(),
                    dim=1,
                )
                denominator += float(relative_weight) * len(ids)

            if denominator <= 0.0:
                raise ValueError(
                    "joint-specific arm relative weights must sum positive"
                )

            return numerator / denominator

        arm_bad = 1.0 - deadzone_score(
            rms(self._arm_ids), free_tolerance=arms_free_rms_rad, zero_at=arms_zero_rms_rad
        )
        if arms_only:
            return arm_bad
        waist_bad = 1.0 - deadzone_score(
            rms(self._waist_ids), free_tolerance=waist_free_rms_rad, zero_at=waist_zero_rms_rad
        )
        lateral_leg_bad = 1.0 - deadzone_score(
            rms(self._lateral_leg_ids),
            free_tolerance=lateral_legs_free_rms_rad,
            zero_at=lateral_legs_zero_rms_rad,
        )
        return torch.stack((arm_bad, waist_bad, lateral_leg_bad), dim=1).amax(dim=1)


class swing_target_progress(_PotentialDifferenceTerm):
    """Four-phase left/right single-step swing target with explicit guard rails.

    The term intentionally fixes one swing foot during acquisition.  It rewards
    a physical sequence rather than an unconstrained lift:

    0. unload swing foot while loading the opposite support foot;
    1. lift inside the original lateral lane;
    2. advance at controlled clearance toward the tread centre;
    3. acquire stable upper-tread support.

    Each phase contributes one quarter of a bounded [0, 1] potential, so a
    completed swing can earn its configured weight only once per episode.
    """

    _PHASE_COUNT = 4

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._dwell = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._lane_y = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._start_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._geometry_initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._phase[env_ids] = 0
        self._dwell[env_ids] = 0
        self._lane_y[env_ids] = 0.0
        self._start_x[env_ids] = 0.0
        self._geometry_initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        swing_left: bool = True,
        support_force_bodyweight_fraction: float = 0.45,
        swing_unload_bodyweight_fraction: float = 0.20,
        pelvis_support_tolerance_m: float = 0.08,
        pelvis_support_zero_m: float = 0.18,
        lift_clearance_m: float = 0.030,
        lateral_lane_tolerance_m: float = 0.030,
        lateral_lane_zero_m: float = 0.080,
        foot_roll_tolerance_rad: float = 0.12,
        foot_roll_zero_rad: float = 0.35,
        foot_yaw_tolerance_rad: float = 0.12,
        foot_yaw_zero_rad: float = 0.35,
        root_roll_tolerance_rad: float = 0.10,
        root_roll_zero_rad: float = 0.30,
        root_pitch_tolerance_rad: float = 0.20,
        root_pitch_zero_rad: float = 0.45,
        root_yaw_tolerance_rad: float = 0.10,
        root_yaw_zero_rad: float = 0.30,
        maximum_lower_support_foot_speed_mps: float = 0.20,
        transition_dwell_s: float = 0.06,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        upper_support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if not 0.0 < support_force_bodyweight_fraction <= 1.0:
            raise ValueError("support_force_bodyweight_fraction must be in (0, 1]")
        if not 0.0 <= swing_unload_bodyweight_fraction < support_force_bodyweight_fraction:
            raise ValueError("swing_unload_bodyweight_fraction must be below support threshold")
        if lift_clearance_m <= 0.0:
            raise ValueError("lift_clearance_m must be positive")
        if transition_dwell_s <= 0.0:
            raise ValueError("transition_dwell_s must be positive")

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=upper_support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )

        swing = 0 if bool(swing_left) else 1
        support = 1 - swing
        new = ~self._geometry_initialized
        self._lane_y[new] = m["sole_center_y"][new, swing].detach()
        self._start_x[new] = m["sole_center_x"][new, swing].detach()
        self._geometry_initialized[new] = True

        bodyweight = m["bodyweight_n"].clamp_min(1.0)
        support_force_fraction = m["vertical_force"][:, support] / bodyweight
        swing_force_fraction = m["vertical_force"][:, swing] / bodyweight

        support_force_score = (
            (support_force_fraction - 0.25)
            / max(float(support_force_bodyweight_fraction) - 0.25, 1.0e-6)
        ).clamp(0.0, 1.0)
        swing_unload_score = (
            (0.35 - swing_force_fraction)
            / max(0.35 - float(swing_unload_bodyweight_fraction), 1.0e-6)
        ).clamp(0.0, 1.0)
        support_height_score = deadzone_score(
            m["sole_bottom"][:, support].abs(),
            free_tolerance=0.010,
            zero_at=0.040,
        )
        support_speed_score = deadzone_score(
            m["foot_speed"][:, support],
            free_tolerance=maximum_lower_support_foot_speed_mps,
            zero_at=max(0.50, maximum_lower_support_foot_speed_mps + 0.20),
        )
        support_quality_lower = torch.minimum(
            torch.minimum(support_force_score, support_height_score),
            support_speed_score,
        )

        pelvis_score = deadzone_score(
            (m["root_y"] - m["sole_center_y"][:, support]).abs(),
            free_tolerance=pelvis_support_tolerance_m,
            zero_at=pelvis_support_zero_m,
        )
        lateral_score = deadzone_score(
            (m["sole_center_y"][:, swing] - self._lane_y).abs(),
            free_tolerance=lateral_lane_tolerance_m,
            zero_at=lateral_lane_zero_m,
        )
        foot_orientation_score = torch.minimum(
            deadzone_score(
                m["foot_roll"][:, swing].abs(),
                free_tolerance=foot_roll_tolerance_rad,
                zero_at=foot_roll_zero_rad,
            ),
            deadzone_score(
                m["foot_yaw"][:, swing].abs(),
                free_tolerance=foot_yaw_tolerance_rad,
                zero_at=foot_yaw_zero_rad,
            ),
        )
        body_orientation_score = torch.minimum(
            torch.minimum(
                deadzone_score(
                    m["root_roll"].abs(),
                    free_tolerance=root_roll_tolerance_rad,
                    zero_at=root_roll_zero_rad,
                ),
                deadzone_score(
                    m["root_pitch"].abs(),
                    free_tolerance=root_pitch_tolerance_rad,
                    zero_at=root_pitch_zero_rad,
                ),
            ),
            deadzone_score(
                m["root_yaw"].abs(),
                free_tolerance=root_yaw_tolerance_rad,
                zero_at=root_yaw_zero_rad,
            ),
        )

        lift_target = m["riser_height"] + float(lift_clearance_m)
        lift_progress = (
            m["sole_bottom"][:, swing] / lift_target.clamp_min(1.0e-4)
        ).clamp(0.0, 1.0)
        height_corridor = target_band_score(
            m["sole_bottom"][:, swing],
            lower=m["riser_height"] + 0.010,
            upper=m["riser_height"] + float(lift_clearance_m) + 0.030,
            falloff=0.040,
        )

        target_x = float(m["riser_x"]) + 0.5 * float(m["tread_depth"])
        advance_denominator = (target_x - self._start_x).clamp_min(0.05)
        forward_progress = (
            (m["sole_center_x"][:, swing] - self._start_x) / advance_denominator
        ).clamp(0.0, 1.0)

        phase0 = (
            support_quality_lower
            + swing_unload_score
            + pelvis_score
            + body_orientation_score
        ) / 4.0
        phase1 = lift_progress * torch.minimum(
            torch.minimum(lateral_score, foot_orientation_score),
            torch.minimum(support_quality_lower, body_orientation_score),
        )
        phase2 = forward_progress * torch.minimum(
            torch.minimum(height_corridor, lateral_score),
            torch.minimum(
                foot_orientation_score,
                torch.minimum(support_quality_lower, body_orientation_score),
            ),
        )
        phase3 = m["physical_support"][:, swing] * torch.minimum(
            torch.minimum(lateral_score, foot_orientation_score),
            body_orientation_score,
        )
        phase_scores = torch.stack((phase0, phase1, phase2, phase3), dim=1)
        active_phase = self._phase.clamp(max=self._PHASE_COUNT - 1)
        active_score = phase_scores.gather(1, active_phase.unsqueeze(1)).squeeze(1)
        complete = self._phase >= self._PHASE_COUNT
        active_score = torch.where(complete, torch.ones_like(active_score), active_score)

        transition0 = (
            (support_force_fraction >= float(support_force_bodyweight_fraction))
            & (swing_force_fraction <= float(swing_unload_bodyweight_fraction))
            & (pelvis_score >= 0.70)
            & (body_orientation_score >= 0.75)
        )
        transition1 = (
            (m["sole_bottom"][:, swing] >= m["riser_height"] + 0.010)
            & (lateral_score >= 0.75)
            & (foot_orientation_score >= 0.75)
            & (support_quality_lower >= 0.70)
            & (body_orientation_score >= 0.75)
        )
        transition2 = (
            ((m["sole_center_x"][:, swing] - target_x).abs() <= 0.040)
            & (height_corridor >= 0.70)
            & (lateral_score >= 0.75)
            & (foot_orientation_score >= 0.75)
            & (support_quality_lower >= 0.65)
        )
        transition3 = (
            (m["physical_support"][:, swing] >= 0.60)
            & (lateral_score >= 0.75)
            & (foot_orientation_score >= 0.75)
            & (body_orientation_score >= 0.75)
        )
        transition_by_phase = torch.stack(
            (transition0, transition1, transition2, transition3), dim=1
        )
        transition = transition_by_phase.gather(
            1, active_phase.unsqueeze(1)
        ).squeeze(1) & ~complete
        # Compute the potential using the current phase before advancing the
        # state machine. This keeps phase boundaries continuous: a completed
        # phase ends near k/4 and the next step starts at exactly k/4 instead
        # of creating a one-step reward spike from the previous phase score.
        potential = (
            self._phase.to(dtype=active_score.dtype).clamp(max=self._PHASE_COUNT)
            + torch.where(self._phase < self._PHASE_COUNT, active_score, torch.zeros_like(active_score))
        ) / float(self._PHASE_COUNT)
        potential = potential.clamp(0.0, 1.0)

        self._dwell = torch.where(transition, self._dwell + 1, torch.zeros_like(self._dwell))
        dwell_steps = max(1, int(round(float(transition_dwell_s) / float(env.step_dt))))
        advance_phase = (self._dwell >= dwell_steps) & ~complete
        self._phase[advance_phase] += 1
        self._dwell[advance_phase] = 0

        # Expose exact phase/target state for read-only diagnostics.
        env._g1_stair_motion_phase = self._phase.detach().clone()
        env._g1_stair_motion_phase_progress = active_score.detach().clone()
        env._g1_stair_swing_lane_y = self._lane_y.detach().clone()
        env._g1_stair_swing_start_x = self._start_x.detach().clone()
        env._g1_stair_swing_target_x = torch.full_like(self._start_x, target_x)
        env._g1_stair_swing_left = bool(swing_left)

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class swing_motion_guard(_SingleStepRewardBase):
    """Bounded penalty for sideways/high/unstable swing exploits."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._lane_y = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._start_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._lane_y[env_ids] = 0.0
        self._start_x[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        swing_left: bool = True,
        lateral_free_m: float = 0.030,
        lateral_zero_m: float = 0.100,
        maximum_height_above_riser_m: float = 0.070,
        overheight_zero_m: float = 0.180,
        backward_free_m: float = 0.020,
        backward_zero_m: float = 0.100,
        foot_roll_free_rad: float = 0.15,
        foot_roll_zero_rad: float = 0.45,
        foot_yaw_free_rad: float = 0.15,
        foot_yaw_zero_rad: float = 0.45,
        pelvis_support_free_m: float = 0.10,
        pelvis_support_zero_m: float = 0.22,
        support_force_bodyweight_fraction: float = 0.25,
        activation_height_m: float = 0.010,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        upper_support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=upper_support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )
        swing = 0 if bool(swing_left) else 1
        support = 1 - swing
        new = ~self._initialized
        self._lane_y[new] = m["sole_center_y"][new, swing].detach()
        self._start_x[new] = m["sole_center_x"][new, swing].detach()
        self._initialized[new] = True

        bodyweight = m["bodyweight_n"].clamp_min(1.0)
        support_force_fraction = m["vertical_force"][:, support] / bodyweight
        target_x = float(m["riser_x"]) + 0.5 * float(m["tread_depth"])
        support_lift_bad = (
            (m["sole_bottom"][:, support] - 0.010).clamp_min(0.0) / 0.050
        ).clamp(0.0, 1.0)
        swing_active = (
            (m["sole_bottom"][:, swing] >= float(activation_height_m))
            | (m["vertical_force"][:, swing] / bodyweight <= 0.20)
            | (support_lift_bad > 0.0)
        )

        lateral_bad = 1.0 - deadzone_score(
            (m["sole_center_y"][:, swing] - self._lane_y).abs(),
            free_tolerance=lateral_free_m,
            zero_at=lateral_zero_m,
        )
        max_height = m["riser_height"] + float(maximum_height_above_riser_m)
        overheight_bad = (
            (m["sole_bottom"][:, swing] - max_height).clamp_min(0.0)
            / max(float(overheight_zero_m) - float(maximum_height_above_riser_m), 1.0e-6)
        ).clamp(0.0, 1.0)
        backward_bad = (
            (self._start_x - m["sole_center_x"][:, swing] - float(backward_free_m)).clamp_min(0.0)
            / max(float(backward_zero_m) - float(backward_free_m), 1.0e-6)
        ).clamp(0.0, 1.0)
        forward_overshoot_bad = (
            (m["sole_center_x"][:, swing] - (target_x + 0.050)).clamp_min(0.0)
            / 0.150
        ).clamp(0.0, 1.0)
        near_landing = (
            (m["sole_center_x"][:, swing] - target_x).abs() <= 0.050
        ) & (
            (m["sole_bottom"][:, swing] - m["riser_height"]).abs() <= 0.050
        )
        downward_speed = (-m["foot_vertical_velocity"][:, swing]).clamp_min(0.0)
        touchdown_impact_bad = (
            (downward_speed - 0.30).clamp_min(0.0) / 0.70
        ).clamp(0.0, 1.0) * near_landing.to(dtype=downward_speed.dtype)
        foot_roll_bad = 1.0 - deadzone_score(
            m["foot_roll"][:, swing].abs(),
            free_tolerance=foot_roll_free_rad,
            zero_at=foot_roll_zero_rad,
        )
        foot_yaw_bad = 1.0 - deadzone_score(
            m["foot_yaw"][:, swing].abs(),
            free_tolerance=foot_yaw_free_rad,
            zero_at=foot_yaw_zero_rad,
        )
        pelvis_bad = 1.0 - deadzone_score(
            (m["root_y"] - m["sole_center_y"][:, support]).abs(),
            free_tolerance=pelvis_support_free_m,
            zero_at=pelvis_support_zero_m,
        )
        support_loss = (
            1.0 - support_force_fraction / max(float(support_force_bodyweight_fraction), 1.0e-6)
        ).clamp(0.0, 1.0)
        both_airborne = (
            (m["vertical_force"][:, 0] < 5.0)
            & (m["vertical_force"][:, 1] < 5.0)
            & (m["sole_bottom"][:, 0] > 0.010)
            & (m["sole_bottom"][:, 1] > 0.010)
        ).to(dtype=lateral_bad.dtype)

        severity = torch.stack(
            (
                lateral_bad,
                overheight_bad,
                backward_bad,
                forward_overshoot_bad,
                touchdown_impact_bad,
                support_lift_bad,
                foot_roll_bad,
                foot_yaw_bad,
                pelvis_bad,
                support_loss,
                both_airborne,
            ),
            dim=1,
        ).amax(dim=1)
        return severity * swing_active.to(dtype=severity.dtype)


def body_heading_guard(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    free_yaw_rad: float = 0.10,
    zero_at_yaw_rad: float = 0.35,
) -> torch.Tensor:
    """Penalty severity for turning away from the stair heading."""

    robot = env.scene[asset_cfg.name]
    _, _, yaw = _quat_to_euler_wxyz(robot.data.root_quat_w)
    return 1.0 - deadzone_score(
        yaw.abs(), free_tolerance=free_yaw_rad, zero_at=zero_at_yaw_rad
    )


class swing_lift_progress(_PotentialDifferenceTerm):
    """Reward a supported single-foot lift up to the current riser height.

    The opposite foot must remain load-bearing on the lower floor, which
    prevents a two-foot jump from satisfying the lift precursor.  The
    potential hands off continuously to upper-tread support so touchdown does
    not create a negative shaping spike.
    """

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        lower_support_height_tolerance_m: float = 0.020,
        lower_support_force_bodyweight_fraction: float = 0.30,
        maximum_lower_support_foot_speed_mps: float = 0.20,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if lower_support_height_tolerance_m <= 0.0:
            raise ValueError(
                "lower_support_height_tolerance_m must be positive"
            )
        if not 0.0 < lower_support_force_bodyweight_fraction <= 1.0:
            raise ValueError(
                "lower_support_force_bodyweight_fraction must be in (0, 1]"
            )
        if maximum_lower_support_foot_speed_mps <= 0.0:
            raise ValueError(
                "maximum_lower_support_foot_speed_mps must be positive"
            )

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )

        # Lift to the step-top height first.  The existing coupled swing term
        # subsequently asks for the additional clearance margin and forward
        # insertion.  Reaching the tread height therefore earns a dense,
        # bounded precursor without requiring horizontal motion first.
        supported_lift = supported_single_foot_lift_score(
            sole_bottom_m=m["sole_bottom"],
            vertical_force_n=m["vertical_force"],
            foot_speed_mps=m["foot_speed"],
            bodyweight_n=m["bodyweight_n"],
            riser_height_m=m["riser_height"],
            lower_support_height_tolerance_m=(
                lower_support_height_tolerance_m
            ),
            lower_support_force_bodyweight_fraction=(
                lower_support_force_bodyweight_fraction
            ),
            maximum_lower_support_foot_speed_mps=(
                maximum_lower_support_foot_speed_mps
            ),
        )

        # Once the lifted foot acquires the upper tread, preserve the potential
        # through touchdown and later body transfer.
        upper_support = m["physical_support"].amax(dim=1)
        potential = torch.maximum(supported_lift, upper_support)

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class swing_clearance_progress(_PotentialDifferenceTerm):
    """Couple swing-foot vertical clearance with forward advance."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        activation_distance_m: float = 0.30,
        insertion_target_m: float = 0.10,
        clearance_margin_m: float = 0.015,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )

        flight = coupled_swing_score(
            sole_bottom_m=m["sole_bottom"],
            front_x_m=m["front_x"],
            riser_x_m=float(m["riser_x"]),
            riser_height_m=m["riser_height"].unsqueeze(1),
            activation_distance_m=activation_distance_m,
            insertion_target_m=insertion_target_m,
            clearance_margin_m=clearance_margin_m,
        )

        # Make the potential continuous through touchdown: once a foot has
        # actually acquired good upper support, lowering from swing clearance
        # onto the tread must not create a negative shaping spike.
        foot_progress = torch.maximum(
            flight,
            m["physical_support"],
        )
        potential = foot_progress.amax(dim=1)

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class upper_foothold_progress(_PotentialDifferenceTerm):
    """Reward acquiring and maintaining one physically stable upper foothold."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )
        potential = m["physical_support"].amax(dim=1)

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class body_transfer_progress(_PotentialDifferenceTerm):
    """Reward COM/root transfer only while an upper foothold is load-bearing."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._initial_root_z = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self._height_initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initial_root_z[env_ids] = 0.0
        self._height_initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        transfer_start_before_riser_m: float = 0.12,
        transfer_target_past_riser_m: float = 0.10,
        required_height_fraction: float = 0.80,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        if not 0.0 < required_height_fraction <= 1.5:
            raise ValueError("required_height_fraction must be in (0, 1.5]")

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )

        new = ~self._height_initialized
        self._initial_root_z[new] = m["root_z"][new].detach()
        self._height_initialized[new] = True

        x_start = float(m["riser_x"]) - float(transfer_start_before_riser_m)
        x_target = float(m["riser_x"]) + float(transfer_target_past_riser_m)
        x_fraction = (
            (m["root_x"] - x_start)
            / max(x_target - x_start, 1.0e-6)
        ).clamp(0.0, 1.0)

        height_target = (
            m["riser_height"] * float(required_height_fraction)
        ).clamp_min(1.0e-4)
        z_fraction = (
            (m["root_z"] - self._initial_root_z)
            / height_target
        ).clamp(0.0, 1.0)

        upper_support = m["physical_support"].amax(dim=1)
        potential = upper_support * 0.5 * (x_fraction + z_fraction)

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class trailing_foot_recovery(_PotentialDifferenceTerm):
    """Reward recovering the second foot onto the upper tread."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._initial_root_z = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self._height_initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initial_root_z[env_ids] = 0.0
        self._height_initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        required_height_fraction: float = 0.70,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )

        new = ~self._height_initialized
        self._initial_root_z[new] = m["root_z"][new].detach()
        self._height_initialized[new] = True

        height_target = (
            m["riser_height"] * float(required_height_fraction)
        ).clamp_min(1.0e-4)
        z_fraction = (
            (m["root_z"] - self._initial_root_z)
            / height_target
        ).clamp(0.0, 1.0)
        both_support = m["physical_support"].amin(dim=1)
        potential = both_support * z_fraction

        return self._potential_rate(
            potential,
            dt=float(env.step_dt),
            valid_mask=m["task_safe"],
        )


class riser_strike_event(_SingleStepRewardBase):
    """Emit one event for each low horizontal foot impact at the riser face."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._active = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._active[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        face_band_m: float = 0.025,
        clearance_margin_m: float = 0.010,
        horizontal_force_bodyweight_fraction: float = 0.05,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
    ) -> torch.Tensor:
        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=minimum_upright_gravity_z,
        )

        riser_x = float(m["riser_x"])
        near_face = (
            (m["front_x"] >= riser_x - float(face_band_m))
            & (m["rear_x"] <= riser_x + float(face_band_m))
        )
        too_low = m["sole_bottom"] < (
            m["riser_height"].unsqueeze(1) + float(clearance_margin_m)
        )
        if not 0.0 < float(horizontal_force_bodyweight_fraction) <= 1.0:
            raise ValueError(
                "horizontal_force_bodyweight_fraction must be in (0, 1]"
            )
        horizontal_threshold = (
            m["bodyweight_n"].unsqueeze(1)
            * float(horizontal_force_bodyweight_fraction)
        )
        horizontal_impact = (
            m["horizontal_force_x"] >= horizontal_threshold
        )
        current = (near_face & too_low & horizontal_impact).any(dim=1)
        onset = current & ~self._active
        self._active.copy_(current)
        return onset.to(m["root_x"].dtype) / float(env.step_dt)


class single_step_success(_SingleStepRewardBase):
    """One-time bonus after both feet and the body settle on the upper level."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._initial_root_z = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self._initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._dwell_s = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )
        self._triggered = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initial_root_z[env_ids] = 0.0
        self._initialized[env_ids] = False
        self._dwell_s[env_ids] = 0.0
        self._triggered[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        minimum_support_score: float = 0.60,
        base_past_riser_margin_m: float = 0.05,
        required_height_fraction: float = 0.75,
        support_minimum_upright_gravity_z: float = 0.80,
        minimum_upright_gravity_z: float = 0.90,
        maximum_root_linear_speed_mps: float = 0.35,
        maximum_root_angular_speed_radps: float = 0.60,
        minimum_dwell_s: float = 0.12,
        tread_margin_m: float = 0.015,
        foot_height_tolerance_m: float = 0.025,
        support_force_bodyweight_fraction: float = 0.15,
        maximum_supported_foot_speed_mps: float = 0.15,
    ) -> torch.Tensor:
        if minimum_dwell_s <= 0.0:
            raise ValueError("minimum_dwell_s must be positive")

        m = self._measure(
            env,
            asset_cfg=asset_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            tread_margin_m=tread_margin_m,
            foot_height_tolerance_m=foot_height_tolerance_m,
            support_force_bodyweight_fraction=support_force_bodyweight_fraction,
            maximum_supported_foot_speed_mps=maximum_supported_foot_speed_mps,
            minimum_upright_gravity_z=float(
                support_minimum_upright_gravity_z
            ),
        )

        new = ~self._initialized
        self._initial_root_z[new] = m["root_z"][new].detach()
        self._initialized[new] = True

        height_ok = (
            m["root_z"] - self._initial_root_z
            >= m["riser_height"] * float(required_height_fraction)
        )
        both_support = (
            m["support"] >= float(minimum_support_score)
        ).all(dim=1)
        candidate = (
            both_support
            & (
                m["root_x"]
                >= float(m["riser_x"]) + float(base_past_riser_margin_m)
            )
            & height_ok
            & (m["upright"] >= float(minimum_upright_gravity_z))
            & (m["root_lin_speed"] <= float(maximum_root_linear_speed_mps))
            & (m["root_ang_speed"] <= float(maximum_root_angular_speed_radps))
            & ~m["forbidden_contact"]
        )

        dt = float(env.step_dt)
        self._dwell_s = torch.where(
            candidate,
            self._dwell_s + dt,
            torch.zeros_like(self._dwell_s),
        )
        completed = (
            self._dwell_s + 1.0e-12 >= float(minimum_dwell_s)
        )
        event = completed & ~self._triggered
        self._triggered |= event
        return event.to(m["root_x"].dtype) / dt
