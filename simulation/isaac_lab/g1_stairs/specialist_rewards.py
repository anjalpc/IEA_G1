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
    G1_SOLE_SPHERE_CENTRES_M,
    G1_SOLE_SPHERE_RADIUS_M,
)
from .reward_math import (
    coupled_swing_score,
    support_quality,
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
        sole_bottom = (
            sole_centres_local[..., 2] - G1_SOLE_SPHERE_RADIUS_M
        ).amin(dim=2)

        foot_speed = torch.linalg.vector_norm(
            foot_lin_vel_w,
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
            "sole_bottom": sole_bottom,
            "foot_speed": foot_speed,
            "vertical_force": vertical_force,
            "bodyweight_n": self._bodyweight_n,
            "horizontal_force_x": horizontal_force_x,
            "support": support,
            "physical_support": physical_support,
            "root_x": local_root[:, 0],
            "root_z": local_root[:, 2],
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
