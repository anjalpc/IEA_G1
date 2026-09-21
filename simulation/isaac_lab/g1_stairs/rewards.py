"""Minimal Phase 12 rewards for the fixed nominal staircase."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_progress(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return forward world-X velocity; RewardManager integrates it over dt."""
    robot = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_w[:, 0]


class stair_progress(ManagerTermBase):
    """Reward each newly reached stair level exactly once per episode."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._highest_level = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._highest_level[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        step_edges_m: tuple[float, ...] = (1.0, 1.25, 1.50),
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        progress_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        edges = torch.as_tensor(
            step_edges_m,
            dtype=progress_x.dtype,
            device=progress_x.device,
        )
        level = torch.sum(progress_x.unsqueeze(-1) >= edges, dim=-1)
        newly_reached = torch.clamp(level - self._highest_level, min=0)
        self._highest_level = torch.maximum(self._highest_level, level)

        # RewardManager multiplies every term by step_dt. Divide discrete
        # transitions by step_dt so one newly reached level contributes 1.0.
        return newly_reached.to(progress_x.dtype) / env.step_dt


class progress_threshold_event(ManagerTermBase):
    """Reward crossing a forward-progress threshold once per episode."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._triggered = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._triggered[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold_m: float = 1.75,
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        progress_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]

        reached = progress_x >= threshold_m
        newly_reached = reached & ~self._triggered
        self._triggered |= reached

        return newly_reached.to(progress_x.dtype) / env.step_dt


def stair_swing_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    riser_edges_m: tuple[float, ...] = (1.0, 1.25, 1.50),
    tread_heights_m: tuple[float, ...] = (0.20, 0.40, 0.60),
    activation_distance_m: float = 0.20,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.05,
    std: float = 0.01,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    """Reward a moving swing foot for clearing the upcoming stair riser.

    Unlike Unitree's flat-ground foot-clearance reward, this term uses
    robot-relative stair geometry. Each riser gets its own required world-Z
    target equal to the next tread height plus a small clearance margin.

    Reward is active only shortly before and over each riser lip. Overshooting
    the requested clearance is not penalized.
    """
    if len(riser_edges_m) != len(tread_heights_m):
        raise ValueError(
            "riser_edges_m and tread_heights_m must have equal length"
        )

    robot = env.scene[asset_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    foot_x = (
        foot_pos_w[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )
    foot_z = (
        foot_pos_w[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    horizontal_speed = torch.linalg.norm(
        foot_vel_w[:, :, :2],
        dim=2,
    )
    swing_gate = torch.tanh(tanh_mult * horizontal_speed)

    reward = torch.zeros(
        env.num_envs,
        dtype=foot_z.dtype,
        device=env.device,
    )

    for edge_m, tread_height_m in zip(
        riser_edges_m,
        tread_heights_m,
        strict=True,
    ):
        active = (
            (foot_x >= edge_m - activation_distance_m)
            & (foot_x <= edge_m + lip_distance_m)
        )

        target_z = tread_height_m + clearance_margin_m
        shortfall = torch.clamp(target_z - foot_z, min=0.0)

        clearance_reward = torch.exp(
            -torch.square(shortfall) / std
        )

        reward += torch.sum(
            clearance_reward
            * swing_gate
            * active.to(foot_z.dtype),
            dim=1,
        )

    return reward


def anticipatory_stair_swing_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    riser_edges_m: tuple[float, ...] = (1.0, 1.25, 1.50),
    tread_heights_m: tuple[float, ...] = (0.20, 0.40, 0.60),
    lookahead_distance_m: float = 0.50,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.05,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    """Penalize insufficient swing-foot height while approaching each riser.

    The desired foot height ramps smoothly from the current tread height to
    the upcoming tread height plus a clearance margin. The term activates
    before the riser so PPO receives a dense learning signal before impact.

    The support foot is naturally suppressed because its world-frame
    horizontal speed is near zero, while the swing foot has non-zero speed.
    """
    if len(riser_edges_m) != len(tread_heights_m):
        raise ValueError(
            "riser_edges_m and tread_heights_m must have equal length"
        )

    robot = env.scene[asset_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    foot_x = (
        foot_pos_w[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )
    foot_z = (
        foot_pos_w[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    horizontal_speed = torch.linalg.norm(
        foot_vel_w[:, :, :2],
        dim=2,
    )
    swing_gate = torch.tanh(
        tanh_mult * horizontal_speed
    )

    reward = torch.zeros(
        env.num_envs,
        dtype=foot_z.dtype,
        device=env.device,
    )

    previous_tread_height = 0.0

    for edge_m, tread_height_m in zip(
        riser_edges_m,
        tread_heights_m,
        strict=True,
    ):
        start_x = edge_m - lookahead_distance_m

        active = (
            (foot_x >= start_x)
            & (foot_x <= edge_m + lip_distance_m)
        )

        approach_phase = torch.clamp(
            (foot_x - start_x) / lookahead_distance_m,
            min=0.0,
            max=1.0,
        )

        target_at_riser = (
            tread_height_m + clearance_margin_m
        )

        desired_z = (
            previous_tread_height
            + approach_phase
            * (target_at_riser - previous_tread_height)
        )

        shortfall = torch.clamp(
            desired_z - foot_z,
            min=0.0,
        )

        # Dense penalty: a foot 0.10 m below the requested trajectory
        # contributes -0.10 before weighting. This avoids the vanishing
        # signal of the localized exponential reward used in Pretrained-B.
        reward -= torch.sum(
            shortfall
            * swing_gate
            * active.to(foot_z.dtype),
            dim=1,
        )

        previous_tread_height = tread_height_m

    return reward


def contact_gated_stair_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    riser_edges_m: tuple[float, ...] = (1.0, 1.25, 1.50),
    tread_heights_m: tuple[float, ...] = (0.20, 0.40, 0.60),
    nominal_ankle_height_m: float = 0.10,
    lookahead_distance_m: float = 0.35,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.05,
) -> torch.Tensor:
    """Penalize low height only for the genuinely airborne foot near a riser.

    The ankle target is referenced to its nominal flat-ground world height,
    then shifted upward by the upcoming tread height and a clearance margin.
    The stance foot receives no clearance shaping.
    """
    if len(riser_edges_m) != len(tread_heights_m):
        raise ValueError(
            "riser_edges_m and tread_heights_m must have equal length"
        )

    robot = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]

    foot_x = (
        foot_pos_w[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )
    foot_z = (
        foot_pos_w[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    is_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    )
    is_swing = ~is_contact

    penalty = torch.zeros(
        env.num_envs,
        dtype=foot_z.dtype,
        device=env.device,
    )

    # Prevent overlapping stair windows from penalizing one foot for
    # multiple risers simultaneously.
    assigned = torch.zeros_like(is_swing)

    previous_tread_height = 0.0

    for edge_m, tread_height_m in zip(
        riser_edges_m,
        tread_heights_m,
        strict=True,
    ):
        start_x = edge_m - lookahead_distance_m

        active = (
            is_swing
            & ~assigned
            & (foot_x >= start_x)
            & (foot_x <= edge_m + lip_distance_m)
        )

        phase = torch.clamp(
            (foot_x - start_x) / lookahead_distance_m,
            min=0.0,
            max=1.0,
        )

        start_target = (
            nominal_ankle_height_m + previous_tread_height
        )
        end_target = (
            nominal_ankle_height_m
            + tread_height_m
            + clearance_margin_m
        )

        desired_z = (
            start_target
            + phase * (end_target - start_target)
        )

        shortfall = torch.clamp(
            desired_z - foot_z,
            min=0.0,
        )

        penalty -= torch.sum(
            shortfall * active.to(foot_z.dtype),
            dim=1,
        )

        assigned |= active
        previous_tread_height = tread_height_m

    return penalty


def validated_tread_contact(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    first_riser_x_m: float = 1.0,
    tread_depth_m: float = 0.25,
    step_count: int = 3,
    nominal_ankle_height_m: float = 0.10,
    x_margin_m: float = 0.02,
    z_tolerance_m: float = 0.04,
    upright_min: float = 0.80,
) -> torch.Tensor:
    """Reward real foot contact on progressively higher stair treads.

    A step counts only when:
    - an ankle is physically in contact,
    - its X position lies on that tread,
    - its ankle height matches the tread surface,
    - the robot remains upright.

    Returns 0, 1/3, 2/3, or 1 for a three-step staircase.
    """

    robot = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    terrain_generator = env.scene.terrain.cfg.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]

    riser_height_m = float(stair_cfg.riser_height_range[0])

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]

    foot_x = (
        foot_pos_w[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )
    foot_z = (
        foot_pos_w[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    is_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
        > 0.0
    )

    upright = (
        -robot.data.projected_gravity_b[:, 2]
        >= upright_min
    )

    validated_level = torch.zeros(
        env.num_envs,
        dtype=foot_z.dtype,
        device=env.device,
    )

    for step_index in range(1, step_count + 1):
        tread_start = (
            first_riser_x_m
            + (step_index - 1) * tread_depth_m
        )
        tread_end = tread_start + tread_depth_m

        expected_ankle_z = (
            nominal_ankle_height_m
            + step_index * riser_height_m
        )

        on_tread_x = (
            (foot_x >= tread_start + x_margin_m)
            & (foot_x <= tread_end - x_margin_m)
        )

        at_tread_height = (
            torch.abs(foot_z - expected_ankle_z)
            <= z_tolerance_m
        )

        valid_contact = (
            is_contact
            & on_tread_x
            & at_tread_height
        ).any(dim=1)

        level_value = torch.full_like(
            validated_level,
            float(step_index),
        )

        validated_level = torch.where(
            valid_contact,
            torch.maximum(validated_level, level_value),
            validated_level,
        )

    validated_level = (
        validated_level / float(step_count)
    )

    return validated_level * upright.to(foot_z.dtype)
