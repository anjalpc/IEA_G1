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



def curriculum_riser_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    first_riser_x_m: float = 1.0,
    nominal_ankle_height_m: float = 0.10,
    lookahead_distance_m: float = 0.45,
    forward_margin_m: float = 0.06,
    clearance_margin_m: float = 0.025,
    height_sigma_m: float = 0.035,
    minimum_horizontal_speed_mps: float = 0.05,
) -> torch.Tensor:
    """Dense swing-foot target immediately around the next riser.

    Uses the live curriculum terrain height instead of nominal 20 cm
    coordinates. Only activates near the first riser so ordinary walking
    remains governed by the transferred Unitree locomotion prior.
    """
    robot = env.scene[asset_cfg.name]

    terrain_generator = env.scene.terrain.cfg.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]
    riser_height_m = float(stair_cfg.riser_height_range[0])

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    horizontal_speed = torch.linalg.vector_norm(
        foot_vel[:, :, :2],
        dim=2,
    )

    target_z = (
        nominal_ankle_height_m
        + riser_height_m
        + clearance_margin_m
    )

    error = foot_z - target_z

    height_score = torch.exp(
        -(error * error)
        / max(height_sigma_m * height_sigma_m, 1.0e-6)
    )

    moving = (
        horizontal_speed
        >= minimum_horizontal_speed_mps
    ).to(height_score.dtype)

    # Whichever foot is currently swinging may earn the shaping reward.
    swing_score = torch.max(
        height_score * moving,
        dim=1,
    ).values

    near_riser = (
        (root_x >= first_riser_x_m - lookahead_distance_m)
        & (root_x <= first_riser_x_m + forward_margin_m)
    )

    upright = torch.clamp(
        -robot.data.projected_gravity_b[:, 2],
        min=0.0,
        max=1.0,
    )

    return (
        swing_score
        * near_riser.to(swing_score.dtype)
        * upright
    )


def curriculum_foot_over_tread(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    first_riser_x_m: float = 1.0,
    nominal_ankle_height_m: float = 0.10,
    x_margin_m: float = 0.01,
    clearance_margin_m: float = 0.01,
    upright_min: float = 0.80,
) -> torch.Tensor:
    """Reward getting either ankle physically above the first tread.

    This is deliberately weaker than validated tread contact. It supplies a
    gradient between clearing the lip and actually loading the upper tread.
    """
    robot = env.scene[asset_cfg.name]

    terrain_generator = env.scene.terrain.cfg.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]

    riser_height_m = float(stair_cfg.riser_height_range[0])
    tread_depth_m = float(stair_cfg.tread_depth)

    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )
    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    tread_start = first_riser_x_m + x_margin_m
    tread_end = (
        first_riser_x_m
        + tread_depth_m
        - x_margin_m
    )

    minimum_z = (
        nominal_ankle_height_m
        + riser_height_m
        + clearance_margin_m
    )

    above_tread = (
        (foot_x >= tread_start)
        & (foot_x <= tread_end)
        & (foot_z >= minimum_z)
    ).any(dim=1)

    upright = (
        -robot.data.projected_gravity_b[:, 2]
        >= upright_min
    )

    return (
        above_tread
        & upright
    ).to(foot_z.dtype)



def pre_riser_foot_lift(
    env,
    asset_cfg,
    foot_body_names,
    first_riser_x_m: float = 1.0,
    nominal_ankle_height_m: float = 0.10,
    lookahead_m: float = 0.45,
    safety_clearance_m: float = 0.06,
    min_forward_speed_mps: float = 0.03,
    vertical_speed_scale_mps: float = 0.30,
):
    """Dense distance-conditioned anticipatory swing-foot lift reward.

    Unlike the previous formulation, this reward does not wait for the foot
    to already be high.

    As a forward-moving foot approaches the riser it receives reward for:
      1. positive vertical velocity, which teaches *when to start lifting*;
      2. actual foot height, which teaches *how high to lift*.

    The reward becomes stronger as the foot gets closer to the riser.
    """
    import torch

    robot = env.scene[asset_cfg.name]

    terrain_generator = env.scene.terrain.cfg.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]
    riser_height = float(stair_cfg.riser_height_range[0])

    body_ids = robot.find_bodies(foot_body_names)[0]

    foot_pos = robot.data.body_pos_w[:, body_ids, :]
    foot_vel = robot.data.body_lin_vel_w[:, body_ids, :]

    origin = env.scene.env_origins

    foot_x = foot_pos[..., 0] - origin[:, None, 0]
    foot_z = foot_pos[..., 2] - origin[:, None, 2]

    # Positive while the foot is still behind the riser.
    distance_to_riser = first_riser_x_m - foot_x

    in_approach_window = (
        (distance_to_riser > 0.0)
        & (distance_to_riser <= lookahead_m)
    )

    moving_forward = (
        foot_vel[..., 0] >= min_forward_speed_mps
    )

    # Stronger shaping as the foot approaches the vertical face.
    proximity = torch.clamp(
        1.0 - distance_to_riser / max(lookahead_m, 1.0e-6),
        0.0,
        1.0,
    )

    # Dense signal: reward upward swing velocity immediately.
    upward_velocity_score = torch.clamp(
        foot_vel[..., 2] / max(vertical_speed_scale_mps, 1.0e-6),
        0.0,
        1.0,
    )

    # Height target adapts to the live curriculum stair height.
    safe_z = (
        nominal_ankle_height_m
        + riser_height
        + safety_clearance_m
    )

    lift_start_z = nominal_ankle_height_m + 0.005

    height_score = torch.clamp(
        (foot_z - lift_start_z)
        / max(safe_z - lift_start_z, 1.0e-6),
        0.0,
        1.0,
    )

    # Early learning is driven mainly by upward velocity.
    # Height becomes increasingly important near the riser.
    lift_score = (
        0.65 * upward_velocity_score
        + 0.35 * height_score
    )

    per_foot = (
        in_approach_window.float()
        * moving_forward.float()
        * proximity
        * lift_score
    )

    projected_gravity = robot.data.projected_gravity_b

    upright = torch.clamp(
        (-projected_gravity[:, 2] - 0.75) / 0.25,
        0.0,
        1.0,
    )

    return torch.max(per_foot, dim=1).values * upright


def riser_crossing_clearance(
    env,
    asset_cfg,
    foot_body_names,
    first_riser_x_m: float = 1.0,
    nominal_ankle_height_m: float = 0.10,
    crossing_before_m: float = 0.04,
    crossing_after_m: float = 0.06,
    safety_clearance_m: float = 0.06,
    min_forward_speed_mps: float = 0.03,
):
    """Reward a foot crossing the riser plane with safe vertical clearance."""
    import torch

    robot = env.scene[asset_cfg.name]

    terrain_generator = env.scene.terrain.cfg.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]
    riser_height = float(stair_cfg.riser_height_range[0])

    body_ids = robot.find_bodies(foot_body_names)[0]

    foot_pos = robot.data.body_pos_w[:, body_ids, :]
    foot_vel = robot.data.body_lin_vel_w[:, body_ids, :]

    origin = env.scene.env_origins

    foot_x = foot_pos[..., 0] - origin[:, None, 0]
    foot_z = foot_pos[..., 2] - origin[:, None, 2]

    safe_z = (
        nominal_ankle_height_m
        + riser_height
        + safety_clearance_m
    )

    near_plane = (
        (foot_x >= first_riser_x_m - crossing_before_m)
        & (foot_x <= first_riser_x_m + crossing_after_m)
    )

    moving_forward = (
        foot_vel[..., 0] >= min_forward_speed_mps
    )

    # 0 below safe height, 1 once full safety clearance is achieved.
    clearance_score = torch.clamp(
        (
            foot_z
            - (
                nominal_ankle_height_m
                + riser_height
            )
        )
        / max(safety_clearance_m, 1.0e-6),
        0.0,
        1.0,
    )

    per_foot = (
        near_plane.float()
        * moving_forward.float()
        * clearance_score
    )

    return torch.max(per_foot, dim=1).values


def riser_face_strike_proxy(
    env,
    asset_cfg,
    foot_body_names,
    first_riser_x_m: float = 1.0,
    nominal_ankle_height_m: float = 0.10,
    face_window_m: float = 0.035,
    minimum_clearance_m: float = 0.025,
    min_forward_speed_mps: float = 0.03,
):
    """Penalize a moving foot reaching the riser plane while still too low.

    This is intentionally a geometric proxy for a vertical-face strike.
    It does not depend on PhysX contact classification.
    """
    import torch

    robot = env.scene[asset_cfg.name]

    terrain_generator = env.scene.terrain.cfg.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]
    riser_height = float(stair_cfg.riser_height_range[0])

    body_ids = robot.find_bodies(foot_body_names)[0]

    foot_pos = robot.data.body_pos_w[:, body_ids, :]
    foot_vel = robot.data.body_lin_vel_w[:, body_ids, :]

    origin = env.scene.env_origins

    foot_x = foot_pos[..., 0] - origin[:, None, 0]
    foot_z = foot_pos[..., 2] - origin[:, None, 2]

    required_z = (
        nominal_ankle_height_m
        + riser_height
        + minimum_clearance_m
    )

    at_face = (
        torch.abs(foot_x - first_riser_x_m)
        <= face_window_m
    )

    too_low = foot_z < required_z

    moving_forward = (
        foot_vel[..., 0] >= min_forward_speed_mps
    )

    strike = (
        at_face
        & too_low
        & moving_forward
    ).float()

    # Return positive magnitude. RewTerm weight will be negative.
    return torch.max(strike, dim=1).values

def validated_tread_contact(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    first_riser_x_m: float = 1.0,
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
    tread_depth_m = float(stair_cfg.tread_depth)
    step_count = int(stair_cfg.step_count)

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


class _HalfwayTransitionRuntime:
    """One shared FIRST_SUPPORT state machine for all 021 reward terms."""

    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        minimum_support_duration_s: float,
    ):
        import math

        self.required_support_steps = max(
            1,
            math.ceil(
                float(minimum_support_duration_s)
                / env.step_dt
            ),
        )

        self.state = torch.full(
            (env.num_envs,),
            1,  # FIRST_SUPPORT
            dtype=torch.long,
            device=env.device,
        )

        self.support_steps = torch.zeros(
            (env.num_envs, 2),
            dtype=torch.long,
            device=env.device,
        )

        self.both_support_start_x = torch.zeros(
            env.num_envs,
            dtype=torch.float,
            device=env.device,
        )

        self.last_step_token = None

        self.events = {
            "base_above": torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=env.device,
            ),
            "both_feet_supported": torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=env.device,
            ),
            "completed": torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=env.device,
            ),
        }

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self.state[env_ids] = 1
        self.support_steps[env_ids] = 0
        self.both_support_start_x[env_ids] = 0.0

        for event in self.events.values():
            event[env_ids] = False

        self.last_step_token = None

    def update(
        self,
        env: "ManagerBasedRLEnv",
        *,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        forbidden_sensor_cfg: SceneEntityCfg,
        first_riser_x_m: float,
        tread_depth_m: float,
        riser_height_m: float,
        nominal_ankle_height_m: float,
        minimum_vertical_support_force_n: float,
        maximum_supported_foot_speed_mps: float,
        minimum_upright_gravity_z: float,
        completion_forward_distance_m: float,
    ) -> None:
        # RewardManager calls three transition reward terms during the same
        # environment step. Compute the physical state machine once only.
        step_token = int(env.common_step_counter)

        if self.last_step_token == step_token:
            return

        self.last_step_token = step_token

        robot = env.scene[asset_cfg.name]
        sensor = env.scene.sensors[sensor_cfg.name]

        foot_pos = robot.data.body_pos_w[
            :, asset_cfg.body_ids
        ]

        if hasattr(robot.data, "body_link_lin_vel_w"):
            body_velocity = robot.data.body_link_lin_vel_w[
                :, asset_cfg.body_ids
            ]
        else:
            body_velocity = robot.data.body_lin_vel_w[
                :, asset_cfg.body_ids
            ]

        origins = env.scene.env_origins

        foot_x = (
            foot_pos[:, :, 0]
            - origins[:, 0].unsqueeze(1)
        )

        foot_z = (
            foot_pos[:, :, 2]
            - origins[:, 2].unsqueeze(1)
        )

        force_z = sensor.data.net_forces_w[
            :, sensor_cfg.body_ids, 2
        ].clamp(min=0)

        speed = torch.linalg.vector_norm(
            body_velocity,
            dim=2,
        )

        upright = (
            -robot.data.projected_gravity_b[:, 2]
            >= minimum_upright_gravity_z
        )

        forbidden_force = sensor.data.net_forces_w[
            :, forbidden_sensor_cfg.body_ids
        ]

        forbidden = (
            torch.linalg.vector_norm(
                forbidden_force,
                dim=2,
            )
            .ge(5.0)
            .any(dim=1)
        )

        support_candidate = (
            (
                foot_x
                >= first_riser_x_m + 0.02
            )
            & (
                foot_x
                <= first_riser_x_m
                + tread_depth_m
                - 0.02
            )
            & (
                torch.abs(
                    foot_z
                    - (
                        nominal_ankle_height_m
                        + riser_height_m
                    )
                )
                <= 0.025
            )
            & (
                force_z
                >= minimum_vertical_support_force_n
            )
            & (
                speed
                <= maximum_supported_foot_speed_mps
            )
            & upright.unsqueeze(1)
            & (~forbidden).unsqueeze(1)
        )

        self.support_steps = torch.where(
            support_candidate,
            self.support_steps + 1,
            torch.zeros_like(self.support_steps),
        )

        supported = (
            self.support_steps
            >= self.required_support_steps
        )

        root_x = (
            robot.data.root_pos_w[:, 0]
            - origins[:, 0]
        )

        root_z = (
            robot.data.root_pos_w[:, 2]
            - origins[:, 2]
        )

        valid = upright & ~forbidden

        # IMPORTANT:
        # All events are calculated from the state at the beginning of this
        # control step. Therefore an environment can advance at most one
        # physical state per step.
        previous_state = self.state.clone()

        entered_base = (
            (previous_state == 1)
            & valid
            & (
                root_x
                >= first_riser_x_m + 0.02
            )
            & (root_z >= 0.825)
        )

        entered_both = (
            (previous_state == 2)
            & valid
            & supported.all(dim=1)
        )

        entered_complete = (
            (previous_state == 3)
            & valid
            & supported.all(dim=1)
            & (
                (
                    root_x
                    - self.both_support_start_x
                )
                >= completion_forward_distance_m
            )
        )

        self.state = torch.where(
            entered_base,
            torch.full_like(self.state, 2),
            self.state,
        )

        self.both_support_start_x = torch.where(
            entered_both,
            root_x,
            self.both_support_start_x,
        )

        self.state = torch.where(
            entered_both,
            torch.full_like(self.state, 3),
            self.state,
        )

        self.state = torch.where(
            entered_complete,
            torch.full_like(self.state, 4),
            self.state,
        )

        self.events["base_above"] = entered_base
        self.events["both_feet_supported"] = entered_both
        self.events["completed"] = entered_complete


def _shared_halfway_runtime(
    env: "ManagerBasedRLEnv",
    minimum_support_duration_s: float,
) -> _HalfwayTransitionRuntime:
    """Return the single 021 physical state machine owned by the environment."""

    name = "_g1_halfway_transition_runtime"

    runtime = getattr(env, name, None)

    if runtime is None:
        runtime = _HalfwayTransitionRuntime(
            env,
            minimum_support_duration_s,
        )
        setattr(env, name, runtime)

    return runtime


class halfway_step_transition(ManagerTermBase):
    """Pay one transition from the shared 021 physical state machine."""

    _TRANSITIONS = {
        "base_above",
        "both_feet_supported",
        "completed",
    }

    def __init__(
        self,
        cfg,
        env: "ManagerBasedRLEnv",
    ):
        super().__init__(cfg, env)

        self._minimum_support_duration_s = float(
            cfg.params.get(
                "minimum_support_duration_s",
                0.15,
            )
        )

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        runtime = getattr(
            self._env,
            "_g1_halfway_transition_runtime",
            None,
        )

        if runtime is not None:
            runtime.reset(env_ids)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        forbidden_sensor_cfg: SceneEntityCfg,
        transition: str,
        first_riser_x_m: float = 1.0,
        tread_depth_m: float = 0.25,
        riser_height_m: float = 0.05,
        nominal_ankle_height_m: float = 0.10,
        minimum_vertical_support_force_n: float = 42.87,
        maximum_supported_foot_speed_mps: float = 0.15,
        minimum_upright_gravity_z: float = 0.80,
        minimum_support_duration_s: float = 0.15,
        completion_forward_distance_m: float = 0.40,
    ) -> torch.Tensor:
        if transition not in self._TRANSITIONS:
            raise ValueError(
                f"unknown halfway transition: {transition}"
            )

        runtime = _shared_halfway_runtime(
            env,
            minimum_support_duration_s,
        )

        runtime.update(
            env,
            asset_cfg=asset_cfg,
            sensor_cfg=sensor_cfg,
            forbidden_sensor_cfg=forbidden_sensor_cfg,
            first_riser_x_m=first_riser_x_m,
            tread_depth_m=tread_depth_m,
            riser_height_m=riser_height_m,
            nominal_ankle_height_m=nominal_ankle_height_m,
            minimum_vertical_support_force_n=(
                minimum_vertical_support_force_n
            ),
            maximum_supported_foot_speed_mps=(
                maximum_supported_foot_speed_mps
            ),
            minimum_upright_gravity_z=(
                minimum_upright_gravity_z
            ),
            completion_forward_distance_m=(
                completion_forward_distance_m
            ),
        )

        return (
            runtime.events[transition].to(torch.float)
            / env.step_dt
        )

def distance_target_foot_lift(
    env,
    asset_cfg,
    foot_body_names,
    first_riser_x_m: float = 1.0,
    nominal_ankle_height_m: float = 0.10,
    lookahead_m: float = 0.45,
    safety_clearance_m: float = 0.06,
    min_forward_speed_mps: float = 0.03,
    vertical_speed_scale_mps: float = 0.30,
):
    """Distance-conditioned swing-foot target-height reward."""

    import torch

    robot = env.scene[asset_cfg.name]

    terrain_generator = env.scene.terrain.cfg.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]
    riser_height = float(stair_cfg.riser_height_range[0])

    body_ids = robot.find_bodies(foot_body_names)[0]

    foot_pos = robot.data.body_pos_w[:, body_ids, :]
    foot_vel = robot.data.body_lin_vel_w[:, body_ids, :]

    origin = env.scene.env_origins

    foot_x = foot_pos[..., 0] - origin[:, None, 0]
    foot_z = foot_pos[..., 2] - origin[:, None, 2]

    distance_to_riser = first_riser_x_m - foot_x

    in_window = (
        (distance_to_riser > 0.0)
        & (distance_to_riser <= lookahead_m)
    )

    moving_forward = foot_vel[..., 0] >= min_forward_speed_mps

    proximity = torch.clamp(
        1.0 - distance_to_riser / max(lookahead_m, 1.0e-6),
        0.0,
        1.0,
    )

    required_extra_height = (
        proximity.square()
        * (riser_height + safety_clearance_m)
    )

    achieved_extra_height = torch.clamp(
        foot_z - nominal_ankle_height_m,
        min=0.0,
    )

    required_for_score = torch.clamp(
        required_extra_height,
        min=0.02,
    )

    height_score = torch.clamp(
        achieved_extra_height / required_for_score,
        0.0,
        1.0,
    )

    upward_velocity_score = torch.clamp(
        foot_vel[..., 2]
        / max(vertical_speed_scale_mps, 1.0e-6),
        0.0,
        1.0,
    )

    trajectory_score = (
        0.85 * height_score
        + 0.15 * upward_velocity_score
    )

    approach_weight = 0.25 + 0.75 * proximity

    per_foot = (
        in_window.float()
        * moving_forward.float()
        * approach_weight
        * trajectory_score
    )

    projected_gravity = robot.data.projected_gravity_b

    upright_gate = torch.clamp(
        (-projected_gravity[:, 2] - 0.80) / 0.20,
        0.0,
        1.0,
    )

    return torch.max(per_foot, dim=1).values * upright_gate
