"""Physics-grounded rewards for unified stair locomotion."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .terrain_state import riser_height_per_env, stair_cfg


def terrain_relative_clearance_shortfall(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_ankle_height_m: float = 0.04,
    lookahead_distance_m: float = 0.35,
    lip_distance_m: float = 0.03,
    clearance_margin_m: float = 0.04,
) -> torch.Tensor:
    """Penalty magnitude when the current swing foot is too low."""

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    cfg = stair_cfg(env)

    riser = riser_height_per_env(env).unsqueeze(1)

    tread = float(cfg.tread_depth)
    step_count = int(cfg.step_count)
    first_riser = float(cfg.start_clearance)

    foot_pos = robot.data.body_pos_w[
        :, asset_cfg.body_ids, :
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    current_contact = (
        sensor.data.current_contact_time[
            :, sensor_cfg.body_ids
        ]
    )

    is_swing = current_contact <= 0.0

    worst = torch.zeros_like(foot_z)
    assigned = torch.zeros_like(is_swing)

    for step_index in range(step_count):

        edge = (
            first_riser
            + step_index * tread
        )

        start_x = (
            edge - lookahead_distance_m
        )

        active = (
            is_swing
            & ~assigned
            & (foot_x >= start_x)
            & (foot_x <= edge + lip_distance_m)
        )

        phase = torch.clamp(
            (foot_x - start_x)
            / lookahead_distance_m,
            0.0,
            1.0,
        )

        previous_surface = (
            step_index * riser
        )

        next_surface = (
            (step_index + 1) * riser
        )

        desired_z = (
            nominal_ankle_height_m
            + previous_surface
            + phase
            * (
                next_surface
                + clearance_margin_m
                - previous_surface
            )
        )

        normalization = torch.clamp(
            riser + clearance_margin_m,
            min=0.05,
        )

        shortfall = torch.clamp(
            (desired_z - foot_z)
            / normalization,
            0.0,
            1.0,
        )

        proximity_weight = (
            0.25 + 0.75 * phase
        )

        value = (
            shortfall
            * proximity_weight
            * active.to(foot_z.dtype)
        )

        worst = torch.maximum(
            worst,
            value,
        )

        assigned |= active

    # Positive magnitude.
    # RewTerm weight is negative.
    return torch.max(
        worst,
        dim=1,
    ).values


def terrain_riser_face_strike_proxy(
    env,
    asset_cfg: SceneEntityCfg,
    nominal_ankle_height_m: float = 0.04,
    face_window_m: float = 0.035,
    clearance_margin_m: float = 0.025,
    min_forward_speed_mps: float = 0.02,
) -> torch.Tensor:
    """Penalize a moving foot reaching a vertical riser while too low."""

    robot = env.scene[asset_cfg.name]
    cfg = stair_cfg(env)

    riser = riser_height_per_env(env).unsqueeze(1)

    tread = float(cfg.tread_depth)
    step_count = int(cfg.step_count)
    first_riser = float(cfg.start_clearance)

    foot_pos = robot.data.body_pos_w[
        :, asset_cfg.body_ids, :
    ]

    foot_vel = robot.data.body_lin_vel_w[
        :, asset_cfg.body_ids, :
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    strike = torch.zeros_like(
        foot_z,
        dtype=torch.bool,
    )

    for step_index in range(step_count):

        edge = (
            first_riser
            + step_index * tread
        )

        required_z = (
            nominal_ankle_height_m
            + (step_index + 1) * riser
            + clearance_margin_m
        )

        strike |= (
            (torch.abs(foot_x - edge) <= face_window_m)
            & (foot_z < required_z)
            & (
                foot_vel[:, :, 0]
                >= min_forward_speed_mps
            )
        )

    return torch.max(
        strike.to(foot_z.dtype),
        dim=1,
    ).values


def terrain_tread_touchdown(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_ankle_height_m: float = 0.04,
    x_margin_m: float = 0.02,
    z_tolerance_m: float = 0.04,
    upright_min: float = 0.80,
) -> torch.Tensor:
    """Reward a newly established foot contact on a valid tread."""

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    cfg = stair_cfg(env)

    riser = riser_height_per_env(env).unsqueeze(1)

    tread = float(cfg.tread_depth)
    step_count = int(cfg.step_count)
    first_riser = float(cfg.start_clearance)

    foot_pos = robot.data.body_pos_w[
        :, asset_cfg.body_ids, :
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    first_contact = sensor.compute_first_contact(
        env.step_dt
    )[:, sensor_cfg.body_ids].bool()

    upright = (
        -robot.data.projected_gravity_b[:, 2]
        >= upright_min
    ).unsqueeze(1)

    touchdown = torch.zeros_like(
        first_contact
    )

    for step_index in range(step_count):

        tread_start = (
            first_riser
            + step_index * tread
        )

        tread_end = (
            tread_start + tread
        )

        expected_z = (
            nominal_ankle_height_m
            + (step_index + 1) * riser
        )

        valid = (
            first_contact
            & upright
            & (
                foot_x
                >= tread_start + x_margin_m
            )
            & (
                foot_x
                <= tread_end - x_margin_m
            )
            & (
                torch.abs(
                    foot_z - expected_z
                )
                <= z_tolerance_m
            )
        )

        touchdown |= valid

    return torch.max(
        touchdown.to(foot_z.dtype),
        dim=1,
    ).values
