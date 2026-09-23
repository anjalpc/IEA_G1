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


# Backward-compatible 035 reward. Retained because env_cfg still exposes historical reward stages.
def terrain_relative_swing_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_ankle_height_m: float = 0.10,
    lookahead_distance_m: float = 0.35,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.04,
    forward_speed_scale_mps: float = 0.15,
) -> torch.Tensor:
    """Reward whichever swing foot safely clears the next physical riser.

    Properties:
    - uses the live staircase geometry;
    - detects the swing foot from physical contact;
    - does not prescribe left versus right;
    - does not use a reference phase or clock;
    - is bounded;
    - gives no reward for holding a foot statically in the air.
    """
    robot = env.scene[asset_cfg.name]
    contact_sensor = (
        env.scene.sensors[sensor_cfg.name]
    )

    stair_cfg = (
        env.scene.terrain.cfg.terrain_generator
        .sub_terrains["three_step_ascent"]
    )

    riser_height_m = float(
        stair_cfg.riser_height_range[0]
    )
    tread_depth_m = float(
        stair_cfg.tread_depth
    )
    step_count = int(
        stair_cfg.step_count
    )
    first_riser_x_m = float(
        stair_cfg.start_clearance
    )

    foot_pos_w = (
        robot.data.body_pos_w[
            :,
            asset_cfg.body_ids,
            :,
        ]
    )

    foot_vel_w = (
        robot.data.body_lin_vel_w[
            :,
            asset_cfg.body_ids,
            :,
        ]
    )

    foot_x = (
        foot_pos_w[:, :, 0]
        - env.scene.env_origins[
            :, 0
        ].unsqueeze(1)
    )

    foot_z = (
        foot_pos_w[:, :, 2]
        - env.scene.env_origins[
            :, 2
        ].unsqueeze(1)
    )

    is_contact = (
        contact_sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
        > 0.0
    )

    is_swing = ~is_contact

    # A static raised leg must not become a reward exploit.
    forward_gate = torch.clamp(
        foot_vel_w[:, :, 0]
        / forward_speed_scale_mps,
        min=0.0,
        max=1.0,
    )

    # Reduce clearance reward as the robot falls.
    upright_gate = torch.clamp(
        -robot.data.projected_gravity_b[
            :, 2
        ],
        min=0.0,
        max=1.0,
    ).unsqueeze(1)

    best_score = torch.zeros_like(
        foot_z
    )

    assigned = torch.zeros_like(
        is_swing
    )

    previous_tread_height_m = 0.0

    for step_index in range(
        step_count
    ):
        edge_m = (
            first_riser_x_m
            + step_index * tread_depth_m
        )

        tread_height_m = (
            step_index + 1
        ) * riser_height_m

        start_x_m = (
            edge_m
            - lookahead_distance_m
        )

        active = (
            is_swing
            & ~assigned
            & (
                foot_x
                >= start_x_m
            )
            & (
                foot_x
                <= edge_m
                + lip_distance_m
            )
        )

        phase = torch.clamp(
            (
                foot_x
                - start_x_m
            )
            / lookahead_distance_m,
            min=0.0,
            max=1.0,
        )

        start_target_z = (
            nominal_ankle_height_m
            + previous_tread_height_m
        )

        end_target_z = (
            nominal_ankle_height_m
            + tread_height_m
            + clearance_margin_m
        )

        desired_z = (
            start_target_z
            + phase
            * (
                end_target_z
                - start_target_z
            )
        )

        normalization = max(
            riser_height_m
            + clearance_margin_m,
            0.05,
        )

        normalized_shortfall = (
            torch.clamp(
                (
                    desired_z
                    - foot_z
                )
                / normalization,
                min=0.0,
                max=1.0,
            )
        )

        score = (
            (
                1.0
                - normalized_shortfall
            )
            * phase
            * forward_gate
            * upright_gate
            * active.to(
                foot_z.dtype
            )
        )

        best_score = torch.maximum(
            best_score,
            score,
        )

        assigned |= active

        previous_tread_height_m = (
            tread_height_m
        )

    # Never reward two airborne feet twice.
    return torch.max(
        best_score,
        dim=1,
    ).values


# Backward-compatible public name retained because env_cfg.py is an
# import-time catalog containing historical and current reward stages.
terrain_riser_face_strike = terrain_riser_face_strike_proxy
