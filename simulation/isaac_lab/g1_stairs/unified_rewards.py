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


def upright_forward_velocity_shortfall(
    env,
    asset_cfg: SceneEntityCfg,
    minimum_forward_velocity_mps: float = 0.15,
    backward_extra_scale: float = 1.5,
    upright_min: float = 0.80,
) -> torch.Tensor:
    """Penalize standing still or moving backward while upright.

    This is velocity-based, not position-progress based, so simply falling
    forward cannot continuously satisfy it.
    """

    robot = env.scene[asset_cfg.name]

    vx = robot.data.root_lin_vel_b[:, 0]

    upright = (
        -robot.data.projected_gravity_b[:, 2]
        >= upright_min
    ).to(vx.dtype)

    forward_shortfall = torch.clamp(
        (
            minimum_forward_velocity_mps
            - vx
        )
        / minimum_forward_velocity_mps,
        0.0,
        1.0,
    )

    backward = torch.clamp(
        -vx / minimum_forward_velocity_mps,
        0.0,
        2.0,
    )

    return upright * (
        forward_shortfall
        + backward_extra_scale * backward
    )








def track_stair_velocity_world_exp(
    env,
    asset_cfg: SceneEntityCfg,
    std: float = 0.35,
    approach_velocity_mps: float = 0.50,
    stair_velocity_mps: float = 0.18,
    slowdown_start_distance_m: float = 0.60,
    slowdown_end_distance_m: float = 0.35,
    platform_velocity_mps: float = 0.50,
    platform_accel_distance_m: float = 0.40,
) -> torch.Tensor:
    """Track staircase-aligned world velocity.

    Unlike the upstream yaw-frame velocity reward, this does not allow the
    robot to rotate left/right and continue earning full forward reward.
    """

    from .terrain_observations import stair_velocity_command

    robot = env.scene[asset_cfg.name]

    target = stair_velocity_command(
        env,
        approach_velocity_mps=approach_velocity_mps,
        stair_velocity_mps=stair_velocity_mps,
        slowdown_start_distance_m=slowdown_start_distance_m,
        slowdown_end_distance_m=slowdown_end_distance_m,
        platform_velocity_mps=platform_velocity_mps,
        platform_accel_distance_m=platform_accel_distance_m,
    )

    error_x = (
        robot.data.root_lin_vel_w[:, 0]
        - target[:, 0]
    )

    # Stair direction is world +X, so desired world lateral speed is zero.
    error_y = robot.data.root_lin_vel_w[:, 1]

    squared_error = (
        error_x * error_x
        + error_y * error_y
    )

    variance = max(float(std) ** 2, 1.0e-8)

    return torch.exp(
        -squared_error / variance
    )


def stair_heading_alignment(
    env,
    asset_cfg: SceneEntityCfg,
    full_penalty_yaw_rad: float = 0.50,
) -> torch.Tensor:
    """Penalize heading away from staircase +X while stairs remain ahead."""

    robot = env.scene[asset_cfg.name]
    cfg = stair_cfg(env)

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    platform_start = (
        float(cfg.start_clearance)
        + int(cfg.step_count)
        * float(cfg.tread_depth)
    )

    qw, qx, qy, qz = (
        robot.data.root_quat_w.unbind(dim=1)
    )

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    yaw_error = torch.atan2(
        torch.sin(yaw),
        torch.cos(yaw),
    )

    normalized = torch.clamp(
        torch.abs(yaw_error)
        / full_penalty_yaw_rad,
        0.0,
        1.0,
    )

    stairs_ahead = (
        root_x < platform_start
    ).to(normalized.dtype)

    return stairs_ahead * normalized * normalized


def terrain_relative_swing_clearance_dynamic(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_ankle_height_m: float = 0.10,
    lookahead_distance_m: float = 0.35,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.04,
    forward_speed_scale_mps: float = 0.12,
) -> torch.Tensor:
    """Reward real swing-foot lift using the exact per-environment riser.

    No left/right sequence, phase clock, or reference gait is prescribed.
    Whichever foot is physically in swing may earn the reward.
    """

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

    current_contact = (
        sensor.data.current_contact_time[
            :, sensor_cfg.body_ids
        ]
    )

    is_swing = current_contact <= 0.0

    # Static raised-foot exploit protection.
    forward_gate = torch.clamp(
        foot_vel[:, :, 0]
        / forward_speed_scale_mps,
        0.0,
        1.0,
    )

    upright_gate = torch.clamp(
        -robot.data.projected_gravity_b[:, 2],
        0.0,
        1.0,
    ).unsqueeze(1)

    best = torch.zeros_like(foot_z)
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

        previous_surface = (
            step_index * riser
        )

        required_surface = (
            (step_index + 1) * riser
        )

        baseline_z = (
            nominal_ankle_height_m
            + previous_surface
        )

        required_z = (
            nominal_ankle_height_m
            + required_surface
            + clearance_margin_m
        )

        denominator = torch.clamp(
            required_z - baseline_z,
            min=0.03,
        )

        lift_fraction = torch.clamp(
            (
                foot_z - baseline_z
            )
            / denominator,
            0.0,
            1.0,
        )

        # Increase importance as the foot approaches the vertical face,
        # while still providing an anticipatory gradient.
        phase = torch.clamp(
            (
                foot_x - start_x
            )
            / lookahead_distance_m,
            0.0,
            1.0,
        )

        proximity_gain = (
            0.50 + 0.50 * phase
        )

        score = (
            lift_fraction
            * proximity_gain
            * forward_gate
            * upright_gate
        )

        best = torch.where(
            active,
            torch.maximum(best, score),
            best,
        )

        assigned |= active

    return torch.max(
        best,
        dim=1,
    ).values


def terrain_relative_swing_clearance_upward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_ankle_height_m: float = 0.10,
    lookahead_distance_m: float = 0.35,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.04,
    upward_speed_scale_mps: float = 0.08,
    forward_speed_scale_mps: float = 0.12,
) -> torch.Tensor:
    """Reward genuine upward swing-foot motion before the next riser.

    039 correction:
    - exact per-environment riser height;
    - no prescribed left/right leg;
    - no gait clock;
    - upward motion can earn full clearance reward even when forward
      foot velocity is small;
    - a weak forward-motion contribution remains;
    - a statically held raised foot does not earn sustained reward.
    """

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

    current_contact = (
        sensor.data.current_contact_time[
            :, sensor_cfg.body_ids
        ]
    )

    is_swing = current_contact <= 0.0

    upward_gate = torch.clamp(
        foot_vel[:, :, 2]
        / upward_speed_scale_mps,
        0.0,
        1.0,
    )

    forward_gate = torch.clamp(
        foot_vel[:, :, 0]
        / forward_speed_scale_mps,
        0.0,
        1.0,
    )

    # Upward motion is now the primary evidence of genuine lift.
    # Forward swing contributes weakly but is no longer required.
    motion_gate = torch.maximum(
        upward_gate,
        0.25 * forward_gate,
    )

    upright_gate = torch.clamp(
        -robot.data.projected_gravity_b[:, 2],
        0.0,
        1.0,
    ).unsqueeze(1)

    best = torch.zeros_like(foot_z)
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

        previous_surface = (
            step_index * riser
        )

        next_surface = (
            (step_index + 1) * riser
        )

        baseline_z = (
            nominal_ankle_height_m
            + previous_surface
        )

        required_z = (
            nominal_ankle_height_m
            + next_surface
            + clearance_margin_m
        )

        normalization = torch.clamp(
            required_z - baseline_z,
            min=0.03,
        )

        lift_fraction = torch.clamp(
            (
                foot_z - baseline_z
            )
            / normalization,
            0.0,
            1.0,
        )

        phase = torch.clamp(
            (
                foot_x - start_x
            )
            / lookahead_distance_m,
            0.0,
            1.0,
        )

        # Give useful signal before the lip while increasing its
        # importance as the foot approaches the riser.
        proximity_gain = (
            0.50 + 0.50 * phase
        )

        score = (
            lift_fraction
            * proximity_gain
            * motion_gate
            * upright_gate
        )

        best = torch.where(
            active,
            torch.maximum(best, score),
            best,
        )

        assigned |= active

    return torch.max(
        best,
        dim=1,
    ).values


def terrain_relative_swing_lift_dense(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_ankle_height_m: float = 0.10,
    lookahead_distance_m: float = 0.40,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.04,
    upward_speed_scale_mps: float = 0.08,
) -> torch.Tensor:
    """Dense 040 reward for discovering actual stair-leg lift.

    Important difference from 039:
    lift height and upward velocity are additive rather than multiplicative.

    This allows the policy to receive useful reward as soon as the swing
    ankle begins to rise, even when horizontal foot velocity is almost zero.

    No left/right sequence, gait clock, reference trajectory, or scripted
    foot selection is used.
    """

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

    current_contact = sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]

    is_swing = current_contact <= 0.0

    upward_gate = torch.clamp(
        foot_vel[:, :, 2]
        / upward_speed_scale_mps,
        0.0,
        1.0,
    )

    upright_gate = torch.clamp(
        -robot.data.projected_gravity_b[:, 2],
        0.0,
        1.0,
    ).unsqueeze(1)

    best = torch.zeros_like(foot_z)
    assigned = torch.zeros_like(is_swing)

    for step_index in range(step_count):

        edge = (
            first_riser
            + step_index * tread
        )

        start_x = (
            edge
            - lookahead_distance_m
        )

        active = (
            is_swing
            & ~assigned
            & (foot_x >= start_x)
            & (foot_x <= edge + lip_distance_m)
        )

        previous_surface = (
            step_index * riser
        )

        next_surface = (
            (step_index + 1) * riser
        )

        baseline_z = (
            nominal_ankle_height_m
            + previous_surface
        )

        required_z = (
            nominal_ankle_height_m
            + next_surface
            + clearance_margin_m
        )

        normalization = torch.clamp(
            required_z - baseline_z,
            min=0.03,
        )

        lift_fraction = torch.clamp(
            (
                foot_z
                - baseline_z
            )
            / normalization,
            0.0,
            1.0,
        )

        phase = torch.clamp(
            (
                foot_x
                - start_x
            )
            / lookahead_distance_m,
            0.0,
            1.0,
        )

        proximity_gain = (
            0.50
            + 0.50 * phase
        )

        # 040:
        # Actual height progress is the dominant discovery signal.
        # Upward velocity gives an additional incentive to initiate lift.
        # They are deliberately NOT multiplied together.
        lift_signal = (
            0.80 * lift_fraction
            + 0.20 * upward_gate
        )

        score = (
            lift_signal
            * proximity_gain
            * upright_gate
        )

        best = torch.where(
            active,
            torch.maximum(
                best,
                score,
            ),
            best,
        )

        assigned |= active

    return torch.max(
        best,
        dim=1,
    ).values


def terrain_relative_swing_clearance_calibrated(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    normal_swing_ceiling_m: float = 0.065,
    lookahead_distance_m: float = 0.40,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.04,
) -> torch.Tensor:
    """041 stair-specific clearance above the measured flat-gait envelope.

    Unlike the 040 dense reward, ordinary flat-ground swing height does not
    receive stair-clearance credit.

    No gait clock, fixed leg order, joint trajectory, or reference motion is
    used.
    """

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

    current_contact = sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]

    is_swing = current_contact <= 0.0

    upright_gate = torch.clamp(
        -robot.data.projected_gravity_b[:, 2],
        0.0,
        1.0,
    ).unsqueeze(1)

    best = torch.zeros_like(foot_z)
    assigned = torch.zeros_like(is_swing)

    for step_index in range(step_count):

        edge = (
            first_riser
            + step_index * tread
        )

        start_x = (
            edge
            - lookahead_distance_m
        )

        active = (
            is_swing
            & ~assigned
            & (foot_x >= start_x)
            & (foot_x <= edge + lip_distance_m)
        )

        previous_surface = (
            step_index * riser
        )

        next_surface = (
            (step_index + 1) * riser
        )

        # The measured flat walking envelope ends around 0.061 m.
        # Reward only motion above a conservative 0.065 m ceiling.
        reward_start_z = (
            normal_swing_ceiling_m
            + previous_surface
        )

        required_z = (
            flat_ankle_height_m
            + next_surface
            + clearance_margin_m
        )

        normalization = torch.clamp(
            required_z - reward_start_z,
            min=0.03,
        )

        clearance_fraction = torch.clamp(
            (
                foot_z
                - reward_start_z
            )
            / normalization,
            0.0,
            1.0,
        )

        proximity = torch.clamp(
            (
                foot_x
                - start_x
            )
            / max(
                lookahead_distance_m,
                1.0e-6,
            ),
            0.0,
            1.0,
        )

        # Preserve an early gradient while increasing importance near the lip.
        proximity_gain = (
            0.5
            + 0.5 * proximity
        )

        score = (
            clearance_fraction
            * proximity_gain
            * upright_gate
        )

        best = torch.where(
            active,
            torch.maximum(best, score),
            best,
        )

        assigned |= active

    return torch.max(
        best,
        dim=1,
    ).values


def terrain_swing_leg_shortening(
    env,
    hip_cfg: SceneEntityCfg,
    ankle_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    reference_leg_length_m: float = 0.655,
    target_leg_length_m: float = 0.610,
    lookahead_distance_m: float = 0.35,
    lip_distance_m: float = 0.04,
    clearance_margin_m: float = 0.04,
) -> torch.Tensor:
    """041 task-gated swing-leg shortening.

    Encourages the swing leg to become shorter near a riser while the foot
    still lacks clearance.

    The reward uses hip-to-ankle distance instead of prescribing a knee angle.
    No left/right order or gait timing is imposed.
    """

    robot = env.scene[hip_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    cfg = stair_cfg(env)

    hip_pos = robot.data.body_pos_w[
        :, hip_cfg.body_ids, :
    ]

    ankle_pos = robot.data.body_pos_w[
        :, ankle_cfg.body_ids, :
    ]

    if hip_pos.shape[1] != ankle_pos.shape[1]:
        raise RuntimeError(
            "hip and ankle body counts differ in "
            "terrain_swing_leg_shortening"
        )

    riser = riser_height_per_env(env).unsqueeze(1)

    tread = float(cfg.tread_depth)
    step_count = int(cfg.step_count)
    first_riser = float(cfg.start_clearance)

    ankle_x = (
        ankle_pos[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )

    ankle_z = (
        ankle_pos[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    leg_length = torch.linalg.vector_norm(
        hip_pos - ankle_pos,
        dim=2,
    )

    current_contact = sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]

    is_swing = current_contact <= 0.0

    shortening_fraction = torch.clamp(
        (
            reference_leg_length_m
            - leg_length
        )
        / max(
            reference_leg_length_m
            - target_leg_length_m,
            1.0e-6,
        ),
        0.0,
        1.0,
    )

    upright_gate = torch.clamp(
        -robot.data.projected_gravity_b[:, 2],
        0.0,
        1.0,
    ).unsqueeze(1)

    best = torch.zeros_like(ankle_z)
    assigned = torch.zeros_like(is_swing)

    for step_index in range(step_count):

        edge = (
            first_riser
            + step_index * tread
        )

        start_x = (
            edge
            - lookahead_distance_m
        )

        previous_surface = (
            step_index * riser
        )

        next_surface = (
            (step_index + 1) * riser
        )

        required_z = (
            flat_ankle_height_m
            + next_surface
            + clearance_margin_m
        )

        # Stop rewarding further shortening once useful clearance exists.
        needs_clearance = (
            ankle_z < required_z
        )

        active = (
            is_swing
            & needs_clearance
            & ~assigned
            & (ankle_x >= start_x)
            & (ankle_x <= edge + lip_distance_m)
        )

        proximity = torch.clamp(
            (
                ankle_x
                - start_x
            )
            / max(
                lookahead_distance_m,
                1.0e-6,
            ),
            0.0,
            1.0,
        )

        proximity_gain = (
            0.5
            + 0.5 * proximity
        )

        score = (
            shortening_fraction
            * proximity_gain
            * upright_gate
        )

        best = torch.where(
            active,
            torch.maximum(best, score),
            best,
        )

        assigned |= active

    return torch.max(
        best,
        dim=1,
    ).values


def terrain_preclearance_forward_speed(
    env,
    asset_cfg: SceneEntityCfg,
    ankle_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    normal_swing_ceiling_m: float = 0.065,
    activation_distance_m: float = 0.30,
    clearance_margin_m: float = 0.04,
    speed_scale_mps: float = 0.20,
) -> torch.Tensor:
    """041 penalty for forward recommit before stair clearance exists.

    The penalty is active only near the next riser and only while neither
    ankle has reached useful stair-specific clearance.

    Once clearance begins, this term releases automatically.
    """

    robot = env.scene[asset_cfg.name]
    cfg = stair_cfg(env)

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    forward_speed = torch.clamp(
        robot.data.root_lin_vel_w[:, 0],
        min=0.0,
    )

    ankle_pos = robot.data.body_pos_w[
        :, ankle_cfg.body_ids, :
    ]

    ankle_z = (
        ankle_pos[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    riser = riser_height_per_env(env)

    tread = float(cfg.tread_depth)
    step_count = int(cfg.step_count)
    first_riser = float(cfg.start_clearance)

    penalty = torch.zeros_like(root_x)

    for step_index in range(step_count):

        edge = (
            first_riser
            + step_index * tread
        )

        distance = edge - root_x

        previous_surface = (
            step_index * riser
        )

        next_surface = (
            (step_index + 1) * riser
        )

        # "Clearance has started" means exceeding normal flat swing,
        # not merely leaving the floor.
        clearance_started_z = (
            normal_swing_ceiling_m
            + previous_surface
        )

        required_z = (
            flat_ankle_height_m
            + next_surface
            + clearance_margin_m
        )

        best_ankle_z = torch.max(
            ankle_z,
            dim=1,
        ).values

        clearance_started = (
            best_ankle_z
            > clearance_started_z
        )

        clearance_complete = (
            best_ankle_z
            >= required_z
        )

        active = (
            (distance >= 0.0)
            & (distance <= activation_distance_m)
            & ~clearance_started
            & ~clearance_complete
        )

        normalized_speed = torch.clamp(
            forward_speed
            / max(
                speed_scale_mps,
                1.0e-6,
            ),
            0.0,
            2.0,
        )

        step_penalty = torch.square(
            normalized_speed
        )

        penalty = torch.where(
            active,
            torch.maximum(
                penalty,
                step_penalty,
            ),
            penalty,
        )

    return penalty

