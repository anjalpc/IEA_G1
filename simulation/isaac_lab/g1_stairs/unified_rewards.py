"""Physics-grounded rewards for unified stair locomotion."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .reward_math import exp_tracking_score

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
    zero_centered: bool = False,
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

    # Historical modes use the original positive tracking reward. 047 opts
    # into a zero-centered form so merely surviving for more control steps
    # cannot accumulate positive return. Perfect tracking is 0; tracking
    # error is negative.
    return exp_tracking_score(
        squared_error,
        variance,
        zero_centered=zero_centered,
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


# ============================================================================
# 042: PHYSICS-GATED STAIR TRANSITION REWARDS
# ============================================================================
#
# Design goals:
#
# - preserve the geometry-conditioned PPO architecture;
# - preserve free leg-order selection;
# - prevent future-stair reward before current-step completion;
# - prevent early high-kick reward;
# - prevent touchdown/contact farming;
# - require real support + upward body transfer to unlock the next stair;
# - allow modest body lean, but suppress the large backward-lean exploit
#   observed in 041c.
#
# The only persistent state is:
#   completed stair count per environment
#   root height at episode start
#
# This is task progress state, not a gait clock or reference trajectory.


def _stair_042_progress_buffers(
    env,
    robot,
):
    """Return per-env stair completion state and episode-start root height."""

    device = robot.data.root_pos_w.device
    num_envs = robot.data.root_pos_w.shape[0]

    completed_name = "_iea_stair_042_completed_steps"
    root_z0_name = "_iea_stair_042_root_z0"

    completed = getattr(
        env,
        completed_name,
        None,
    )

    root_z0 = getattr(
        env,
        root_z0_name,
        None,
    )

    if (
        completed is None
        or completed.shape[0] != num_envs
        or completed.device != device
    ):
        completed = torch.zeros(
            num_envs,
            dtype=torch.long,
            device=device,
        )
        setattr(
            env,
            completed_name,
            completed,
        )

    if (
        root_z0 is None
        or root_z0.shape[0] != num_envs
        or root_z0.device != device
    ):
        root_z0 = robot.data.root_pos_w[:, 2].detach().clone()
        setattr(
            env,
            root_z0_name,
            root_z0,
        )

    episode_length = getattr(
        env,
        "episode_length_buf",
        None,
    )

    if episode_length is not None:
        reset_mask = episode_length <= 1

        completed[reset_mask] = 0
        root_z0[reset_mask] = (
            robot.data.root_pos_w[
                reset_mask,
                2,
            ]
            .detach()
        )

    return completed, root_z0


def _stair_042_target_geometry(
    env,
    robot,
):
    """Return geometry for the one currently unlocked stair."""

    cfg = stair_cfg(env)

    completed, root_z0 = (
        _stair_042_progress_buffers(
            env,
            robot,
        )
    )

    step_count = int(cfg.step_count)
    tread = float(cfg.tread_depth)
    first_riser = float(cfg.start_clearance)

    active = completed < step_count

    target_index = torch.clamp(
        completed,
        min=0,
        max=max(step_count - 1, 0),
    )

    riser = riser_height_per_env(env)

    edge = (
        first_riser
        + target_index.to(
            dtype=robot.data.root_pos_w.dtype
        )
        * tread
    )

    previous_surface = (
        target_index.to(
            dtype=robot.data.root_pos_w.dtype
        )
        * riser
    )

    next_surface = (
        (
            target_index.to(
                dtype=robot.data.root_pos_w.dtype
            )
            + 1.0
        )
        * riser
    )

    return {
        "completed": completed,
        "root_z0": root_z0,
        "target_index": target_index,
        "active": active,
        "riser": riser,
        "edge": edge,
        "previous_surface": previous_surface,
        "next_surface": next_surface,
        "tread": tread,
        "step_count": step_count,
    }


def _stair_042_tilt(
    robot,
):
    """Combined body tilt from projected gravity."""

    upright_cos = torch.clamp(
        -robot.data.projected_gravity_b[:, 2],
        -1.0,
        1.0,
    )

    return torch.acos(upright_cos)


def _stair_042_stability_gate(
    robot,
    free_tilt_rad: float = 0.20944,
    zero_tilt_rad: float = 0.43633,
):
    """1 below ~12 deg, smoothly falls to 0 by ~25 deg."""

    tilt = _stair_042_tilt(robot)

    gate = 1.0 - torch.clamp(
        (
            tilt
            - free_tilt_rad
        )
        / max(
            zero_tilt_rad
            - free_tilt_rad,
            1.0e-6,
        ),
        0.0,
        1.0,
    )

    return torch.square(gate)


def terrain_current_riser_clearance_042(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    normal_swing_ceiling_m: float = 0.065,
    lookahead_distance_m: float = 0.22,
    lip_distance_m: float = 0.03,
    clearance_margin_m: float = 0.04,
) -> torch.Tensor:
    """Clear only the currently unlocked riser.

    041 rewarded swing clearance as early as 0.40 m before a riser and
    started with a 0.5 proximity multiplier.

    042:
      - uses root-to-current-riser distance;
      - activates only inside 0.22 m;
      - proximity starts at zero;
      - uses proximity squared;
      - cannot target a future riser before completion of the current one.
    """

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    foot_pos = robot.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[
            :,
            0,
        ].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    distance = geo["edge"] - root_x

    current_contact = (
        sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    is_swing = current_contact <= 0.0

    reward_start_z = (
        normal_swing_ceiling_m
        + geo["previous_surface"]
    )

    required_z = (
        flat_ankle_height_m
        + geo["next_surface"]
        + clearance_margin_m
    )

    normalization = torch.clamp(
        required_z - reward_start_z,
        min=0.03,
    )

    clearance_fraction = torch.clamp(
        (
            foot_z
            - reward_start_z.unsqueeze(1)
        )
        / normalization.unsqueeze(1),
        0.0,
        1.0,
    )

    proximity = torch.clamp(
        1.0
        - distance
        / max(
            lookahead_distance_m,
            1.0e-6,
        ),
        0.0,
        1.0,
    )

    # 041 started at 50% strength.
    # 042 starts at zero and rises strongly only near the lip.
    proximity_gain = torch.square(
        proximity
    ).unsqueeze(1)

    root_zone = (
        geo["active"]
        & (distance >= 0.0)
        & (
            distance
            <= lookahead_distance_m
        )
    ).unsqueeze(1)

    foot_zone = (
        (foot_x >= (
            geo["edge"]
            - lookahead_distance_m
        ).unsqueeze(1))
        & (foot_x <= (
            geo["edge"]
            + lip_distance_m
        ).unsqueeze(1))
    )

    stability = (
        _stair_042_stability_gate(
            robot,
        )
        .unsqueeze(1)
    )

    score = (
        clearance_fraction
        * proximity_gain
        * stability
    )

    score = torch.where(
        root_zone
        & foot_zone
        & is_swing,
        score,
        torch.zeros_like(score),
    )

    return torch.max(
        score,
        dim=1,
    ).values


def terrain_current_swing_leg_shortening_042(
    env,
    hip_cfg: SceneEntityCfg,
    ankle_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    reference_leg_length_m: float = 0.655,
    target_leg_length_m: float = 0.610,
    lookahead_distance_m: float = 0.20,
    clearance_margin_m: float = 0.04,
) -> torch.Tensor:
    """Compact swing preparation for only the current riser."""

    robot = env.scene[hip_cfg.name]
    sensor = env.scene.sensors[
        sensor_cfg.name
    ]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    hip_pos = robot.data.body_pos_w[
        :,
        hip_cfg.body_ids,
        :,
    ]

    ankle_pos = robot.data.body_pos_w[
        :,
        ankle_cfg.body_ids,
        :,
    ]

    if hip_pos.shape[1] != ankle_pos.shape[1]:
        raise RuntimeError(
            "042 hip/ankle body count mismatch"
        )

    ankle_x = (
        ankle_pos[:, :, 0]
        - env.scene.env_origins[
            :,
            0,
        ].unsqueeze(1)
    )

    ankle_z = (
        ankle_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    leg_length = torch.linalg.vector_norm(
        hip_pos - ankle_pos,
        dim=2,
    )

    shortening = torch.clamp(
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

    contact_time = (
        sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    is_swing = contact_time <= 0.0

    required_z = (
        flat_ankle_height_m
        + geo["next_surface"]
        + clearance_margin_m
    )

    needs_clearance = (
        ankle_z
        < required_z.unsqueeze(1)
    )

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    distance = geo["edge"] - root_x

    proximity = torch.clamp(
        1.0
        - distance
        / max(
            lookahead_distance_m,
            1.0e-6,
        ),
        0.0,
        1.0,
    )

    proximity = torch.square(
        proximity
    ).unsqueeze(1)

    active = (
        geo["active"]
        & (distance >= 0.0)
        & (
            distance
            <= lookahead_distance_m
        )
    ).unsqueeze(1)

    foot_before_lip = (
        ankle_x
        <= (
            geo["edge"]
            + 0.02
        ).unsqueeze(1)
    )

    stability = (
        _stair_042_stability_gate(
            robot,
        )
        .unsqueeze(1)
    )

    score = (
        shortening
        * proximity
        * stability
    )

    score = torch.where(
        active
        & is_swing
        & needs_clearance
        & foot_before_lip,
        score,
        torch.zeros_like(score),
    )

    return torch.max(
        score,
        dim=1,
    ).values


def terrain_excessive_swing_height_042(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    clearance_margin_m: float = 0.04,
    free_extra_height_m: float = 0.05,
    scale_m: float = 0.05,
    activation_distance_m: float = 0.25,
) -> torch.Tensor:
    """Penalize the extreme high-kick exploit seen late in 041c."""

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[
        sensor_cfg.name
    ]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    foot_pos = robot.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    contact_time = (
        sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    is_swing = contact_time <= 0.0

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    distance = geo["edge"] - root_x

    required_z = (
        flat_ankle_height_m
        + geo["next_surface"]
        + clearance_margin_m
    )

    free_ceiling = (
        required_z
        + free_extra_height_m
    )

    excess = torch.clamp(
        (
            foot_z
            - free_ceiling.unsqueeze(1)
        )
        / max(
            scale_m,
            1.0e-6,
        ),
        min=0.0,
        max=2.0,
    )

    active = (
        geo["active"]
        & (distance >= -0.05)
        & (
            distance
            <= activation_distance_m
        )
    ).unsqueeze(1)

    penalty = torch.where(
        active & is_swing,
        torch.square(excess),
        torch.zeros_like(excess),
    )

    return torch.max(
        penalty,
        dim=1,
    ).values


def terrain_current_preclearance_speed_042(
    env,
    asset_cfg: SceneEntityCfg,
    ankle_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    clearance_margin_m: float = 0.04,
    activation_distance_m: float = 0.25,
    speed_scale_mps: float = 0.20,
) -> torch.Tensor:
    """Slow only for the currently unlocked riser."""

    robot = env.scene[asset_cfg.name]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    distance = geo["edge"] - root_x

    forward_speed = torch.clamp(
        robot.data.root_lin_vel_w[:, 0],
        min=0.0,
    )

    ankle_pos = robot.data.body_pos_w[
        :,
        ankle_cfg.body_ids,
        :,
    ]

    ankle_z = (
        ankle_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    required_z = (
        flat_ankle_height_m
        + geo["next_surface"]
        + clearance_margin_m
    )

    clearance_complete = (
        torch.max(
            ankle_z,
            dim=1,
        ).values
        >= required_z
    )

    active = (
        geo["active"]
        & (distance >= 0.0)
        & (
            distance
            <= activation_distance_m
        )
        & ~clearance_complete
    )

    normalized = torch.clamp(
        forward_speed
        / max(
            speed_scale_mps,
            1.0e-6,
        ),
        0.0,
        2.0,
    )

    return torch.where(
        active,
        torch.square(normalized),
        torch.zeros_like(normalized),
    )


def terrain_current_riser_blocked_042(
    env,
    asset_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    clearance_margin_m: float = 0.025,
    face_window_m: float = 0.035,
) -> torch.Tensor:
    """Penalize low-foot approach to only the current riser face."""

    robot = env.scene[asset_cfg.name]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    foot_pos = robot.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[
            :,
            0,
        ].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    required_z = (
        flat_ankle_height_m
        + geo["next_surface"]
        + clearance_margin_m
    )

    near_face = (
        torch.abs(
            foot_x
            - geo["edge"].unsqueeze(1)
        )
        <= face_window_m
    )

    too_low = (
        foot_z
        < required_z.unsqueeze(1)
    )

    forward = (
        robot.data.root_lin_vel_w[:, 0]
        > 0.02
    ).unsqueeze(1)

    blocked = (
        near_face
        & too_low
        & forward
        & geo["active"].unsqueeze(1)
    )

    return torch.any(
        blocked,
        dim=1,
    ).to(
        dtype=robot.data.root_pos_w.dtype
    )


def terrain_current_tread_placement_042(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    x_entry_margin_m: float = 0.02,
    x_exit_margin_m: float = 0.04,
    height_window_m: float = 0.10,
    descent_speed_scale_mps: float = 0.20,
) -> torch.Tensor:
    """Reward controlled descent into the current tread interior.

    This is pre-contact placement, not touchdown credit.
    """

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[
        sensor_cfg.name
    ]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    foot_pos = robot.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_vel = robot.data.body_lin_vel_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[
            :,
            0,
        ].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    contact_time = (
        sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    still_swing = contact_time <= 0.0

    target_z = (
        flat_ankle_height_m
        + geo["next_surface"]
    )

    tread_start = (
        geo["edge"]
        + x_entry_margin_m
    )

    tread_end = (
        geo["edge"]
        + geo["tread"]
        - x_exit_margin_m
    )

    inside_tread = (
        (
            foot_x
            >= tread_start.unsqueeze(1)
        )
        & (
            foot_x
            <= tread_end.unsqueeze(1)
        )
    )

    above_target = (
        foot_z
        - target_z.unsqueeze(1)
    )

    near_height = (
        (above_target >= -0.01)
        & (
            above_target
            <= height_window_m
        )
    )

    descending = torch.clamp(
        -foot_vel[:, :, 2]
        / max(
            descent_speed_scale_mps,
            1.0e-6,
        ),
        0.0,
        1.0,
    )

    height_gate = torch.clamp(
        1.0
        - torch.abs(above_target)
        / max(
            height_window_m,
            1.0e-6,
        ),
        0.0,
        1.0,
    )

    stability = (
        _stair_042_stability_gate(
            robot,
        )
        .unsqueeze(1)
    )

    score = (
        descending
        * height_gate
        * stability
    )

    score = torch.where(
        geo["active"].unsqueeze(1)
        & still_swing
        & inside_tread
        & near_height,
        score,
        torch.zeros_like(score),
    )

    return torch.max(
        score,
        dim=1,
    ).values


def terrain_tread_support_acquisition_042(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    x_entry_margin_m: float = 0.02,
    x_exit_margin_m: float = 0.04,
    z_tolerance_m: float = 0.025,
    min_contact_time_s: float = 0.08,
    max_reward_contact_time_s: float = 0.18,
) -> torch.Tensor:
    """Short reward pulse when real tread support is first established.

    The upper contact-time limit prevents standing still on one tread from
    farming reward indefinitely.
    """

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[
        sensor_cfg.name
    ]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    foot_pos = robot.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[
            :,
            0,
        ].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    contact_time = (
        sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    target_z = (
        flat_ankle_height_m
        + geo["next_surface"]
    )

    tread_start = (
        geo["edge"]
        + x_entry_margin_m
    )

    tread_end = (
        geo["edge"]
        + geo["tread"]
        - x_exit_margin_m
    )

    inside_tread = (
        (
            foot_x
            >= tread_start.unsqueeze(1)
        )
        & (
            foot_x
            <= tread_end.unsqueeze(1)
        )
    )

    correct_height = (
        torch.abs(
            foot_z
            - target_z.unsqueeze(1)
        )
        <= z_tolerance_m
    )

    stable_contact = (
        (
            contact_time
            >= min_contact_time_s
        )
        & (
            contact_time
            <= max_reward_contact_time_s
        )
    )

    stability = (
        _stair_042_stability_gate(
            robot,
        )
        .unsqueeze(1)
    )

    score = torch.where(
        geo["active"].unsqueeze(1)
        & inside_tread
        & correct_height
        & stable_contact,
        stability,
        torch.zeros_like(stability),
    )

    return torch.max(
        score,
        dim=1,
    ).values


def terrain_supported_body_rise_042(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    x_entry_margin_m: float = 0.02,
    x_exit_margin_m: float = 0.04,
    z_tolerance_m: float = 0.03,
    min_contact_time_s: float = 0.08,
    vertical_speed_scale_mps: float = 0.15,
) -> torch.Tensor:
    """Reward upward body transfer only while a foot supports the tread.

    Static standing produces zero because positive root vertical velocity is
    required.
    """

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[
        sensor_cfg.name
    ]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    foot_pos = robot.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[
            :,
            0,
        ].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    contact_time = (
        sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    target_z = (
        flat_ankle_height_m
        + geo["next_surface"]
    )

    tread_start = (
        geo["edge"]
        + x_entry_margin_m
    )

    tread_end = (
        geo["edge"]
        + geo["tread"]
        - x_exit_margin_m
    )

    valid_support = (
        (
            foot_x
            >= tread_start.unsqueeze(1)
        )
        & (
            foot_x
            <= tread_end.unsqueeze(1)
        )
        & (
            torch.abs(
                foot_z
                - target_z.unsqueeze(1)
            )
            <= z_tolerance_m
        )
        & (
            contact_time
            >= min_contact_time_s
        )
    )

    any_support = torch.any(
        valid_support,
        dim=1,
    )

    root_height_gain = (
        robot.data.root_pos_w[:, 2]
        - geo["root_z0"]
    )

    body_progress = torch.clamp(
        (
            root_height_gain
            - geo["previous_surface"]
        )
        / torch.clamp(
            geo["riser"],
            min=1.0e-4,
        ),
        0.0,
        1.0,
    )

    upward_speed = torch.clamp(
        robot.data.root_lin_vel_w[:, 2]
        / max(
            vertical_speed_scale_mps,
            1.0e-6,
        ),
        0.0,
        1.0,
    )

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    root_follow = torch.clamp(
        (
            root_x
            - (
                geo["edge"]
                - 0.12
            )
        )
        / 0.20,
        0.0,
        1.0,
    )

    stability = _stair_042_stability_gate(
        robot,
    )

    score = (
        upward_speed
        * (
            0.25
            + 0.75 * body_progress
        )
        * root_follow
        * stability
    )

    return torch.where(
        geo["active"]
        & any_support,
        score,
        torch.zeros_like(score),
    )


def terrain_step_completion_event_042(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    flat_ankle_height_m: float = 0.035,
    x_entry_margin_m: float = 0.02,
    x_exit_margin_m: float = 0.04,
    z_tolerance_m: float = 0.03,
    min_contact_time_s: float = 0.10,
    min_body_progress: float = 0.70,
    min_root_follow: float = 0.65,
    max_tilt_rad: float = 0.31416,
) -> torch.Tensor:
    """One-time event: current stair has been physically completed.

    Completion requires:
      - a foot supported inside the current tread;
      - correct tread height;
      - sustained contact;
      - substantial body elevation;
      - body progression toward/over the support;
      - tilt below ~18 degrees.

    Only after this event does the next stair become reward-eligible.
    """

    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[
        sensor_cfg.name
    ]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    foot_pos = robot.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_x = (
        foot_pos[:, :, 0]
        - env.scene.env_origins[
            :,
            0,
        ].unsqueeze(1)
    )

    foot_z = (
        foot_pos[:, :, 2]
        - env.scene.env_origins[
            :,
            2,
        ].unsqueeze(1)
    )

    contact_time = (
        sensor.data.current_contact_time[
            :,
            sensor_cfg.body_ids,
        ]
    )

    target_z = (
        flat_ankle_height_m
        + geo["next_surface"]
    )

    tread_start = (
        geo["edge"]
        + x_entry_margin_m
    )

    tread_end = (
        geo["edge"]
        + geo["tread"]
        - x_exit_margin_m
    )

    valid_support = (
        (
            foot_x
            >= tread_start.unsqueeze(1)
        )
        & (
            foot_x
            <= tread_end.unsqueeze(1)
        )
        & (
            torch.abs(
                foot_z
                - target_z.unsqueeze(1)
            )
            <= z_tolerance_m
        )
        & (
            contact_time
            >= min_contact_time_s
        )
    )

    any_support = torch.any(
        valid_support,
        dim=1,
    )

    root_height_gain = (
        robot.data.root_pos_w[:, 2]
        - geo["root_z0"]
    )

    body_progress = torch.clamp(
        (
            root_height_gain
            - geo["previous_surface"]
        )
        / torch.clamp(
            geo["riser"],
            min=1.0e-4,
        ),
        0.0,
        1.0,
    )

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    root_follow = torch.clamp(
        (
            root_x
            - (
                geo["edge"]
                - 0.12
            )
        )
        / 0.20,
        0.0,
        1.0,
    )

    tilt_ok = (
        _stair_042_tilt(robot)
        <= max_tilt_rad
    )

    newly_completed = (
        geo["active"]
        & any_support
        & (
            body_progress
            >= min_body_progress
        )
        & (
            root_follow
            >= min_root_follow
        )
        & tilt_ok
    )

    reward = newly_completed.to(
        dtype=robot.data.root_pos_w.dtype
    )

    # This is deliberately a one-step state transition.
    geo["completed"][newly_completed] += 1

    return reward


def terrain_stair_tilt_excess_042(
    env,
    asset_cfg: SceneEntityCfg,
    activation_distance_m: float = 0.35,
    free_tilt_rad: float = 0.17453,
    strong_tilt_rad: float = 0.34907,
) -> torch.Tensor:
    """Penalize large stair-zone tilt while allowing normal weight shift.

    Approximately:
      <= 10 deg: no penalty
      10-20 deg: smooth penalty
      >= 20 deg: full penalty
    """

    robot = env.scene[asset_cfg.name]

    geo = _stair_042_target_geometry(
        env,
        robot,
    )

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    distance = geo["edge"] - root_x

    tilt = _stair_042_tilt(robot)

    excess = torch.clamp(
        (
            tilt
            - free_tilt_rad
        )
        / max(
            strong_tilt_rad
            - free_tilt_rad,
            1.0e-6,
        ),
        0.0,
        1.5,
    )

    active = (
        geo["active"]
        & (distance >= -0.12)
        & (
            distance
            <= activation_distance_m
        )
    )

    return torch.where(
        active,
        torch.square(excess),
        torch.zeros_like(excess),
    )

