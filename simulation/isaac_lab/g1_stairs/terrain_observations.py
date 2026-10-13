"""Compact physical staircase observations for unified stair locomotion."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg

from .contracts import G1_SOLE_SPHERE_RADIUS_M
from .terrain_state import riser_height_per_env, stair_cfg


def _preferred_foothold_features(
    *,
    preferred: torch.Tensor,
    current_target: torch.Tensor,
    riser_height: torch.Tensor,
    rear_x: torch.Tensor,
    front_x: torch.Tensor,
    sole_bottom: torch.Tensor,
    first_riser: float,
    tread_depth: float,
    step_count: int,
    tread_margin_m: float,
):
    """Return count-independent preferred-foot foothold errors."""

    if tread_depth <= 0.0:
        raise ValueError("tread_depth must be positive")
    if tread_margin_m <= 0.0:
        raise ValueError("tread_margin_m must be positive")
    if not 2.0 * tread_margin_m < tread_depth:
        raise ValueError("tread margin leaves no safe landing interval")

    valid = (
        (preferred >= 0)
        & (current_target > 0)
        & (current_target < int(step_count))
        & (riser_height > 1.0e-8)
    )

    pref_idx = preferred.clamp(0, 1)[:, None]

    pref_rear = rear_x.gather(1, pref_idx).squeeze(1)
    pref_front = front_x.gather(1, pref_idx).squeeze(1)
    pref_bottom = sole_bottom.gather(1, pref_idx).squeeze(1)

    # Current X projection changes with foot orientation.
    projected_span = (
        pref_front - pref_rear
    ).clamp_min(0.0)

    target_f = current_target.to(pref_rear.dtype)

    # Use acquisition runtime target, not base/root "next riser".
    target_riser_x = (
        float(first_riser)
        + target_f * float(tread_depth)
    )

    safe_start = (
        target_riser_x
        + float(tread_margin_m)
    )

    safe_end = (
        target_riser_x
        + float(tread_depth)
        - float(tread_margin_m)
    )

    # Complete current oriented sole must fit inside the tread.
    max_safe_rear = (
        safe_end - projected_span
    )

    feasible = (
        valid
        & (max_safe_rear >= safe_start)
    )

    # Midpoint of feasible rear-edge interval.
    target_rear = 0.5 * (
        safe_start + max_safe_rear
    )

    target_tread_z = (
        target_f + 1.0
    ) * riser_height

    zero = torch.zeros_like(pref_rear)

    # Mask BEFORE division.
    dx_raw = torch.where(
        feasible,
        target_rear - pref_rear,
        zero,
    )

    dz_raw = torch.where(
        feasible,
        target_tread_z - pref_bottom,
        zero,
    )

    # Geometry-derived scaling:
    # 2*tread + current preferred-foot projected X span.
    dx_scale = (
        2.0 * float(tread_depth)
        + projected_span
    ).clamp_min(1.0e-6)

    # Never divide flat / invalid rows by zero.
    dz_scale = torch.where(
        feasible,
        riser_height,
        torch.ones_like(riser_height),
    )

    dx = dx_raw / dx_scale
    dz = dz_raw / dz_scale

    features = torch.stack(
        (dx, dz),
        dim=1,
    )

    if not torch.isfinite(features).all():
        raise RuntimeError(
            "non-finite preferred foothold observation"
        )

    return (
        features,
        feasible,
        target_rear,
        safe_start,
        safe_end,
        projected_span,
    )


def stair_geometry(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return deployable local stair observations.

    Base geometry:
      0 stairs ahead
      1 normalized riser height
      2 normalized tread depth
      3 relative staircase yaw
      4 distance to next geometric riser

    C1 optionally appends:
      5 preferred-left
      6 preferred-right

    C2 optionally appends:
      7 preferred-foot rear-edge error to desired next-tread foothold
      8 preferred-foot sole-height error to next-tread surface

    C1/C2 transition features are zero before the first verified stair
    acquisition and after the final stair acquisition.
    """

    robot = env.scene[asset_cfg.name]
    cfg = stair_cfg(env)

    riser_height = riser_height_per_env(env)

    tread_depth = float(cfg.tread_depth)
    step_count = int(cfg.step_count)
    first_riser = float(cfg.start_clearance)

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    platform_start = (
        first_riser
        + step_count * tread_depth
    )

    has_stairs = (
        riser_height > 1.0e-8
    ).to(root_x.dtype)

    active = (
        root_x < platform_start
    ).to(root_x.dtype) * has_stairs

    edges = (
        first_riser
        + torch.arange(
            step_count,
            dtype=root_x.dtype,
            device=root_x.device,
        )
        * tread_depth
    )

    distances = (
        edges.unsqueeze(0)
        - root_x.unsqueeze(1)
    )

    upcoming = torch.where(
        distances >= 0.0,
        distances,
        torch.full_like(
            distances,
            torch.inf,
        ),
    )

    next_distance = torch.min(
        upcoming,
        dim=1,
    ).values

    next_distance = torch.where(
        torch.isfinite(next_distance),
        next_distance,
        torch.zeros_like(next_distance),
    )

    next_distance = torch.clamp(
        next_distance,
        0.0,
        1.0,
    )

    qw, qx, qy, qz = (
        robot.data.root_quat_w.unbind(dim=1)
    )

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    relative_yaw = torch.atan2(
        torch.sin(-yaw),
        torch.cos(-yaw),
    ) / math.pi

    geometry = torch.stack(
        (
            active,
            active * (riser_height / 0.20),
            active * (tread_depth / 0.25),
            active * relative_yaw,
            active * next_distance,
        ),
        dim=1,
    )

    # --------------------------------------------------------
    # C1: preferred next swing-foot identity
    # --------------------------------------------------------
    if not bool(
        getattr(
            env.cfg,
            "preferred_transition_foot_observation",
            False,
        )
    ):
        return geometry

    transition = torch.zeros(
        (env.num_envs, 2),
        dtype=root_x.dtype,
        device=root_x.device,
    )

    runtime = getattr(
        env,
        "_g1_v04_acquisition",
        None,
    )

    valid = torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=root_x.device,
    )

    preferred = torch.full(
        (env.num_envs,),
        -1,
        dtype=torch.long,
        device=root_x.device,
    )

    current_target = torch.zeros(
        env.num_envs,
        dtype=torch.long,
        device=root_x.device,
    )

    if (
        runtime is not None
        and bool(runtime.dynamic_alternating_rewards)
    ):
        preferred = runtime.preferred_foot
        current_target = runtime.target_step[:, 0]

        valid = (
            (preferred >= 0)
            & (current_target > 0)
            & (current_target < step_count)
            & (riser_height > 1.0e-8)
        )

        transition[:, 0] = (
            valid & (preferred == 0)
        ).to(root_x.dtype)

        transition[:, 1] = (
            valid & (preferred == 1)
        ).to(root_x.dtype)

    c1 = torch.cat(
        (geometry, transition),
        dim=1,
    )

    if getattr(env.cfg, "gait_phase_enabled", False):
        clock = env.action_manager.get_term("gait_frequency").observation()
        return torch.cat((c1, clock), dim=1)

    # --------------------------------------------------------
    # C2: preferred-foot next-tread foothold error
    # --------------------------------------------------------
    foothold_enabled = bool(
        getattr(
            env.cfg,
            "preferred_foothold_observation",
            False,
        )
    )

    foothold_ablation = bool(
        getattr(
            env.cfg,
            "preferred_foothold_observation_ablation",
            False,
        )
    )

    if foothold_enabled and foothold_ablation:
        raise RuntimeError(
            "preferred foothold observation and ablation "
            "cannot both be enabled"
        )

    if not foothold_enabled and not foothold_ablation:
        return c1

    foothold = torch.zeros(
        (env.num_envs, 2),
        dtype=root_x.dtype,
        device=root_x.device,
    )

    if runtime is not None:
        data = runtime.robot.data

        foot_pos = (
            data.body_pos_w[:, runtime.robot_foot_ids]
            - env.scene.env_origins[:, None, :]
        )

        foot_quat = data.body_quat_w[
            :, runtime.robot_foot_ids
        ]

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

        q_xyz = quaternions[..., 1:]

        uv = torch.cross(
            q_xyz,
            offsets,
            dim=-1,
        )

        rotated = (
            offsets
            + 2.0
            * (
                quaternions[..., :1] * uv
                + torch.cross(
                    q_xyz,
                    uv,
                    dim=-1,
                )
            )
        )

        sole_points = (
            foot_pos.unsqueeze(2)
            + rotated
        )

        radius = float(
            G1_SOLE_SPHERE_RADIUS_M
        )

        rear_x = (
            sole_points[..., 0].amin(dim=2)
            - radius
        )

        front_x = (
            sole_points[..., 0].amax(dim=2)
            + radius
        )

        sole_bottom = (
            sole_points[..., 2].amin(dim=2)
            - radius
        )

        (
            foothold,
            _feasible,
            _target_rear,
            _safe_start,
            _safe_end,
            _projected_span,
        ) = _preferred_foothold_features(
            preferred=preferred,
            current_target=current_target,
            riser_height=riser_height,
            rear_x=rear_x,
            front_x=front_x,
            sole_bottom=sole_bottom,
            first_riser=first_riser,
            tread_depth=tread_depth,
            step_count=step_count,
            tread_margin_m=float(
                getattr(
                    env.cfg,
                    "preferred_foothold_tread_margin_m",
                    0.015,
                )
            ),
        )

    # Matched model309 continuation:
    # retain identical 489-D architecture but zero C2 information.
    if foothold_ablation:
        foothold.zero_()

    return torch.cat(
        (c1, foothold),
        dim=1,
    )

def stair_velocity_command(
    env,
    approach_velocity_mps: float = 0.50,
    stair_velocity_mps: float = 0.18,
    slowdown_start_distance_m: float = 0.60,
    slowdown_end_distance_m: float = 0.35,
    platform_velocity_mps: float = 0.50,
    platform_accel_distance_m: float = 0.40,
    centerline_gain_s: float = 0.0,
    max_lateral_velocity_mps: float = 0.0,
    heading_gain_s: float = 0.0,
    max_yaw_rate_radps: float = 0.0,
    stop_after_ascent_start: bool = False,
) -> torch.Tensor:
    """Return geometry-conditioned [vx, vy, yaw_rate] command.

    The observation dimensionality remains exactly three.

    - walk normally on the flat approach;
    - slow smoothly before the first riser;
    - climb all three treads carefully;
    - accelerate smoothly after reaching the platform.
    """

    if not (
        0.0 < stair_velocity_mps
        <= approach_velocity_mps
        <= 0.50
    ):
        raise ValueError(
            "expected 0 < stair_velocity <= approach_velocity <= 0.50"
        )

    if not (
        slowdown_start_distance_m
        > slowdown_end_distance_m
        > 0.0
    ):
        raise ValueError(
            "slowdown_start_distance_m must exceed "
            "slowdown_end_distance_m > 0"
        )

    if platform_accel_distance_m <= 0.0:
        raise ValueError(
            "platform_accel_distance_m must be positive"
        )

    for name, value in (
        ("centerline_gain_s", centerline_gain_s),
        ("max_lateral_velocity_mps", max_lateral_velocity_mps),
        ("heading_gain_s", heading_gain_s),
        ("max_yaw_rate_radps", max_yaw_rate_radps),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    if (centerline_gain_s > 0.0) != (max_lateral_velocity_mps > 0.0):
        raise ValueError(
            "centerline correction requires positive gain and lateral limit"
        )

    if (heading_gain_s > 0.0) != (max_yaw_rate_radps > 0.0):
        raise ValueError(
            "heading correction requires positive gain and yaw-rate limit"
        )

    robot = env.scene["robot"]
    cfg = stair_cfg(env)

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )
    root_y = (
        robot.data.root_pos_w[:, 1]
        - env.scene.env_origins[:, 1]
    )

    yaw = torch.zeros_like(root_x)
    if heading_gain_s > 0.0:
        qw, qx, qy, qz = (
            robot.data.root_quat_w.unbind(dim=1)
        )
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        yaw = torch.atan2(
            torch.sin(yaw),
            torch.cos(yaw),
        )

    first_riser = float(cfg.start_clearance)

    platform_start = (
        first_riser
        + int(cfg.step_count) * float(cfg.tread_depth)
    )

    distance_to_first = first_riser - root_x

    # 0 before slowdown starts, 1 once the careful-speed zone begins.
    slowdown_alpha = torch.clamp(
        (
            slowdown_start_distance_m
            - distance_to_first
        )
        / (
            slowdown_start_distance_m
            - slowdown_end_distance_m
        ),
        0.0,
        1.0,
    )

    approach_speed = (
        approach_velocity_mps
        + slowdown_alpha
        * (
            stair_velocity_mps
            - approach_velocity_mps
        )
    )

    speed = torch.where(
        root_x < first_riser,
        approach_speed,
        torch.full_like(
            root_x,
            stair_velocity_mps,
        ),
    )

    # Stay slow at platform entry, then smoothly return to walking speed.
    platform_alpha = torch.clamp(
        (
            root_x
            - platform_start
        )
        / platform_accel_distance_m,
        0.0,
        1.0,
    )

    platform_speed = (
        stair_velocity_mps
        + platform_alpha
        * (
            platform_velocity_mps
            - stair_velocity_mps
        )
    )

    speed = torch.where(
        root_x >= platform_start,
        platform_speed,
        speed,
    )

    lateral = torch.zeros_like(speed)
    if centerline_gain_s > 0.0:
        lateral = torch.clamp(
            -centerline_gain_s * root_y,
            -max_lateral_velocity_mps,
            max_lateral_velocity_mps,
        )

    yaw_rate = torch.zeros_like(speed)
    if heading_gain_s > 0.0:
        yaw_rate = torch.clamp(
            -heading_gain_s * yaw,
            -max_yaw_rate_radps,
            max_yaw_rate_radps,
        )

    if stop_after_ascent_start:
        raise RuntimeError(
            "stop_after_ascent_start belonged to the archived structured-step runtime"
        )

    return torch.stack(
        (
            speed,
            lateral,
            yaw_rate,
        ),
        dim=1,
    )
