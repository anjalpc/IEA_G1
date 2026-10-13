"""Compact physical staircase observations for unified stair locomotion."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg

from .terrain_state import riser_height_per_env, stair_cfg


def stair_geometry(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return five normalized physical stair-state features.

    Features:
      0 stairs ahead
      1 actual per-environment riser height
      2 tread depth
      3 relative staircase yaw
      4 distance to next riser
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

    # Flat rows use a zero riser and must advertise no staircase. This
    # makes the five explicit geometry features causally different from
    # the stair rows instead of presenting a fictitious zero-height stair.
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

    # C1: expose only the locally preferred next swing foot.
    # This is independent of absolute stair count and therefore scales
    # to arbitrary-length staircases once the acquisition runtime does.
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

    runtime = getattr(env, "_g1_v04_acquisition", None)

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

        # Runtime foot ordering is [left, right].
        transition[:, 0] = (
            valid & (preferred == 0)
        ).to(root_x.dtype)
        transition[:, 1] = (
            valid & (preferred == 1)
        ).to(root_x.dtype)

    return torch.cat((geometry, transition), dim=1)


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
