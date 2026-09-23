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

    active = (
        root_x < platform_start
    ).to(root_x.dtype)

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

    return torch.stack(
        (
            active,
            active * (riser_height / 0.20),
            active * (tread_depth / 0.25),
            active * relative_yaw,
            active * next_distance,
        ),
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

    robot = env.scene["robot"]
    cfg = stair_cfg(env)

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
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

    zeros = torch.zeros_like(speed)

    return torch.stack(
        (
            speed,
            zeros,
            zeros,
        ),
        dim=1,
    )
