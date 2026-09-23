"""Compact physical staircase observations for unified stair locomotion."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg


def stair_geometry(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return five normalized state-dependent stair features.

    Features:
      0. stairs still ahead
      1. normalized riser height
      2. normalized tread depth
      3. relative stair yaw
      4. distance to next riser

    Geometry comes from the live terrain configuration.
    No reference clock, gait phase, or prescribed swing leg is used.
    """
    robot = env.scene[asset_cfg.name]

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

    # env origin is the robot's fixed starting position.
    # start_clearance is therefore the robot-relative X of the first riser.
    first_riser_x_m = float(
        stair_cfg.start_clearance
    )

    root_x = (
        robot.data.root_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    platform_start_x_m = (
        first_riser_x_m
        + step_count * tread_depth_m
    )

    stairs_ahead = (
        root_x < platform_start_x_m
    )

    riser_edges = (
        first_riser_x_m
        + torch.arange(
            step_count,
            dtype=root_x.dtype,
            device=root_x.device,
        )
        * tread_depth_m
    )

    distances = (
        riser_edges.unsqueeze(0)
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

    next_riser_distance = torch.min(
        upcoming,
        dim=1,
    ).values

    next_riser_distance = torch.where(
        torch.isfinite(next_riser_distance),
        next_riser_distance,
        torch.zeros_like(
            next_riser_distance
        ),
    )

    # One metre covers the complete nominal approach to the first riser.
    next_riser_distance = torch.clamp(
        next_riser_distance,
        min=0.0,
        max=1.0,
    )

    # Isaac Lab root quaternion is WXYZ.
    qw, qx, qy, qz = (
        robot.data.root_quat_w.unbind(dim=1)
    )

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    # Staircase direction is +X, yaw = 0.
    relative_yaw = torch.atan2(
        torch.sin(-yaw),
        torch.cos(-yaw),
    ) / math.pi

    active = stairs_ahead.to(
        root_x.dtype
    )

    return torch.stack(
        (
            active,
            active * (
                riser_height_m / 0.20
            ),
            active * (
                tread_depth_m / 0.25
            ),
            active * relative_yaw,
            active * next_riser_distance,
        ),
        dim=1,
    )
