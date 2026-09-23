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
