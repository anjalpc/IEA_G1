"""Small task-frame rewards for geometry-conditioned locomotion."""

from __future__ import annotations

import torch
from .reward_math import exp_tracking_score


def track_stair_axis_velocity(
    env,
    target_velocity_mps: float = 0.35,
    std_mps: float = 0.25,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward world/stair-frame +X velocity while suppressing lateral speed."""

    if not 0.0 <= target_velocity_mps <= 0.50:
        raise ValueError("target_velocity_mps must be in [0.0, 0.50]")
    if std_mps <= 0.0:
        raise ValueError("std_mps must be positive")

    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    robot = env.scene[asset_name]
    velocity = robot.data.root_lin_vel_w[:, :2]
    squared_error = (
        torch.square(velocity[:, 0] - target_velocity_mps)
        + torch.square(velocity[:, 1])
    )
    return exp_tracking_score(
        squared_error,
        std_mps * std_mps,
        zero_centered=False,
    )


def stair_centerline_error(
    env,
    scale_m: float = 0.25,
    asset_cfg=None,
) -> torch.Tensor:
    """Normalized absolute lateral displacement from the stair centerline."""

    if scale_m <= 0.0:
        raise ValueError("scale_m must be positive")

    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    robot = env.scene[asset_name]
    local_y = (
        robot.data.root_pos_w[:, 1]
        - env.scene.env_origins[:, 1]
    )
    return torch.abs(local_y) / float(scale_m)
