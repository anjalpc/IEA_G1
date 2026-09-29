"""Small task-frame rewards for geometry-conditioned locomotion."""

from __future__ import annotations

import torch
from .reward_math import (
    exp_tracking_score,
    finish_transition_gate,
    normalized_planar_velocity_error,
)


def track_stair_axis_velocity(
    env,
    target_velocity_mps: float = 0.35,
    std_mps: float = 0.25,
    asset_cfg=None,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward world/stair-frame +X velocity while suppressing lateral speed."""

    if not 0.0 <= target_velocity_mps <= 0.50:
        raise ValueError("target_velocity_mps must be in [0.0, 0.50]")
    if std_mps <= 0.0:
        raise ValueError("std_mps must be positive")

    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    robot = env.scene[asset_name]
    velocity = robot.data.root_lin_vel_w[:, :2]
    target = target_velocity_mps
    if command_name is not None:
        # Opt-in: use the forward command presented to the policy.
        target = env.command_manager.get_command(command_name)[:, 0]
    squared_error = (
        torch.square(velocity[:, 0] - target)
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



def finish_planar_velocity_error(
    env,
    cruise_speed_mps: float = 0.35,
    scale_mps: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg=None,
) -> torch.Tensor:
    """Penalize residual planar momentum only during the finish transition."""

    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    robot = env.scene[asset_name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    gate = finish_transition_gate(
        command_x, cruise_speed_mps=cruise_speed_mps,
    )
    severity = normalized_planar_velocity_error(
        robot.data.root_lin_vel_w[:, :2],
        command_x,
        scale_mps=scale_mps,
    )
    return gate * severity


def finish_centerline_error(
    env,
    cruise_speed_mps: float = 0.35,
    scale_m: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg=None,
) -> torch.Tensor:
    """Increase centerline pressure only while slowing for the platform finish."""

    if scale_m <= 0.0:
        raise ValueError("scale_m must be positive")
    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    robot = env.scene[asset_name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    gate = finish_transition_gate(
        command_x, cruise_speed_mps=cruise_speed_mps,
    )
    local_y = robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return gate * (torch.abs(local_y) / float(scale_m)).clamp(0.0, 2.0)

def stable_platform_finish(env):
    """Reward ongoing verified settling and a one-shot successful finish."""
    from .geometry_episode import Result
    runtime = getattr(env, "_g1_geometry_episode", None)
    if runtime is None or not runtime.contract.stable_finish:
        return torch.zeros(env.num_envs, device=env.device)
    tracker = runtime.tracker
    settling = (tracker.dwell > 0).to(torch.float32)
    completed = (tracker.reason == Result.SUCCESS).to(torch.float32)
    return settling + completed / env.step_dt
