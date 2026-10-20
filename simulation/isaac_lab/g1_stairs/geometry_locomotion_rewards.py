"""Small task-frame rewards for geometry-conditioned locomotion."""

from __future__ import annotations

import torch
from .reward_math import (
    exp_tracking_score,
    finish_transition_gate,
    normalized_planar_velocity_error,
)



def forward_stall(
    env,
    minimum_forward_speed_mps: float = 0.10,
    command_name: str = "base_velocity",
    asset_cfg=None,
):
    """Penalize failure to move forward while forward motion is commanded."""
    if minimum_forward_speed_mps <= 0.0:
        raise ValueError("minimum_forward_speed_mps must be positive")

    robot = env.scene[asset_cfg.name]
    forward_speed = robot.data.root_lin_vel_w[:, 0]
    forward_command = env.command_manager.get_command(command_name)[:, 0]

    deficit = torch.clamp(
        (minimum_forward_speed_mps - forward_speed)
        / minimum_forward_speed_mps,
        min=0.0,
        max=1.0,
    )
    return deficit * (forward_command > minimum_forward_speed_mps)


def geometry_failure_termination_event(env) -> torch.Tensor:
    """One-shot penalty for geometry-contract failures, never success.

    Uses the native non-timeout termination flag but explicitly excludes the
    shared geometry SUCCESS reason. This closes suicide/course-boundary escape
    without penalizing valid task completion.
    """
    from .geometry_episode import Result

    terminated = getattr(env, "reset_terminated", None)
    if terminated is None:
        raise RuntimeError(
            "geometry_failure_termination_event requires env.reset_terminated"
        )
    runtime = getattr(env, "_g1_geometry_episode", None)
    if runtime is None:
        raise RuntimeError(
            "geometry_failure_termination_event requires geometry episode runtime"
        )
    if float(env.step_dt) <= 0.0:
        raise ValueError("reward step dt must be positive")

    failure = terminated.to(dtype=torch.bool) & (
        runtime.tracker.reason != Result.SUCCESS
    )
    return failure.to(dtype=torch.float32) / float(env.step_dt)


def track_ang_vel_z_zero_centered(
    env,
    std: float,
    command_name: str,
    asset_cfg=None,
) -> torch.Tensor:
    """Track yaw-rate command with the Unitree exponential score minus one."""

    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    robot = env.scene[asset_name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - robot.data.root_ang_vel_b[:, 2]
    )
    return exp_tracking_score(
        ang_vel_error,
        std * std,
        zero_centered=True,
    )


def track_stair_axis_velocity(
    env,
    target_velocity_mps: float = 0.35,
    std_mps: float = 0.25,
    asset_cfg=None,
    command_name: str | None = None,
    ascent_scale: float = 1.0,
) -> torch.Tensor:
    """Reward world/stair-frame +X velocity while suppressing lateral speed."""

    if not 0.0 <= target_velocity_mps <= 0.50:
        raise ValueError("target_velocity_mps must be in [0.0, 0.50]")
    if std_mps <= 0.0:
        raise ValueError("std_mps must be positive")
    if not 0.0 <= ascent_scale <= 1.0:
        raise ValueError("ascent_scale must be in [0, 1]")

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
    score = exp_tracking_score(
        squared_error,
        std_mps * std_mps,
        zero_centered=False,
    )
    if ascent_scale < 1.0:
        from .terrain_state import riser_height_per_env, stair_cfg
        cfg = stair_cfg(env)
        local_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        platform_start = float(cfg.start_clearance) + int(cfg.step_count) * float(cfg.tread_depth)
        on_stairs = (
            (riser_height_per_env(env) > 1.0e-8)
            & (local_x >= float(cfg.start_clearance) - 0.10)
            & (local_x < platform_start)
        )
        scale = torch.where(
            on_stairs,
            torch.full_like(score, float(ascent_scale)),
            torch.ones_like(score),
        )
        score = score * scale
    return score


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


def finish_arm_target_return(
    env,
    cruise_speed_mps: float = 0.35,
    command_name: str = "base_velocity",
    asset_cfg=None,
) -> torch.Tensor:
    """Gently return arm position targets to defaults only during finish."""

    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    robot = env.scene[asset_name]
    target = robot.data.joint_pos_target
    if target is None:
        raise RuntimeError(
            "finish arm target return requires joint_pos_target"
        )

    joint_ids = slice(None)
    if asset_cfg is not None and asset_cfg.joint_ids is not None:
        joint_ids = asset_cfg.joint_ids

    target_error = (
        target[:, joint_ids]
        - robot.data.default_joint_pos[:, joint_ids]
    )
    command_x = env.command_manager.get_command(command_name)[:, 0]
    gate = finish_transition_gate(
        command_x,
        cruise_speed_mps=cruise_speed_mps,
    )
    return gate * torch.mean(target_error.square(), dim=1)


def stable_platform_finish(env):
    """Reward bounded non-recyclable settling progress and final success."""
    from .geometry_episode import Result

    runtime = getattr(env, "_g1_geometry_episode", None)
    if runtime is None or not runtime.contract.stable_finish:
        return torch.zeros(env.num_envs, device=env.device)

    tracker = runtime.tracker
    required = float(tracker.cfg.supported_dwell_s)
    settling_progress = tracker.dwell_peak_increment / required
    completed = (tracker.reason == Result.SUCCESS).to(torch.float32)
    return (settling_progress + completed) / float(env.step_dt)
