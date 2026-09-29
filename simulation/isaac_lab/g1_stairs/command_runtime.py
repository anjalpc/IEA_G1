"""Experiment-driven command configuration for geometry locomotion."""

from __future__ import annotations

import math
from typing import Any


def apply_stair_heading_command(
    env_cfg: Any,
    resolved: dict,
) -> dict[str, float] | None:
    """Enable Isaac Lab absolute-heading control without changing command size.

    The Unitree actor still receives exactly three command channels
    ``[vx, vy, wz]``.  Isaac Lab derives ``wz`` from the world-frame heading
    error and clips it to the configured yaw-rate range.
    """

    command = resolved.get("command", {})
    if not isinstance(command, dict):
        raise ValueError("command must be a mapping")

    heading = command.get("stair_heading", {})
    if heading in (None, {}):
        return None
    if not isinstance(heading, dict):
        raise ValueError("command.stair_heading must be a mapping")
    if not bool(heading.get("enabled", False)):
        return None

    target_heading_rad = float(
        heading.get("target_heading_rad", 0.0)
    )
    heading_control_stiffness = float(
        heading.get("control_stiffness", 1.0)
    )
    max_yaw_rate_radps = float(
        heading.get("max_yaw_rate_radps", 0.6)
    )

    if not -math.pi <= target_heading_rad <= math.pi:
        raise ValueError(
            "command.stair_heading.target_heading_rad must be in [-pi, pi]"
        )
    if heading_control_stiffness <= 0.0:
        raise ValueError(
            "command.stair_heading.control_stiffness must be > 0"
        )
    if not 0.0 < max_yaw_rate_radps <= 2.0:
        raise ValueError(
            "command.stair_heading.max_yaw_rate_radps must be in (0, 2]"
        )

    base_velocity = env_cfg.commands.base_velocity

    # UniformVelocityCommandCfg uses ranges.ang_vel_z to clip the yaw-rate
    # generated from heading error, so it must not remain at the nominal
    # zero-yaw range when heading mode is enabled.
    base_velocity.heading_command = True
    base_velocity.rel_heading_envs = 1.0
    base_velocity.rel_standing_envs = 0.0
    base_velocity.heading_control_stiffness = (
        heading_control_stiffness
    )
    base_velocity.ranges.heading = (
        target_heading_rad,
        target_heading_rad,
    )
    base_velocity.ranges.ang_vel_z = (
        -max_yaw_rate_radps,
        max_yaw_rate_radps,
    )

    # Keep limit ranges consistent for any command curriculum/runtime code.
    if hasattr(base_velocity, "limit_ranges"):
        base_velocity.limit_ranges.heading = (
            target_heading_rad,
            target_heading_rad,
        )
        base_velocity.limit_ranges.ang_vel_z = (
            -max_yaw_rate_radps,
            max_yaw_rate_radps,
        )

    return {
        "target_heading_rad": target_heading_rad,
        "control_stiffness": heading_control_stiffness,
        "max_yaw_rate_radps": max_yaw_rate_radps,
    }


def platform_finish_speed(
    local_x, stop_x, cruise_speed, feet_safe, settling, reset,
    *, slowdown_distance=0.35, creep_speed=0.08, resume_x=None,
):
    """Pure tensor command logic; preserve settling until episode reset."""
    import torch

    if not (0 < creep_speed <= cruise_speed <= 0.50):
        raise ValueError("require 0 < creep <= cruise <= 0.50")
    if not math.isfinite(slowdown_distance) or slowdown_distance <= 0:
        raise ValueError("slowdown_distance must be positive and finite")
    if resume_x is None:
        resume_x = stop_x - 0.10
    if resume_x >= stop_x:
        raise ValueError("resume_x must be below stop_x")
    settling = (settling & ~reset & (local_x >= resume_x)) | (
        ~reset & feet_safe & (local_x >= stop_x)
    )
    speed = (
        cruise_speed * ((stop_x - local_x) / slowdown_distance).clamp(0.0, 1.0)
    ).clamp_min(creep_speed)
    return torch.where(settling, torch.zeros_like(speed), speed), settling
