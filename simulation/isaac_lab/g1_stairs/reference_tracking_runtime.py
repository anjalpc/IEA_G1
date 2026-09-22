"""Apply resolved reference-tracking configuration to StairEnvCfg."""

from __future__ import annotations

from typing import Any


def apply_reference_tracking_config(
    env_cfg: Any,
    resolved: dict,
) -> dict:
    cfg = resolved.get("reference_tracking", {})

    if not isinstance(cfg, dict):
        raise ValueError("reference_tracking must be a mapping")

    if not cfg.get("enabled", False):
        return {}

    metadata = cfg.get("metadata")
    if not metadata:
        raise ValueError(
            "reference_tracking.metadata is required when enabled"
        )

    start_frame = int(cfg.get("start_frame", 0))
    end_frame = int(cfg.get("end_frame", 260))

    if end_frame < start_frame:
        raise ValueError(
            "reference_tracking.end_frame must be >= start_frame"
        )

    position_std = float(
        cfg.get("joint_position_std_rad", 0.35)
    )

    velocity_std = float(
        cfg.get("joint_velocity_std_radps", 2.0)
    )

    rewards = env_cfg.rewards

    pos_term = rewards.reference_joint_position_tracking
    vel_term = rewards.reference_joint_velocity_tracking

    pos_term.params.update(
        {
            "metadata_path": metadata,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "std_rad": position_std,
        }
    )

    vel_term.params.update(
        {
            "metadata_path": metadata,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "std_radps": velocity_std,
        }
    )

    return {
        "enabled": True,
        "metadata": metadata,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "joint_position_std_rad": position_std,
        "joint_velocity_std_radps": velocity_std,
    }
