"""Apply resolved reference-tracking configuration to StairEnvCfg."""

from __future__ import annotations

from typing import Any

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from . import reference_tracking


def _reference_observation_terms(
    metadata: str,
    start_frame: int,
    end_frame: int,
):
    common = {
        "asset_cfg": SceneEntityCfg("robot"),
        "metadata_path": metadata,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    return {
        "reference_joint_position_error": ObsTerm(
            func=reference_tracking.reference_joint_position_error,
            params=dict(common),
            clip=(-3.0, 3.0),
            history_length=1,
        ),
        "reference_joint_velocity_error": ObsTerm(
            func=reference_tracking.reference_joint_velocity_error,
            params=dict(common),
            clip=(-20.0, 20.0),
            history_length=1,
        ),
        "reference_phase": ObsTerm(
            func=reference_tracking.reference_phase,
            params={
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
            clip=(0.0, 1.0),
            history_length=1,
        ),
    }


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

    rewards.reference_joint_position_tracking.params.update(
        {
            "metadata_path": metadata,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "std_rad": position_std,
        }
    )

    rewards.reference_joint_velocity_tracking.params.update(
        {
            "metadata_path": metadata,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "std_radps": velocity_std,
        }
    )

    conditioning_enabled = bool(
        cfg.get("conditioning_enabled", False)
    )

    if conditioning_enabled:
        terms = _reference_observation_terms(
            metadata,
            start_frame,
            end_frame,
        )

        for group_name in ("policy", "critic"):
            group = getattr(
                env_cfg.observations,
                group_name,
            )

            for name, term in terms.items():
                setattr(group, name, term)

    return {
        "enabled": True,
        "conditioning_enabled": conditioning_enabled,
        "metadata": metadata,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "joint_position_std_rad": position_std,
        "joint_velocity_std_radps": velocity_std,
        "conditioning_observations": (
            59 if conditioning_enabled else 0
        ),
    }
