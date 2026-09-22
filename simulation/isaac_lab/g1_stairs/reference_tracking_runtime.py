"""Apply resolved reference-tracking configuration to StairEnvCfg."""

from __future__ import annotations

from typing import Any

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)

from . import reference_tracking
from .contracts import BODY_JOINTS
from .reference_reset import ReferenceStateReset
from .reference_residual_action import (
    ReferenceResidualJointPositionActionCfg,
)


def _reference_observation_terms(
    metadata: str,
    start_frame: int,
    end_frame: int,
    root_conditioning_enabled: bool,
):
    common = {
        "asset_cfg": SceneEntityCfg("robot"),
        "metadata_path": metadata,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    terms = {
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

    if root_conditioning_enabled:
        terms.update(
            {
                "reference_root_position_error": ObsTerm(
                    func=reference_tracking.reference_root_position_error,
                    params=dict(common),
                    clip=(-1.5, 1.5),
                    history_length=1,
                ),
                "reference_root_orientation_error": ObsTerm(
                    func=reference_tracking.reference_root_orientation_error,
                    params=dict(common),
                    clip=(-1.5, 1.5),
                    history_length=1,
                ),
                "reference_root_linear_velocity_error": ObsTerm(
                    func=reference_tracking.reference_root_linear_velocity_error,
                    params=dict(common),
                    clip=(-3.0, 3.0),
                    history_length=1,
                ),
                "reference_root_angular_velocity_error": ObsTerm(
                    func=reference_tracking.reference_root_angular_velocity_error,
                    params=dict(common),
                    clip=(-5.0, 5.0),
                    history_length=1,
                ),
            }
        )

    return terms


def apply_reference_tracking_config(
    env_cfg: Any,
    resolved: dict,
) -> dict:
    cfg = resolved.get(
        "reference_tracking",
        {},
    )

    if not isinstance(cfg, dict):
        raise ValueError(
            "reference_tracking must be a mapping"
        )

    if not cfg.get(
        "enabled",
        False,
    ):
        return {}

    metadata = cfg.get("metadata")

    if not metadata:
        raise ValueError(
            "reference_tracking.metadata is required "
            "when enabled"
        )

    start_frame = int(
        cfg.get(
            "start_frame",
            0,
        )
    )

    end_frame = int(
        cfg.get(
            "end_frame",
            260,
        )
    )

    frequency_hz = float(
        cfg.get(
            "frequency_hz",
            50.0,
        )
    )

    if end_frame < start_frame:
        raise ValueError(
            "reference_tracking.end_frame must be "
            ">= start_frame"
        )

    if frequency_hz <= 0.0:
        raise ValueError(
            "reference_tracking.frequency_hz "
            "must be > 0"
        )

    terminate_at_reference_end = bool(
        cfg.get(
            "terminate_at_reference_end",
            False,
        )
    )

    reference_control_steps = (
        end_frame
        - start_frame
        + 1
    )

    reference_duration_s = (
        reference_control_steps
        / frequency_hz
    )

    if terminate_at_reference_end:
        env_cfg.episode_length_s = (
            reference_duration_s
        )

    position_std = float(
        cfg.get(
            "joint_position_std_rad",
            0.35,
        )
    )

    velocity_std = float(
        cfg.get(
            "joint_velocity_std_radps",
            2.0,
        )
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

    # ---------------------------------------------------------
    # Whole-body root tracking
    # ---------------------------------------------------------
    root_cfg = cfg.get(
        "root_tracking",
        {},
    )

    if root_cfg is None:
        root_cfg = {}

    if not isinstance(
        root_cfg,
        dict,
    ):
        raise ValueError(
            "reference_tracking.root_tracking "
            "must be a mapping"
        )

    root_tracking_enabled = bool(
        root_cfg.get(
            "enabled",
            False,
        )
    )

    root_conditioning_enabled = bool(
        root_cfg.get(
            "conditioning_enabled",
            root_tracking_enabled,
        )
    )

    root_position_std_m = float(
        root_cfg.get(
            "position_std_m",
            0.15,
        )
    )

    root_orientation_std_rad = float(
        root_cfg.get(
            "orientation_std_rad",
            0.35,
        )
    )

    root_linear_velocity_std_mps = float(
        root_cfg.get(
            "linear_velocity_std_mps",
            0.50,
        )
    )

    root_angular_velocity_std_radps = float(
        root_cfg.get(
            "angular_velocity_std_radps",
            1.0,
        )
    )

    if root_tracking_enabled:
        rewards.reference_root_position_tracking.params.update(
            {
                "metadata_path": metadata,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "std_m": root_position_std_m,
            }
        )

        rewards.reference_root_orientation_tracking.params.update(
            {
                "metadata_path": metadata,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "std_rad": root_orientation_std_rad,
            }
        )

        rewards.reference_root_linear_velocity_tracking.params.update(
            {
                "metadata_path": metadata,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "std_mps": root_linear_velocity_std_mps,
            }
        )

        rewards.reference_root_angular_velocity_tracking.params.update(
            {
                "metadata_path": metadata,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "std_radps": root_angular_velocity_std_radps,
            }
        )

    # ---------------------------------------------------------
    # Reference-aligned reset
    # ---------------------------------------------------------
    reset_cfg = cfg.get(
        "aligned_reset",
        {},
    )

    if reset_cfg is None:
        reset_cfg = {}

    if not isinstance(
        reset_cfg,
        dict,
    ):
        raise ValueError(
            "reference_tracking.aligned_reset "
            "must be a mapping"
        )

    aligned_reset_enabled = bool(
        reset_cfg.get(
            "enabled",
            False,
        )
    )

    if aligned_reset_enabled:
        env_cfg.events.reset_base = None
        env_cfg.events.reset_robot_joints = None

        env_cfg.events.reference_state_reset = EventTerm(
            func=ReferenceStateReset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "metadata_path": metadata,
                "frame": start_frame,
            },
        )

    # ---------------------------------------------------------
    # Residual control
    # ---------------------------------------------------------
    residual_cfg = cfg.get(
        "residual_control",
        {},
    )

    if residual_cfg is None:
        residual_cfg = {}

    if not isinstance(
        residual_cfg,
        dict,
    ):
        raise ValueError(
            "reference_tracking.residual_control "
            "must be a mapping"
        )

    residual_enabled = bool(
        residual_cfg.get(
            "enabled",
            False,
        )
    )

    residual_scale_rad = float(
        residual_cfg.get(
            "scale_rad",
            0.15,
        )
    )

    raw_action_clip = float(
        residual_cfg.get(
            "raw_action_clip",
            1.0,
        )
    )

    if residual_enabled:
        env_cfg.actions.JointPositionAction = (
            ReferenceResidualJointPositionActionCfg(
                asset_name="robot",
                joint_names=list(
                    BODY_JOINTS
                ),
                preserve_order=True,
                scale=1.0,
                offset=0.0,
                use_default_offset=False,
                metadata_path=metadata,
                start_frame=start_frame,
                end_frame=end_frame,
                residual_scale_rad=residual_scale_rad,
                raw_action_clip=raw_action_clip,
            )
        )

    # ---------------------------------------------------------
    # Actor / critic reference conditioning
    # ---------------------------------------------------------
    conditioning_enabled = bool(
        cfg.get(
            "conditioning_enabled",
            False,
        )
    )

    if conditioning_enabled:
        terms = (
            _reference_observation_terms(
                metadata,
                start_frame,
                end_frame,
                root_conditioning_enabled,
            )
        )

        for group_name in (
            "policy",
            "critic",
        ):
            group = getattr(
                env_cfg.observations,
                group_name,
            )

            for name, term in (
                terms.items()
            ):
                setattr(
                    group,
                    name,
                    term,
                )

    conditioning_observations = 0

    if conditioning_enabled:
        # 29 q error + 29 dq error + phase.
        conditioning_observations = 59

        # Position + orientation + linear velocity
        # + angular velocity = 4 x 3.
        if root_conditioning_enabled:
            conditioning_observations += 12

    return {
        "enabled": True,
        "conditioning_enabled": conditioning_enabled,
        "metadata": metadata,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frequency_hz": frequency_hz,
        "reference_control_steps": reference_control_steps,
        "reference_duration_s": reference_duration_s,
        "terminate_at_reference_end": terminate_at_reference_end,
        "joint_position_std_rad": position_std,
        "joint_velocity_std_radps": velocity_std,
        "conditioning_observations": conditioning_observations,
        "root_tracking_enabled": root_tracking_enabled,
        "root_conditioning_enabled": root_conditioning_enabled,
        "aligned_reset_enabled": aligned_reset_enabled,
        "residual_control_enabled": residual_enabled,
        "residual_scale_rad": (
            residual_scale_rad
            if residual_enabled
            else None
        ),
        "raw_action_clip": (
            raw_action_clip
            if residual_enabled
            else None
        ),
    }
