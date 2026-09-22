"""Residual joint-position control around the validated stair reference."""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.joint_actions import (
    JointPositionAction,
)
from isaaclab.utils import configclass

from .reference_motion import load_reference_motion


def _resolve_reference_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / path


class ReferenceResidualJointPositionAction(
    JointPositionAction
):
    """Apply policy actions as bounded residuals around q_ref(t)."""

    cfg: "ReferenceResidualJointPositionActionCfg"

    def __init__(
        self,
        cfg: "ReferenceResidualJointPositionActionCfg",
        env,
    ) -> None:
        super().__init__(cfg, env)

        self._env = env

        motion = load_reference_motion(
            _resolve_reference_path(cfg.metadata_path)
        )

        self._start_frame = int(cfg.start_frame)
        self._end_frame = int(cfg.end_frame)

        if self._start_frame < 0:
            raise ValueError(
                "reference residual start_frame must be >= 0"
            )

        if self._end_frame >= motion.frame_count:
            raise ValueError(
                f"reference residual end_frame "
                f"{self._end_frame} exceeds "
                f"{motion.frame_count - 1}"
            )

        if self._end_frame < self._start_frame:
            raise ValueError(
                "reference residual end_frame must be "
                ">= start_frame"
            )

        if motion.dof_pos.shape[1] != self.action_dim:
            raise RuntimeError(
                "reference action dimension mismatch: "
                f"reference={motion.dof_pos.shape[1]}, "
                f"action={self.action_dim}"
            )

        self._reference = torch.as_tensor(
            motion.dof_pos,
            dtype=torch.float32,
            device=self.device,
        )

        self._residual_scale = float(
            cfg.residual_scale_rad
        )

        self._raw_clip = float(
            cfg.raw_action_clip
        )

        if self._residual_scale <= 0.0:
            raise ValueError(
                "residual_scale_rad must be > 0"
            )

        if self._raw_clip <= 0.0:
            raise ValueError(
                "raw_action_clip must be > 0"
            )

        print(
            "[INFO] Reference residual action:",
            {
                "start_frame": self._start_frame,
                "end_frame": self._end_frame,
                "residual_scale_rad": self._residual_scale,
                "raw_action_clip": self._raw_clip,
                "action_dim": self.action_dim,
            },
            flush=True,
        )

    def process_actions(
        self,
        actions: torch.Tensor,
    ) -> None:
        """Convert policy output into q_ref + bounded residual."""

        self._raw_actions[:] = actions

        frame = torch.clamp(
            self._env.episode_length_buf
            + self._start_frame,
            min=self._start_frame,
            max=self._end_frame,
        ).long()

        q_ref = self._reference[frame]

        bounded_action = torch.clamp(
            self._raw_actions,
            min=-self._raw_clip,
            max=self._raw_clip,
        )

        residual = (
            bounded_action
            * self._residual_scale
        )

        target = q_ref + residual

        # Keep all commanded targets inside the robot's soft
        # position limits even when the policy saturates.
        limits = (
            self._asset.data.soft_joint_pos_limits[
                :, self._joint_ids, :
            ]
        )

        target = torch.clamp(
            target,
            min=limits[..., 0],
            max=limits[..., 1],
        )

        self._processed_actions[:] = target


@configclass
class ReferenceResidualJointPositionActionCfg(
    JointPositionActionCfg
):
    """Configuration for reference-residual joint control."""

    class_type: type = (
        ReferenceResidualJointPositionAction
    )

    metadata_path: str = (
        "experiments/g1_stairs/reference_data/"
        "stair_ascent_reference.yaml"
    )

    start_frame: int = 0
    end_frame: int = 260

    residual_scale_rad: float = 0.15
    raw_action_clip: float = 1.0
