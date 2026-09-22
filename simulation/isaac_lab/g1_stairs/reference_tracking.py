"""Closed-loop tracking rewards for the validated stair reference motion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .contracts import BODY_JOINTS
from .reference_motion import load_reference_motion

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / path


def _joint_ids(robot) -> list[int]:
    result = robot.find_joints(
        list(BODY_JOINTS),
        preserve_order=True,
    )

    ids = result[0] if isinstance(result, tuple) else result

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    return list(ids)


class reference_joint_position_tracking(ManagerTermBase):
    """Track the validated 29-DoF reference joint trajectory."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        params = cfg.params
        metadata_path = _resolve_path(params["metadata_path"])

        motion = load_reference_motion(metadata_path)

        self._start_frame = int(params.get("start_frame", 0))
        self._end_frame = int(
            params.get("end_frame", motion.frame_count - 1)
        )

        if self._start_frame < 0:
            raise ValueError("reference start_frame must be >= 0")

        if self._end_frame >= motion.frame_count:
            raise ValueError(
                f"reference end_frame {self._end_frame} exceeds "
                f"{motion.frame_count - 1}"
            )

        self._reference = torch.as_tensor(
            motion.dof_pos,
            dtype=torch.float32,
            device=env.device,
        )

        robot = env.scene["robot"]
        self._joint_ids = _joint_ids(robot)

        if len(self._joint_ids) != len(BODY_JOINTS):
            raise RuntimeError(
                "reference tracking did not resolve all 29 body joints"
            )

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 260,
        std_rad: float = 0.35,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        robot = env.scene[asset_cfg.name]

        frame = torch.clamp(
            env.episode_length_buf + self._start_frame,
            min=self._start_frame,
            max=self._end_frame,
        ).long()

        target = self._reference[frame]

        actual = robot.data.joint_pos[
            :, self._joint_ids
        ]

        error = actual - target

        mse = torch.mean(
            torch.square(error),
            dim=1,
        )

        variance = max(float(std_rad) ** 2, 1.0e-8)

        return torch.exp(-mse / variance)


class reference_joint_velocity_tracking(ManagerTermBase):
    """Track the validated 29-DoF reference joint velocities."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        params = cfg.params
        metadata_path = _resolve_path(params["metadata_path"])

        motion = load_reference_motion(metadata_path)

        self._start_frame = int(params.get("start_frame", 0))
        self._end_frame = int(
            params.get("end_frame", motion.frame_count - 1)
        )

        self._reference = torch.as_tensor(
            motion.dof_vel,
            dtype=torch.float32,
            device=env.device,
        )

        robot = env.scene["robot"]
        self._joint_ids = _joint_ids(robot)

        if len(self._joint_ids) != len(BODY_JOINTS):
            raise RuntimeError(
                "reference tracking did not resolve all 29 body joints"
            )

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 260,
        std_radps: float = 2.0,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        robot = env.scene[asset_cfg.name]

        frame = torch.clamp(
            env.episode_length_buf + self._start_frame,
            min=self._start_frame,
            max=self._end_frame,
        ).long()

        target = self._reference[frame]

        actual = robot.data.joint_vel[
            :, self._joint_ids
        ]

        error = actual - target

        mse = torch.mean(
            torch.square(error),
            dim=1,
        )

        variance = max(float(std_radps) ** 2, 1.0e-8)

        return torch.exp(-mse / variance)


class reference_joint_position_error(ManagerTermBase):
    """29-D joint-position error q_ref - q_actual."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        params = cfg.params
        motion = load_reference_motion(
            _resolve_path(params["metadata_path"])
        )

        self._start_frame = int(params.get("start_frame", 0))
        self._end_frame = int(
            params.get("end_frame", motion.frame_count - 1)
        )

        self._reference = torch.as_tensor(
            motion.dof_pos,
            dtype=torch.float32,
            device=env.device,
        )

        robot = env.scene["robot"]
        self._joint_ids = _joint_ids(robot)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 260,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        frame = torch.clamp(
            env.episode_length_buf + self._start_frame,
            min=self._start_frame,
            max=self._end_frame,
        ).long()

        robot = env.scene[asset_cfg.name]

        q_ref = self._reference[frame]
        q_actual = robot.data.joint_pos[:, self._joint_ids]

        return q_ref - q_actual


class reference_joint_velocity_error(ManagerTermBase):
    """29-D joint-velocity error dq_ref - dq_actual."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        params = cfg.params
        motion = load_reference_motion(
            _resolve_path(params["metadata_path"])
        )

        self._start_frame = int(params.get("start_frame", 0))
        self._end_frame = int(
            params.get("end_frame", motion.frame_count - 1)
        )

        self._reference = torch.as_tensor(
            motion.dof_vel,
            dtype=torch.float32,
            device=env.device,
        )

        robot = env.scene["robot"]
        self._joint_ids = _joint_ids(robot)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 260,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        frame = torch.clamp(
            env.episode_length_buf + self._start_frame,
            min=self._start_frame,
            max=self._end_frame,
        ).long()

        robot = env.scene[asset_cfg.name]

        dq_ref = self._reference[frame]
        dq_actual = robot.data.joint_vel[:, self._joint_ids]

        return dq_ref - dq_actual


def reference_phase(
    env: "ManagerBasedRLEnv",
    start_frame: int = 0,
    end_frame: int = 260,
) -> torch.Tensor:
    """Normalized reference phase in [0, 1]."""

    length = max(end_frame - start_frame, 1)

    frame = torch.clamp(
        env.episode_length_buf + start_frame,
        min=start_frame,
        max=end_frame,
    ).to(torch.float32)

    phase = (frame - float(start_frame)) / float(length)

    return phase.unsqueeze(-1)
