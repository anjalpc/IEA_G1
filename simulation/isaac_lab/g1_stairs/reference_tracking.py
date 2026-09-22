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


# =====================================================================
# Whole-body reference state
# =====================================================================

from isaaclab.utils import math as math_utils

from .reference_state import load_reference_state


class _ReferenceRootTerm(ManagerTermBase):
    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        params = cfg.params

        state = load_reference_state(
            params["metadata_path"]
        )

        self._start_frame = int(
            params.get("start_frame", 0)
        )

        self._end_frame = int(
            params.get(
                "end_frame",
                state.frame_count - 1,
            )
        )

        device = env.device

        self._root_pos = torch.as_tensor(
            state.root_pos_task,
            dtype=torch.float32,
            device=device,
        )

        self._root_quat = torch.as_tensor(
            state.root_quat_task,
            dtype=torch.float32,
            device=device,
        )

        self._root_lin_vel = torch.as_tensor(
            state.root_lin_vel_task,
            dtype=torch.float32,
            device=device,
        )

        self._root_ang_vel = torch.as_tensor(
            state.root_ang_vel_task,
            dtype=torch.float32,
            device=device,
        )

    def _frame(
        self,
        env: "ManagerBasedRLEnv",
    ) -> torch.Tensor:
        return torch.clamp(
            env.episode_length_buf
            + self._start_frame,
            min=self._start_frame,
            max=self._end_frame,
        ).long()

    def _target_position(
        self,
        env: "ManagerBasedRLEnv",
        robot,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        target = self._root_pos[
            frame
        ].clone()

        # Reference XY is applied relative to the configured
        # task start so every parallel environment receives
        # the same motion in its own world tile.
        reference_xy0 = self._root_pos[
            self._start_frame,
            :2,
        ]

        delta_xy = (
            target[:, :2]
            - reference_xy0.unsqueeze(0)
        )

        default_root = (
            robot.data.default_root_state[
                :, :3
            ]
            + env.scene.env_origins
        )

        target[:, :2] = (
            default_root[:, :2]
            + delta_xy
        )

        # Reference Z is absolute relative to local ground.
        target[:, 2] = (
            env.scene.env_origins[:, 2]
            + target[:, 2]
        )

        return target


class reference_root_position_error(
    _ReferenceRootTerm
):
    """3-D reference pelvis/root position error."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        robot = env.scene[
            asset_cfg.name
        ]

        frame = self._frame(env)

        target = self._target_position(
            env,
            robot,
            frame,
        )

        return (
            target
            - robot.data.root_link_pos_w
        )


class reference_root_orientation_error(
    _ReferenceRootTerm
):
    """3-D axis-angle orientation error."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        robot = env.scene[
            asset_cfg.name
        ]

        frame = self._frame(env)

        target = self._root_quat[
            frame
        ]

        return math_utils.quat_box_minus(
            target,
            robot.data.root_link_quat_w,
        )


class reference_root_linear_velocity_error(
    _ReferenceRootTerm
):
    """3-D world-frame pelvis/root linear-velocity error."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        robot = env.scene[
            asset_cfg.name
        ]

        frame = self._frame(env)

        return (
            self._root_lin_vel[
                frame
            ]
            - robot.data.root_link_lin_vel_w
        )


class reference_root_angular_velocity_error(
    _ReferenceRootTerm
):
    """3-D world-frame pelvis/root angular-velocity error."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
    ) -> torch.Tensor:
        del metadata_path, start_frame, end_frame

        robot = env.scene[
            asset_cfg.name
        ]

        frame = self._frame(env)

        return (
            self._root_ang_vel[
                frame
            ]
            - robot.data.root_link_ang_vel_w
        )


class reference_root_position_tracking(
    reference_root_position_error
):
    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
        std_m: float = 0.15,
    ) -> torch.Tensor:
        error = super().__call__(
            env,
            asset_cfg,
            metadata_path,
            start_frame,
            end_frame,
        )

        squared = torch.sum(
            torch.square(error),
            dim=1,
        )

        return torch.exp(
            -squared
            / max(
                float(std_m) ** 2,
                1.0e-8,
            )
        )


class reference_root_orientation_tracking(
    reference_root_orientation_error
):
    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
        std_rad: float = 0.35,
    ) -> torch.Tensor:
        error = super().__call__(
            env,
            asset_cfg,
            metadata_path,
            start_frame,
            end_frame,
        )

        squared = torch.sum(
            torch.square(error),
            dim=1,
        )

        return torch.exp(
            -squared
            / max(
                float(std_rad) ** 2,
                1.0e-8,
            )
        )


class reference_root_linear_velocity_tracking(
    reference_root_linear_velocity_error
):
    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
        std_mps: float = 0.50,
    ) -> torch.Tensor:
        error = super().__call__(
            env,
            asset_cfg,
            metadata_path,
            start_frame,
            end_frame,
        )

        mse = torch.mean(
            torch.square(error),
            dim=1,
        )

        return torch.exp(
            -mse
            / max(
                float(std_mps) ** 2,
                1.0e-8,
            )
        )


class reference_root_angular_velocity_tracking(
    reference_root_angular_velocity_error
):
    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        metadata_path: str = "",
        start_frame: int = 0,
        end_frame: int = 246,
        std_radps: float = 1.0,
    ) -> torch.Tensor:
        error = super().__call__(
            env,
            asset_cfg,
            metadata_path,
            start_frame,
            end_frame,
        )

        mse = torch.mean(
            torch.square(error),
            dim=1,
        )

        return torch.exp(
            -mse
            / max(
                float(std_radps) ** 2,
                1.0e-8,
            )
        )
