"""Reference-aligned environment reset for motion imitation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import (
    ManagerTermBase,
    SceneEntityCfg,
)

from .contracts import BODY_JOINTS
from .reference_state import (
    load_reference_state,
)


if TYPE_CHECKING:
    from isaaclab.envs import (
        ManagerBasedRLEnv,
    )


def _joint_ids(robot) -> list[int]:
    result = robot.find_joints(
        list(BODY_JOINTS),
        preserve_order=True,
    )

    ids = (
        result[0]
        if isinstance(result, tuple)
        else result
    )

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    return list(ids)


class ReferenceStateReset(
    ManagerTermBase
):
    """Reset root and 29 body joints into one reference frame."""

    def __init__(
        self,
        cfg,
        env: "ManagerBasedRLEnv",
    ):
        super().__init__(
            cfg,
            env,
        )

        params = cfg.params

        state = load_reference_state(
            params["metadata_path"]
        )

        self._frame = int(
            params.get(
                "frame",
                0,
            )
        )

        if not (
            0
            <= self._frame
            < state.frame_count
        ):
            raise ValueError(
                "Reference reset frame "
                f"{self._frame} outside "
                f"[0, {state.frame_count - 1}]"
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

        self._q = torch.as_tensor(
            state.dof_pos,
            dtype=torch.float32,
            device=device,
        )

        self._dq = torch.as_tensor(
            state.dof_vel,
            dtype=torch.float32,
            device=device,
        )

        robot = env.scene["robot"]

        self._joint_ids = _joint_ids(
            robot
        )

        if len(self._joint_ids) != 29:
            raise RuntimeError(
                "Reference reset did not resolve "
                "all 29 body joints"
            )

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = (
            SceneEntityCfg("robot")
        ),
        metadata_path: str = "",
        frame: int = 0,
    ) -> None:
        del metadata_path, frame

        robot = env.scene[
            asset_cfg.name
        ]

        if env_ids is None:
            env_ids = torch.arange(
                env.num_envs,
                dtype=torch.long,
                device=env.device,
            )

        count = len(env_ids)

        root_state = (
            robot.data.default_root_state[
                env_ids
            ].clone()
        )

        # Convert default local root position into each
        # environment's world frame.
        root_state[:, 0:3] += (
            env.scene.env_origins[
                env_ids
            ]
        )

        # Keep the exact task start X/Y for every environment,
        # but use the reference pelvis height.
        root_state[:, 2] = (
            env.scene.env_origins[
                env_ids,
                2,
            ]
            + self._root_pos[
                self._frame,
                2,
            ]
        )

        root_state[:, 3:7] = (
            self._root_quat[
                self._frame
            ].unsqueeze(0).expand(
                count,
                -1,
            )
        )

        root_state[:, 7:10] = (
            self._root_lin_vel[
                self._frame
            ].unsqueeze(0).expand(
                count,
                -1,
            )
        )

        root_state[:, 10:13] = (
            self._root_ang_vel[
                self._frame
            ].unsqueeze(0).expand(
                count,
                -1,
            )
        )

        q = (
            self._q[
                self._frame
            ]
            .unsqueeze(0)
            .expand(
                count,
                -1,
            )
            .clone()
        )

        dq = (
            self._dq[
                self._frame
            ]
            .unsqueeze(0)
            .expand(
                count,
                -1,
            )
            .clone()
        )

        # Root-link state is intentional: the frozen
        # reference trajectory describes the pelvis/root link.
        robot.write_root_link_state_to_sim(
            root_state,
            env_ids=env_ids,
        )

        robot.write_joint_state_to_sim(
            q,
            dq,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )

        # Avoid a stale position target between episode reset
        # and the first action-manager update.
        robot.set_joint_position_target(
            q,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
