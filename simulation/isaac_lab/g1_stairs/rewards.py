"""Minimal Phase 12 rewards for the fixed nominal staircase."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_progress(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return forward world-X velocity; RewardManager integrates it over dt."""
    robot = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_w[:, 0]


class stair_progress(ManagerTermBase):
    """Reward each newly reached stair level exactly once per episode."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._highest_level = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._highest_level[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        step_edges_m: tuple[float, ...] = (1.0, 1.25, 1.50),
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        progress_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        edges = torch.as_tensor(
            step_edges_m,
            dtype=progress_x.dtype,
            device=progress_x.device,
        )
        level = torch.sum(progress_x.unsqueeze(-1) >= edges, dim=-1)
        newly_reached = torch.clamp(level - self._highest_level, min=0)
        self._highest_level = torch.maximum(self._highest_level, level)

        # RewardManager multiplies every term by step_dt. Divide discrete
        # transitions by step_dt so one newly reached level contributes 1.0.
        return newly_reached.to(progress_x.dtype) / env.step_dt


class progress_threshold_event(ManagerTermBase):
    """Reward crossing a forward-progress threshold once per episode."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._triggered = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._triggered[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold_m: float = 1.75,
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        progress_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]

        reached = progress_x >= threshold_m
        newly_reached = reached & ~self._triggered
        self._triggered |= reached

        return newly_reached.to(progress_x.dtype) / env.step_dt
