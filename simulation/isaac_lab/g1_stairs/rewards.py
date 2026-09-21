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


def stair_swing_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    riser_edges_m: tuple[float, ...] = (1.0, 1.25, 1.50),
    tread_heights_m: tuple[float, ...] = (0.20, 0.40, 0.60),
    activation_distance_m: float = 0.20,
    lip_distance_m: float = 0.05,
    clearance_margin_m: float = 0.05,
    std: float = 0.01,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    """Reward a moving swing foot for clearing the upcoming stair riser.

    Unlike Unitree's flat-ground foot-clearance reward, this term uses
    robot-relative stair geometry. Each riser gets its own required world-Z
    target equal to the next tread height plus a small clearance margin.

    Reward is active only shortly before and over each riser lip. Overshooting
    the requested clearance is not penalized.
    """
    if len(riser_edges_m) != len(tread_heights_m):
        raise ValueError(
            "riser_edges_m and tread_heights_m must have equal length"
        )

    robot = env.scene[asset_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    foot_x = (
        foot_pos_w[:, :, 0]
        - env.scene.env_origins[:, 0].unsqueeze(1)
    )
    foot_z = (
        foot_pos_w[:, :, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )

    horizontal_speed = torch.linalg.norm(
        foot_vel_w[:, :, :2],
        dim=2,
    )
    swing_gate = torch.tanh(tanh_mult * horizontal_speed)

    reward = torch.zeros(
        env.num_envs,
        dtype=foot_z.dtype,
        device=env.device,
    )

    for edge_m, tread_height_m in zip(
        riser_edges_m,
        tread_heights_m,
        strict=True,
    ):
        active = (
            (foot_x >= edge_m - activation_distance_m)
            & (foot_x <= edge_m + lip_distance_m)
        )

        target_z = tread_height_m + clearance_margin_m
        shortfall = torch.clamp(target_z - foot_z, min=0.0)

        clearance_reward = torch.exp(
            -torch.square(shortfall) / std
        )

        reward += torch.sum(
            clearance_reward
            * swing_gate
            * active.to(foot_z.dtype),
            dim=1,
        )

    return reward
