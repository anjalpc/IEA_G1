from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from simulation.isaac_lab.g1_stairs.geometry_locomotion_rewards import (
    track_ang_vel_z_zero_centered,
)
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
)


def test_zero_centered_yaw_is_exact_unitree_score_minus_one() -> None:
    command = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    root_ang_vel_b = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
        dtype=torch.float32,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_ang_vel_b=root_ang_vel_b)
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(
            get_command=lambda _name: command,
        ),
    )

    score = track_ang_vel_z_zero_centered(
        env,
        std=0.5,
        command_name="base_velocity",
    )
    error = torch.square(command[:, 2] - root_ang_vel_b[:, 2])
    expected = torch.exp(-error / (0.5**2)) - 1.0

    assert torch.allclose(score, expected)
    assert score[0].item() == 0.0
    assert score[1].item() < score[0].item()


def test_c7_yaw_zero_centered_leaf_changes_only_yaw_reward_identity() -> None:
    root = Path(__file__).resolve().parents[4]
    resolved, _ = resolve_config(
        root
        / "experiments/g1_stairs/configs/"
        "c7_liftoff_commit_yaw_zero_centered_diag100.yaml"
    )
