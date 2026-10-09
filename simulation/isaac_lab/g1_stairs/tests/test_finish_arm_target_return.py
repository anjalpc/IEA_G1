from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from simulation.isaac_lab.g1_stairs.geometry_locomotion_rewards import (
    finish_arm_target_return,
    geometry_failure_termination_event,
)


class _CommandManager:
    def __init__(self, forward_command: torch.Tensor):
        self._command = torch.stack(
            (
                forward_command,
                torch.zeros_like(forward_command),
                torch.zeros_like(forward_command),
            ),
            dim=1,
        )

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "base_velocity"
        return self._command


def test_finish_arm_target_return_is_zero_before_stop_and_ramps_to_full() -> None:
    target = torch.tensor(
        [
            [1.0, 0.0, 9.0],
            [1.0, 0.0, 9.0],
            [1.0, 0.0, 9.0],
        ]
    )
    default = torch.zeros_like(target)

    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos_target=target,
            default_joint_pos=default,
        )
    )

    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=_CommandManager(
            torch.tensor([0.35, 0.175, 0.0])
        ),
    )

    asset_cfg = SimpleNamespace(
        name="robot",
        joint_ids=[0, 1],
    )

    cost = finish_arm_target_return(
        env,
        cruise_speed_mps=0.35,
        command_name="base_velocity",
        asset_cfg=asset_cfg,
    )

    assert cost.tolist() == pytest.approx([0.0, 0.25, 0.5])


def test_geometry_failure_termination_event_penalizes_failure_not_success() -> None:
    from simulation.isaac_lab.g1_stairs.geometry_episode import Result

    tracker = SimpleNamespace(
        reason=torch.tensor([Result.FALL, Result.SUCCESS, Result.RUNNING])
    )
    env = SimpleNamespace(
        reset_terminated=torch.tensor([True, True, False]),
        step_dt=0.02,
        _g1_geometry_episode=SimpleNamespace(tracker=tracker),
    )

    event = geometry_failure_termination_event(env)
    assert event.tolist() == pytest.approx([50.0, 0.0, 0.0])
