import torch

from isaac_lab.g1_stairs.reward_math import phase_gated_posture_scale


def test_low_command_off_platform_cannot_activate_finish_pressure():
    progress = torch.tensor([0.0])
    command = torch.tensor([0.0])

    scale, finish = phase_gated_posture_scale(
        progress,
        command,
        cruise_speed_mps=0.35,
    )

    assert torch.allclose(finish, torch.tensor([0.0]))
    assert torch.allclose(scale, torch.tensor([0.25]))


def test_platform_progress_strengthens_posture_without_finish_gate():
    progress = torch.tensor([1.0])
    command = torch.tensor([0.35])

    scale, finish = phase_gated_posture_scale(
        progress,
        command,
        cruise_speed_mps=0.35,
    )

    assert torch.allclose(finish, torch.tensor([0.0]))
    assert torch.allclose(scale, torch.tensor([1.0]))


def test_finish_pressure_requires_platform_progress():
    progress = torch.tensor([1.0])
    command = torch.tensor([0.0])

    scale, finish = phase_gated_posture_scale(
        progress,
        command,
        cruise_speed_mps=0.35,
    )

    assert torch.allclose(finish, torch.tensor([1.0]))
    assert torch.allclose(scale, torch.tensor([2.0]))


def test_phase_scale_is_explicitly_bounded():
    progress = torch.tensor([1.0])
    command = torch.tensor([0.0])

    scale, _ = phase_gated_posture_scale(
        progress,
        command,
        cruise_speed_mps=0.35,
        weak_scale=0.25,
        platform_gain=4.0,
        finish_gain=4.0,
        maximum_scale=2.0,
    )

    assert torch.allclose(scale, torch.tensor([2.0]))
