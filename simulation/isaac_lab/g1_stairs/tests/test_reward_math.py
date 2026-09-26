from __future__ import annotations

import pytest
import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    coupled_swing_score,
    deadzone_score,
    support_quality,
    supported_single_foot_lift_score,
    target_band_score,
    tread_containment_score,
)


def test_coupled_swing_requires_lift_and_advance() -> None:
    riser_height = torch.tensor([[0.06]])

    no_lift = coupled_swing_score(
        sole_bottom_m=torch.tensor([[0.0]]),
        front_x_m=torch.tensor([[0.44]]),
        riser_x_m=0.425,
        riser_height_m=riser_height,
        activation_distance_m=0.30,
        insertion_target_m=0.10,
        clearance_margin_m=0.015,
    )
    no_advance = coupled_swing_score(
        sole_bottom_m=torch.tensor([[0.075]]),
        front_x_m=torch.tensor([[0.125]]),
        riser_x_m=0.425,
        riser_height_m=riser_height,
        activation_distance_m=0.30,
        insertion_target_m=0.10,
        clearance_margin_m=0.015,
    )
    valid = coupled_swing_score(
        sole_bottom_m=torch.tensor([[0.075]]),
        front_x_m=torch.tensor([[0.525]]),
        riser_x_m=0.425,
        riser_height_m=riser_height,
        activation_distance_m=0.30,
        insertion_target_m=0.10,
        clearance_margin_m=0.015,
    )

    assert no_lift.item() == 0.0
    assert no_advance.item() == 0.0
    assert valid.item() == pytest.approx(1.0)



def test_supported_single_foot_lift_requires_opposite_lower_support() -> None:
    common = dict(
        bodyweight_n=torch.tensor([400.0]),
        riser_height_m=torch.tensor([0.06]),
        lower_support_height_tolerance_m=0.02,
        lower_support_force_bodyweight_fraction=0.30,
        maximum_lower_support_foot_speed_mps=0.20,
    )

    supported = supported_single_foot_lift_score(
        sole_bottom_m=torch.tensor([[0.06, 0.0]]),
        vertical_force_n=torch.tensor([[0.0, 200.0]]),
        foot_speed_mps=torch.tensor([[0.10, 0.0]]),
        **common,
    )
    unloaded = supported_single_foot_lift_score(
        sole_bottom_m=torch.tensor([[0.06, 0.0]]),
        vertical_force_n=torch.tensor([[0.0, 0.0]]),
        foot_speed_mps=torch.tensor([[0.10, 0.0]]),
        **common,
    )
    two_foot_jump = supported_single_foot_lift_score(
        sole_bottom_m=torch.tensor([[0.06, 0.06]]),
        vertical_force_n=torch.tensor([[0.0, 0.0]]),
        foot_speed_mps=torch.tensor([[0.10, 0.10]]),
        **common,
    )
    half_lift = supported_single_foot_lift_score(
        sole_bottom_m=torch.tensor([[0.03, 0.0]]),
        vertical_force_n=torch.tensor([[0.0, 200.0]]),
        foot_speed_mps=torch.tensor([[0.10, 0.0]]),
        **common,
    )

    assert supported.item() == pytest.approx(1.0)
    assert unloaded.item() == 0.0
    assert two_foot_jump.item() == 0.0
    assert half_lift.item() == pytest.approx(0.5)

def test_tread_containment_requires_complete_foot_inside_margin() -> None:
    contained = tread_containment_score(
        torch.tensor([[0.45]]),
        torch.tensor([[0.60]]),
        riser_x_m=0.425,
        tread_depth_m=0.25,
        margin_m=0.015,
    )
    overhang = tread_containment_score(
        torch.tensor([[0.45]]),
        torch.tensor([[0.68]]),
        riser_x_m=0.425,
        tread_depth_m=0.25,
        margin_m=0.015,
    )

    assert contained.item() == 1.0
    assert overhang.item() == 0.0


def test_support_quality_rejects_sliding_or_unloaded_contact() -> None:
    common = dict(
        containment=torch.ones((1, 1)),
        height_score=torch.ones((1, 1)),
        full_support_force_n=80.0,
        maximum_supported_speed_mps=0.20,
    )
    stable = support_quality(
        vertical_force_n=torch.tensor([[80.0]]),
        speed_mps=torch.tensor([[0.0]]),
        **common,
    )
    unloaded = support_quality(
        vertical_force_n=torch.tensor([[0.0]]),
        speed_mps=torch.tensor([[0.0]]),
        **common,
    )
    sliding = support_quality(
        vertical_force_n=torch.tensor([[80.0]]),
        speed_mps=torch.tensor([[0.25]]),
        **common,
    )

    assert stable.item() == 1.0
    assert unloaded.item() == 0.0
    assert sliding.item() == 0.0


def test_support_quality_accepts_per_environment_force_scale() -> None:
    score = support_quality(
        containment=torch.ones((2, 1)),
        height_score=torch.ones((2, 1)),
        vertical_force_n=torch.tensor([[45.0], [60.0]]),
        speed_mps=torch.zeros((2, 1)),
        full_support_force_n=torch.tensor([[45.0], [60.0]]),
        maximum_supported_speed_mps=0.15,
    )

    assert torch.allclose(score, torch.ones_like(score))


def test_deadzone_score_preserves_corridor_and_rejects_large_error() -> None:
    score = deadzone_score(
        torch.tensor([0.01, 0.03, 0.055, 0.08]),
        free_tolerance=0.03,
        zero_at=0.08,
    )
    assert score[0].item() == pytest.approx(1.0)
    assert score[1].item() == pytest.approx(1.0)
    assert 0.0 < score[2].item() < 1.0
    assert score[3].item() == pytest.approx(0.0, abs=1e-6)


def test_target_band_score_rejects_underheight_and_overheight() -> None:
    values = torch.tensor([0.00, 0.07, 0.10, 0.15, 0.20])
    score = target_band_score(
        values,
        lower=0.07,
        upper=0.12,
        falloff=0.04,
    )
    assert score[0].item() == pytest.approx(0.0)
    assert score[1].item() == pytest.approx(1.0)
    assert score[2].item() == pytest.approx(1.0)
    assert 0.0 < score[3].item() < 1.0
    assert score[4].item() == pytest.approx(0.0)
