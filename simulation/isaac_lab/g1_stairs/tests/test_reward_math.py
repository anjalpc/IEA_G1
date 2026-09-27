from __future__ import annotations

import pytest
import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    body_advance_gate,
    bottleneck_score,
    continuous_touchdown_potential,
    coupled_swing_score,
    deadzone_score,
    support_anchor_slip_penalty,
    support_quality,
    support_relative_forward_progress,
    supported_single_foot_lift_score,
    swing_phase_potential,
    swing_arc_waypoints,
    target_band_score,
    tread_containment_score,
)


def test_bottleneck_score_cannot_average_away_a_broken_prerequisite() -> None:
    score = bottleneck_score(
        torch.tensor([1.0, 0.9]),
        torch.tensor([0.0, 0.8]),
        torch.tensor([1.0, 0.7]),
    )
    assert score.tolist() == pytest.approx([0.0, 0.7])


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


def test_swing_arc_waypoints_couple_forward_and_vertical_motion() -> None:
    waypoints = swing_arc_waypoints(
        start_x_m=torch.tensor([0.15]),
        riser_x_m=0.425,
        tread_depth_m=0.25,
        riser_height_m=torch.tensor([0.06]),
        early_x_fraction=0.35,
        clearance_x_fraction=0.60,
        early_lift_riser_fraction=0.55,
        clearance_margin_m=0.025,
        hover_margin_m=0.025,
    )

    assert waypoints["landing_x"].item() == pytest.approx(0.55)
    assert waypoints["early_x"].item() == pytest.approx(0.29)
    assert waypoints["clearance_x"].item() == pytest.approx(0.39)
    assert waypoints["hover_x"].item() == pytest.approx(0.55)
    assert waypoints["early_z"].item() == pytest.approx(0.033)
    assert waypoints["clearance_z"].item() == pytest.approx(0.085)
    assert waypoints["hover_z"].item() == pytest.approx(0.085)
    assert waypoints["touchdown_z"].item() == pytest.approx(0.06)


def test_swing_arc_waypoints_reject_invalid_phase_order() -> None:
    with pytest.raises(ValueError, match="0 < early < clearance < 1"):
        swing_arc_waypoints(
            start_x_m=torch.tensor([0.15]),
            riser_x_m=0.425,
            tread_depth_m=0.25,
            riser_height_m=torch.tensor([0.06]),
            early_x_fraction=0.70,
            clearance_x_fraction=0.60,
            early_lift_riser_fraction=0.55,
            clearance_margin_m=0.025,
            hover_margin_m=0.025,
        )


def test_support_relative_forward_progress_requires_new_body_advance() -> None:
    support_x = torch.tensor([-0.02])
    activation_root_x = torch.tensor([0.00])

    start = support_relative_forward_progress(
        root_x_m=torch.tensor([0.00]),
        support_anchor_x_m=support_x,
        activation_root_x_m=activation_root_x,
        target_forward_from_support_m=0.08,
    )
    halfway = support_relative_forward_progress(
        root_x_m=torch.tensor([0.03]),
        support_anchor_x_m=support_x,
        activation_root_x_m=activation_root_x,
        target_forward_from_support_m=0.08,
    )
    reached = support_relative_forward_progress(
        root_x_m=torch.tensor([0.06]),
        support_anchor_x_m=support_x,
        activation_root_x_m=activation_root_x,
        target_forward_from_support_m=0.08,
    )
    retreat = support_relative_forward_progress(
        root_x_m=torch.tensor([-0.02]),
        support_anchor_x_m=support_x,
        activation_root_x_m=activation_root_x,
        target_forward_from_support_m=0.08,
    )

    assert start.item() == pytest.approx(0.0)
    assert halfway.item() == pytest.approx(0.5)
    assert reached.item() == pytest.approx(1.0)
    assert retreat.item() == pytest.approx(0.0)


def test_swing_phase_potential_preserves_historical_touchdown_credit_by_default() -> None:
    historical = swing_phase_potential(
        phase=torch.tensor([4]),
        active_score=torch.tensor([0.0]),
        phase_count=5,
        reversible_touchdown_credit=False,
    )
    assert historical.item() == pytest.approx(0.8)


def test_swing_phase_potential_v07_does_not_bank_touchdown_entry() -> None:
    touchdown = swing_phase_potential(
        phase=torch.tensor([4, 4]),
        active_score=torch.tensor([0.0, 1.0]),
        phase_count=5,
        reversible_touchdown_credit=True,
    )
    complete = swing_phase_potential(
        phase=torch.tensor([5]),
        active_score=torch.tensor([0.0]),
        phase_count=5,
        reversible_touchdown_credit=True,
    )

    assert touchdown.tolist() == pytest.approx([0.6, 0.8])
    assert complete.item() == pytest.approx(1.0)


def test_body_advance_gate_preserves_v06_phase4_latch() -> None:
    gate = body_advance_gate(
        phase=torch.tensor([4]),
        live_clearance_gate=torch.tensor([False]),
        touchdown_placement_gate=torch.tensor([False]),
        upper_support_gate=torch.tensor([False]),
        require_live_touchdown_commitment=False,
    )
    assert gate.item() is True


def test_body_advance_gate_v07_requires_current_phase4_commitment() -> None:
    phases = torch.tensor([3, 4, 4, 4])
    gate = body_advance_gate(
        phase=phases,
        live_clearance_gate=torch.tensor([True, False, False, False]),
        touchdown_placement_gate=torch.tensor([False, False, True, False]),
        upper_support_gate=torch.tensor([False, False, False, True]),
        require_live_touchdown_commitment=True,
    )
    assert gate.tolist() == [True, False, True, True]


def test_support_anchor_slip_penalty_detects_xy_drift_or_planar_slip() -> None:
    penalty = support_anchor_slip_penalty(
        anchor_error_m=torch.tensor([0.010, 0.060, 0.010, 0.060]),
        planar_speed_mps=torch.tensor([0.010, 0.010, 0.20, 0.20]),
        active_gate=torch.tensor([True, True, True, False]),
        anchor_free_m=0.015,
        anchor_zero_m=0.060,
        speed_free_mps=0.020,
        speed_zero_mps=0.20,
    )
    assert penalty.tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0])


def test_continuous_touchdown_potential_preserves_entry_then_rewards_support() -> None:
    potential = continuous_touchdown_potential(
        entry_potential=torch.tensor([0.78, 0.78, 0.78]),
        commitment_score=torch.tensor([0.75, 0.75, 0.25]),
        upper_support_score=torch.tensor([0.0, 0.5, 0.5]),
        commitment_full_at=0.75,
        commitment_zero_at=0.25,
    )
    assert potential[0].item() == pytest.approx(0.78)
    assert potential[1].item() == pytest.approx(0.89)
    assert potential[2].item() == pytest.approx(0.0)
