from __future__ import annotations

import pytest
import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    body_advance_gate,
    bottleneck_score,
    continuous_touchdown_potential,
    exp_tracking_score,
    finish_transition_gate,
    normalized_planar_velocity_error,
    normalized_excess_forward_velocity,
    coupled_swing_score,
    rear_edge_insertion_fraction,
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



def test_rear_edge_insertion_fraction_preserves_legacy_and_adds_pre_riser_progress() -> None:
    rear_x = torch.tensor([0.8200, 0.9375, 1.0550])
    riser_x = torch.tensor([1.0000, 1.0000, 1.0000])
    safe_start = torch.tensor([1.0150, 1.0150, 1.0150])
    insertion_target = torch.tensor([1.0550, 1.0550, 1.0550])

    legacy = rear_edge_insertion_fraction(
        rear_x_m=rear_x,
        riser_x_m=riser_x,
        safe_start_x_m=safe_start,
        insertion_target_x_m=insertion_target,
        start_before_riser_m=0.0,
    )
    shaped = rear_edge_insertion_fraction(
        rear_x_m=rear_x,
        riser_x_m=riser_x,
        safe_start_x_m=safe_start,
        insertion_target_x_m=insertion_target,
        start_before_riser_m=0.18,
    )

    assert legacy.tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert shaped.tolist() == pytest.approx([0.0, 0.5, 1.0])

    with pytest.raises(
        ValueError,
        match="start_before_riser_m cannot be negative",
    ):
        rear_edge_insertion_fraction(
            rear_x_m=rear_x,
            riser_x_m=riser_x,
            safe_start_x_m=safe_start,
            insertion_target_x_m=insertion_target,
            start_before_riser_m=-0.01,
        )


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


def test_support_anchor_slip_penalty_split_gate_keeps_anchor_active_after_recontact() -> None:
    penalty = support_anchor_slip_penalty(
        anchor_error_m=torch.tensor([0.060, 0.010]),
        planar_speed_mps=torch.tensor([0.010, 0.20]),
        active_gate=torch.tensor([False, False]),
        anchor_gate=torch.tensor([True, True]),
        speed_gate=torch.tensor([False, True]),
        anchor_free_m=0.015,
        anchor_zero_m=0.060,
        speed_free_mps=0.020,
        speed_zero_mps=0.20,
    )
    assert penalty.tolist() == pytest.approx([1.0, 1.0])


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


def test_stair_axis_tracking_prefers_forward_without_lateral_speed() -> None:
    target = 0.35
    std = 0.25
    good_error = torch.tensor([0.0])
    sideways_error = torch.tensor([
        target * target + 0.35 * 0.35
    ])
    good = exp_tracking_score(
        good_error,
        std * std,
        zero_centered=False,
    )
    sideways = exp_tracking_score(
        sideways_error,
        std * std,
        zero_centered=False,
    )
    assert good.item() > sideways.item()


def test_finish_transition_gate_activates_only_as_command_slows() -> None:
    gate = finish_transition_gate(
        torch.tensor([0.35, 0.126, 0.08, 0.0]),
        cruise_speed_mps=0.35,
    )
    assert gate.tolist() == pytest.approx([0.0, 0.64, 0.7714286, 1.0])


def test_finish_planar_error_exposes_model60_zero_command_momentum() -> None:
    severity = normalized_planar_velocity_error(
        torch.tensor([[0.284, -0.143], [0.0, 0.0]]),
        torch.tensor([0.0, 0.0]),
        scale_mps=0.25,
    )
    assert severity[0].item() == pytest.approx(1.61768, rel=1e-4)
    assert severity[1].item() == pytest.approx(0.0)


def test_normalized_excess_forward_velocity_matches_knee_audit_envelope() -> None:
    severity = normalized_excess_forward_velocity(
        torch.tensor([0.098, 0.339, 0.573, 0.920, -0.4]),
        free_velocity_mps=0.20,
        full_penalty_velocity_mps=0.70,
    )
    assert severity.tolist() == pytest.approx(
        [0.0, 0.278, 0.746, 1.0, 0.0],
        abs=1e-3,
    )


def test_normalized_excess_forward_velocity_validates_envelope() -> None:
    with pytest.raises(ValueError):
        normalized_excess_forward_velocity(
            torch.tensor([0.3]),
            free_velocity_mps=0.4,
            full_penalty_velocity_mps=0.4,
        )


def test_phase_arm_tolerance_is_loose_on_ascent_and_tightens_during_finish():
    from simulation.isaac_lab.g1_stairs.reward_math import phase_arm_tolerance

    tolerance = phase_arm_tolerance(
        entered_stairs=torch.tensor([False, True, True, True]),
        safe_platform=torch.tensor([False, False, True, True]),
        command_x_mps=torch.tensor([0.35, 0.35, 0.175, 0.0]),
        approach_tolerance_rad=0.20,
        ascent_tolerance_rad=0.35,
        finish_tolerance_rad=0.20,
        cruise_speed_mps=0.35,
    )

    assert tolerance.tolist() == pytest.approx([
        0.20,   # approach
        0.35,   # ascent
        0.275,  # halfway through finish ramp
        0.20,   # fully stopped
    ])


def test_arm_family_cost_is_independent_of_joint_count_and_tolerance_scale():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        weighted_arm_family_excess_cost,
    )

    shoulder = torch.zeros((2, 2))
    wrist = torch.zeros((2, 6))

    # Same relative violation in different families:
    # shoulder: error 0.40, tolerance 0.20 -> normalized excess 1.0
    # wrist:    error 0.20, tolerance 0.10 -> normalized excess 1.0
    shoulder[0, :] = 0.40
    wrist[1, :] = 0.20

    shoulder_tolerance = torch.full((2,), 0.20)
    wrist_tolerance = torch.full((2,), 0.10)

    cost = weighted_arm_family_excess_cost(
        family_errors=(shoulder, wrist),
        family_tolerances=(shoulder_tolerance, wrist_tolerance),
        family_weights=(1.0, 1.0),
    )

    assert cost[0].item() == pytest.approx(cost[1].item())

def test_arm_family_cost_has_zero_cost_inside_envelope():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        weighted_arm_family_excess_cost,
    )

    error = torch.tensor([
        [0.10, -0.10],
        [0.19, -0.19],
    ])
    tolerance = torch.tensor([0.20, 0.20])

    cost = weighted_arm_family_excess_cost(
        family_errors=(error,),
        family_tolerances=(tolerance,),
        family_weights=(1.0,),
    )

    assert cost.tolist() == pytest.approx([0.0, 0.0])


def test_first_touchdown_placement_score_grades_rear_margin():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        first_touchdown_placement_score,
    )

    rear = torch.tensor([
        -0.150,
        -0.120,
        -0.080,
         0.000,
         0.015,
         0.055,
    ])
    front = torch.full_like(rear, 0.020)

    score = first_touchdown_placement_score(
        rear,
        front,
        shaping_start_margin_m=-0.150,
        full_rear_margin_m=0.055,
        front_safe_margin_m=0.015,
    )

    assert score.tolist() == pytest.approx([
        0.0,
        0.14634146,
        0.34146341,
        0.73170732,
        0.80487805,
        1.0,
    ])


def test_first_touchdown_placement_score_penalizes_front_overshoot():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        first_touchdown_placement_score,
    )

    rear = torch.full((3,), 0.055)
    front = torch.tensor([0.015, 0.0075, 0.0])

    score = first_touchdown_placement_score(
        rear,
        front,
        shaping_start_margin_m=-0.150,
        full_rear_margin_m=0.055,
        front_safe_margin_m=0.015,
    )

    assert score.tolist() == pytest.approx([1.0, 0.5, 0.0])

def test_global_stair_target_update_advances_both_feet():
    import torch
    from simulation.isaac_lab.g1_stairs.reward_math import (
        global_stair_target_update,
    )

    target = torch.tensor([
        [0, 0],
        [1, 1],
        [2, 2],
    ])
    touchdown = torch.tensor([
        [False, True],
        [True, False],
        [False, False],
    ])

    old, acquired, updated = global_stair_target_update(
        target,
        touchdown,
        step_count=3,
    )

    assert old.tolist() == [0, 1, 2]
    assert acquired.tolist() == [True, True, False]
    assert updated.tolist() == [
        [1, 1],
        [2, 2],
        [2, 2],
    ]


def test_global_stair_target_update_counts_two_feet_once():
    import torch
    from simulation.isaac_lab.g1_stairs.reward_math import (
        global_stair_target_update,
    )

    target = torch.tensor([[2, 2]])
    touchdown = torch.tensor([[True, True]])

    old, acquired, updated = global_stair_target_update(
        target,
        touchdown,
        step_count=3,
    )

    assert old.tolist() == [2]
    assert acquired.tolist() == [True]
    assert updated.tolist() == [[3, 3]]


def test_alternating_foot_gate_starts_open_then_selects_one_foot():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        alternating_foot_gate,
    )

    preferred = torch.tensor([-1, 0, 1])
    gate = alternating_foot_gate(preferred)

    assert gate.tolist() == [
        [True, True],
        [True, False],
        [False, True],
    ]


def test_update_alternating_preferred_foot_chooses_opposite():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        update_alternating_preferred_foot,
    )

    preferred = torch.tensor([-1, 1, 0, -1])
    touchdown = torch.tensor([
        [True, False],
        [False, True],
        [False, False],
        [True, True],
    ])
    acquired = torch.tensor([True, True, False, True])

    updated = update_alternating_preferred_foot(
        preferred,
        touchdown,
        acquired,
    )

    assert updated.tolist() == [1, 0, 0, -1]


def test_anticipatory_clearance_target_is_local_and_closes_after_edge():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        anticipatory_clearance_target,
    )

    distance = torch.tensor([0.25, 0.22, 0.135, 0.05, 0.0, -0.01])
    full = torch.full_like(distance, 0.215)

    target, eligible = anticipatory_clearance_target(
        distance,
        full,
        start_distance_m=0.22,
        full_distance_m=0.05,
    )

    assert eligible.tolist() == [False, True, True, True, True, False]
    assert target[0].item() == pytest.approx(0.0)
    assert target[1].item() == pytest.approx(0.0)
    assert target[2].item() == pytest.approx(0.1075, abs=1e-6)
    assert target[3].item() == pytest.approx(0.215, abs=1e-6)
    assert target[4].item() == pytest.approx(0.215, abs=1e-6)


def test_bounded_vertical_progress_cannot_reward_horizontal_or_abrupt_motion():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        bounded_vertical_progress_credit,
    )

    current = torch.tensor([0.020, 0.020, 0.200])
    previous = torch.tensor([0.000, 0.020, 0.000])
    target = torch.tensor([0.200, 0.200, 0.200])
    full = torch.tensor([0.215, 0.215, 0.215])

    credit = bounded_vertical_progress_credit(
        current,
        previous,
        target,
        full,
        max_credit_vertical_speed_mps=0.60,
        dt_s=0.02,
    )

    capped = 0.60 * 0.02 / 0.215
    assert credit.tolist() == pytest.approx([capped, 0.0, capped])


def test_monotonic_anticipatory_clearance_cannot_farm_lift_drop_lift():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        bounded_monotonic_vertical_progress_credit,
    )

    best = torch.zeros(1)
    target = torch.tensor([0.055])
    full = torch.tensor([0.055])

    best, c1 = bounded_monotonic_vertical_progress_credit(
        torch.tensor([0.020]), best, target, full,
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    best, c2 = bounded_monotonic_vertical_progress_credit(
        torch.tensor([0.005]), best, target, full,
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    best, c3 = bounded_monotonic_vertical_progress_credit(
        torch.tensor([0.020]), best, target, full,
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    best, c4 = bounded_monotonic_vertical_progress_credit(
        torch.tensor([0.030]), best, target, full,
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )

    assert c1.item() == pytest.approx(0.020 / 0.055)
    assert c2.item() == pytest.approx(0.0)
    assert c3.item() == pytest.approx(0.0)
    assert c4.item() == pytest.approx(0.010 / 0.055)
    assert best.item() == pytest.approx(0.030)


def test_monotonic_anticipatory_clearance_advances_peak_when_speed_capped():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        bounded_monotonic_vertical_progress_credit,
    )

    best, credit = bounded_monotonic_vertical_progress_credit(
        torch.tensor([0.050]),
        torch.tensor([0.000]),
        torch.tensor([0.055]),
        torch.tensor([0.055]),
        max_credit_vertical_speed_mps=0.60,
        dt_s=0.02,
    )

    assert credit.item() == pytest.approx(0.012 / 0.055)
    assert best.item() == pytest.approx(0.050)


def test_monotonic_anticipatory_clearance_suppresses_strike_bounce():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        bounded_monotonic_vertical_progress_credit,
    )

    best, credit = bounded_monotonic_vertical_progress_credit(
        torch.tensor([0.040]),
        torch.tensor([0.010]),
        torch.tensor([0.055]),
        torch.tensor([0.055]),
        max_credit_vertical_speed_mps=2.0,
        dt_s=0.02,
        suppress_credit=torch.tensor([True]),
    )

    assert credit.item() == pytest.approx(0.0)
    assert best.item() == pytest.approx(0.040)


def test_anticipatory_attempt_starts_locally_and_cannot_recycle_or_restart():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        bounded_anticipatory_attempt_credit,
    )

    best = torch.zeros(1)
    active = torch.tensor([False])
    spent = torch.tensor([False])
    target = torch.tensor([0.055])
    full = torch.tensor([0.055])

    # Ordinary approach/ineligible height cannot consume the local budget.
    best, active, spent, c0 = bounded_anticipatory_attempt_credit(
        torch.tensor([0.025]), best, target, full, active, spent,
        torch.tensor([False]), torch.tensor([False]),
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    assert best.item() == pytest.approx(0.0)
    assert c0.item() == pytest.approx(0.0)

    # First eligible sample establishes the local baseline with no entry bonus.
    best, active, spent, c1 = bounded_anticipatory_attempt_credit(
        torch.tensor([0.025]), best, target, full, active, spent,
        torch.tensor([True]), torch.tensor([False]),
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    assert active.item() is True
    assert best.item() == pytest.approx(0.025)
    assert c1.item() == pytest.approx(0.0)

    best, active, spent, c2 = bounded_anticipatory_attempt_credit(
        torch.tensor([0.035]), best, target, full, active, spent,
        torch.tensor([True]), torch.tensor([False]),
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    assert c2.item() == pytest.approx(0.010 / 0.055)

    # Drop and relift to the same peak cannot recycle credit.
    best, active, spent, c3 = bounded_anticipatory_attempt_credit(
        torch.tensor([0.010]), best, target, full, active, spent,
        torch.tensor([True]), torch.tensor([False]),
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    best, active, spent, c4 = bounded_anticipatory_attempt_credit(
        torch.tensor([0.035]), best, target, full, active, spent,
        torch.tensor([True]), torch.tensor([False]),
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    assert c3.item() == pytest.approx(0.0)
    assert c4.item() == pytest.approx(0.0)

    # Strike/replant invalidates the attempt and blocks another attempt on the
    # same stair target. Collision-induced height receives no credit.
    best, active, spent, c5 = bounded_anticipatory_attempt_credit(
        torch.tensor([0.050]), best, target, full, active, spent,
        torch.tensor([True]), torch.tensor([True]),
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    assert c5.item() == pytest.approx(0.0)
    assert active.item() is False
    assert spent.item() is True

    best2, active2, spent2, c6 = bounded_anticipatory_attempt_credit(
        torch.tensor([0.055]), best, target, full, active, spent,
        torch.tensor([True]), torch.tensor([False]),
        max_credit_vertical_speed_mps=2.0, dt_s=0.02,
    )
    assert c6.item() == pytest.approx(0.0)
    assert active2.item() is False
    assert spent2.item() is True
    assert best2.item() == pytest.approx(best.item())


def test_first_touchdown_reward_is_global_one_shot_and_rejects_ambiguous():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        first_touchdown_reward_event,
    )

    new_attempt = torch.tensor([
        [True, False],
        [True, True],
        [True, False],
        [False, True],
    ])
    preferred = torch.tensor([
        [True, True],
        [True, True],
        [False, True],
        [True, True],
    ])
    seen = torch.tensor([False, False, False, True])

    event, seen_now = first_touchdown_reward_event(
        new_attempt, preferred, seen
    )

    assert event.tolist() == [
        [True, False],
        [False, False],
        [False, False],
        [False, False],
    ]
    assert seen_now.tolist() == [True, True, True, True]


def test_nonpreferred_acquisition_event_does_not_block_first_step():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        nonpreferred_acquisition_event,
    )

    preferred = torch.tensor([-1, 1, 0, 1])
    touchdown = torch.tensor([
        [True, False],   # first acquisition: unconstrained
        [True, False],   # left acquires while right preferred
        [True, False],   # left acquires while left preferred
        [False, True],   # right acquires while right preferred
    ])

    event = nonpreferred_acquisition_event(preferred, touchdown)
    assert event.tolist() == [False, True, False, False]


def test_bounded_impact_severity_distinguishes_hard_strikes_and_saturates():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        bounded_impact_severity,
    )

    force_x = torch.tensor([-30.0, -140.0, -250.0, -659.0, -659.0])
    event = torch.tensor([True, True, True, True, False])

    severity = bounded_impact_severity(
        force_x,
        event,
        minimum_force_n=30.0,
        full_severity_force_n=250.0,
        minimum_event_fraction=0.25,
    )

    assert severity.tolist() == pytest.approx([
        0.25,
        0.625,
        1.0,
        1.0,
        0.0,
    ])


def test_preferred_transition_contact_requires_lift_and_penalizes_only_replant():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        preferred_transition_contact_update,
    )

    armed = torch.tensor([False, False, True, True])
    resolved = torch.tensor([False, False, False, False])
    clear_lift = torch.tensor([False, True, False, False])
    old_contact = torch.tensor([True, False, True, False])
    next_contact = torch.tensor([False, False, False, True])

    armed2, resolved2, replant, direct = preferred_transition_contact_update(
        armed,
        resolved,
        clear_lift,
        old_contact,
        next_contact,
    )

    assert armed2.tolist() == [False, True, True, True]
    assert replant.tolist() == [False, False, True, False]
    assert direct.tolist() == [False, False, False, True]
    assert resolved2.tolist() == [False, False, True, True]


def test_step_over_transition_requires_clearance_before_success():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_transition_update,
    )

    z = torch.tensor([False])
    lifted, cleared, seen, resolved, reward, code = (
        step_over_transition_update(
            torch.tensor([True]),
            z,
            z,
            z,
            z,
            clear_lift=torch.tensor([True]),
            clear_riser=z,
            old_tread_contact=z,
            next_tread_contact=torch.tensor([True]),
            strike_event=z,
            forward_excursion_violation=z,
            acquisition_event=z,
            preferred_acquisition=z,
            single_acquisition=z,
            strike_severity=torch.tensor([0.0]),
        )
    )

    assert lifted.item() is True
    assert cleared.item() is False
    assert seen.item() is False
    assert resolved.item() is True
    assert reward.item() == pytest.approx(-1.0)
    assert code.item() == 5


def test_step_over_transition_direct_next_success_sequence():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_transition_update,
    )

    active = torch.tensor([True])
    z = torch.tensor([False])

    lifted, cleared, seen, resolved, reward, code = (
        step_over_transition_update(
            active,
            z,
            z,
            z,
            z,
            clear_lift=torch.tensor([True]),
            clear_riser=torch.tensor([True]),
            old_tread_contact=z,
            next_tread_contact=torch.tensor([True]),
            strike_event=z,
            forward_excursion_violation=z,
            acquisition_event=z,
            preferred_acquisition=z,
            single_acquisition=z,
            strike_severity=torch.tensor([0.0]),
        )
    )
    assert lifted.item() is True
    assert cleared.item() is True
    assert seen.item() is True
    assert resolved.item() is False
    assert reward.item() == pytest.approx(0.0)
    assert code.item() == 0

    lifted, cleared, seen, resolved, reward, code = (
        step_over_transition_update(
            active,
            lifted,
            cleared,
            seen,
            resolved,
            clear_lift=z,
            clear_riser=z,
            old_tread_contact=z,
            next_tread_contact=z,
            strike_event=z,
            forward_excursion_violation=z,
            acquisition_event=torch.tensor([True]),
            preferred_acquisition=torch.tensor([True]),
            single_acquisition=torch.tensor([True]),
            strike_severity=torch.tensor([0.0]),
        )
    )
    assert resolved.item() is True
    assert reward.item() == pytest.approx(1.0)
    assert code.item() == 1


def test_step_over_transition_replant_resolves_once():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_transition_update,
    )

    active = torch.tensor([True])
    z = torch.tensor([False])
    lifted = torch.tensor([True])

    out = step_over_transition_update(
        active,
        lifted,
        z,
        z,
        z,
        clear_lift=z,
        clear_riser=z,
        old_tread_contact=torch.tensor([True]),
        next_tread_contact=z,
        strike_event=z,
        forward_excursion_violation=z,
        acquisition_event=z,
        preferred_acquisition=z,
        single_acquisition=z,
        strike_severity=torch.tensor([0.0]),
    )
    assert out[3].item() is True
    assert out[4].item() == pytest.approx(-1.0)
    assert out[5].item() == 2

    out2 = step_over_transition_update(
        active,
        out[0],
        out[1],
        out[2],
        out[3],
        clear_lift=z,
        clear_riser=z,
        old_tread_contact=torch.tensor([True]),
        next_tread_contact=z,
        strike_event=z,
        forward_excursion_violation=z,
        acquisition_event=z,
        preferred_acquisition=z,
        single_acquisition=z,
        strike_severity=torch.tensor([0.0]),
    )
    assert out2[4].item() == pytest.approx(0.0)
    assert out2[5].item() == 0


def test_step_over_transition_hard_strike_is_bounded_but_costlier():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_transition_update,
    )

    active = torch.tensor([True, True])
    z = torch.tensor([False, False])
    out = step_over_transition_update(
        active,
        z,
        z,
        z,
        z,
        clear_lift=z,
        clear_riser=z,
        old_tread_contact=z,
        next_tread_contact=z,
        strike_event=torch.tensor([True, True]),
        forward_excursion_violation=z,
        acquisition_event=z,
        preferred_acquisition=z,
        single_acquisition=z,
        strike_severity=torch.tensor([0.1, 1.0]),
        strike_failure_base=-0.5,
        strike_failure_severity_scale=-0.5,
    )
    assert out[4].tolist() == pytest.approx([-0.55, -1.0])
    assert out[5].tolist() == [3, 3]


def test_step_over_transition_nonpreferred_acquisition_fails():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_transition_update,
    )

    t = torch.tensor([True])
    z = torch.tensor([False])
    out = step_over_transition_update(
        t,
        t,
        t,
        t,
        z,
        clear_lift=z,
        clear_riser=z,
        old_tread_contact=z,
        next_tread_contact=z,
        strike_event=z,
        forward_excursion_violation=z,
        acquisition_event=t,
        preferred_acquisition=z,
        single_acquisition=t,
        strike_severity=torch.tensor([0.0]),
    )
    assert out[3].item() is True
    assert out[4].item() == pytest.approx(-1.0)
    assert out[5].item() == 4


def test_step_over_transition_forward_excursion_is_failure():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_transition_update,
    )

    t = torch.tensor([True])
    z = torch.tensor([False])
    out = step_over_transition_update(
        t,
        t,
        z,
        z,
        z,
        clear_lift=z,
        clear_riser=z,
        old_tread_contact=z,
        next_tread_contact=z,
        strike_event=z,
        forward_excursion_violation=t,
        acquisition_event=z,
        preferred_acquisition=z,
        single_acquisition=z,
        strike_severity=torch.tensor([0.0]),
    )
    assert out[3].item() is True
    assert out[4].item() == pytest.approx(-1.0)
    assert out[5].item() == 6


def test_escrowed_transition_reward_clean_sequence_totals_one():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        escrowed_transition_reward,
    )

    paid = torch.tensor([0.0])

    r1, paid = escrowed_transition_reward(
        torch.tensor([0.0]),
        torch.tensor([0]),
        torch.tensor([True]),
        torch.tensor([False]),
        paid,
        lift_milestone_value=0.20,
        clearance_milestone_value=0.30,
    )
    r2, paid = escrowed_transition_reward(
        torch.tensor([0.0]),
        torch.tensor([0]),
        torch.tensor([False]),
        torch.tensor([True]),
        paid,
        lift_milestone_value=0.20,
        clearance_milestone_value=0.30,
    )
    r3, paid = escrowed_transition_reward(
        torch.tensor([1.0]),
        torch.tensor([1]),
        torch.tensor([False]),
        torch.tensor([False]),
        paid,
        lift_milestone_value=0.20,
        clearance_milestone_value=0.30,
    )

    assert r1.item() == pytest.approx(0.20)
    assert r2.item() == pytest.approx(0.30)
    assert r3.item() == pytest.approx(0.50)
    assert (r1 + r2 + r3).item() == pytest.approx(1.0)


def test_escrowed_transition_reward_replant_repays_partial_credit():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        escrowed_transition_reward,
    )

    paid = torch.tensor([0.50])
    reward, paid_after = escrowed_transition_reward(
        torch.tensor([-1.0]),
        torch.tensor([2]),
        torch.tensor([False]),
        torch.tensor([False]),
        paid,
    )

    assert reward.item() == pytest.approx(-1.50)
    assert (torch.tensor(0.50) + reward).item() == pytest.approx(-1.0)
    assert paid_after.item() == pytest.approx(0.50)


def test_escrowed_transition_reward_cannot_farm_repeated_milestones():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        escrowed_transition_reward,
    )

    paid = torch.tensor([0.20])
    reward, paid_after = escrowed_transition_reward(
        torch.tensor([0.0]),
        torch.tensor([0]),
        torch.tensor([False]),
        torch.tensor([False]),
        paid,
    )

    assert reward.item() == pytest.approx(0.0)
    assert paid_after.item() == pytest.approx(0.20)


def test_step_over_strike_failure_is_not_cheaper_than_generic_failure():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_transition_update,
    )

    active = torch.tensor([True])
    z = torch.tensor([False])
    out = step_over_transition_update(
        active,
        z,
        z,
        z,
        z,
        clear_lift=z,
        clear_riser=z,
        old_tread_contact=z,
        next_tread_contact=z,
        strike_event=torch.tensor([True]),
        forward_excursion_violation=z,
        acquisition_event=z,
        preferred_acquisition=z,
        single_acquisition=z,
        strike_severity=torch.tensor([0.10]),
        failure_value=-1.0,
        strike_failure_base=-1.0,
        strike_failure_severity_scale=-0.5,
    )

    assert out[4].item() == pytest.approx(-1.05)
    assert out[4].item() <= -1.0
    assert out[5].item() == 3



def test_crossing_attempt_invalid_latches_until_replant():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        update_crossing_attempt_invalid,
    )

    x = torch.tensor([[False, False]])

    x = update_crossing_attempt_invalid(
        x,
        torch.tensor([[False, True]]),
        torch.tensor([[False, False]]),
    )
    assert x.tolist() == [[False, True]]

    # Collision recovery remains invalid.
    x = update_crossing_attempt_invalid(
        x,
        torch.tensor([[False, False]]),
        torch.tensor([[False, False]]),
    )
    assert x.tolist() == [[False, True]]

    # Safe lower-support replant permits a new swing.
    x = update_crossing_attempt_invalid(
        x,
        torch.tensor([[False, False]]),
        torch.tensor([[False, True]]),
    )
    assert x.tolist() == [[False, False]]


def test_crossing_attempt_strike_wins_over_replant():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        update_crossing_attempt_invalid,
    )

    x = update_crossing_attempt_invalid(
        torch.tensor([[False, True]]),
        torch.tensor([[False, True]]),
        torch.tensor([[False, True]]),
    )
    assert x.tolist() == [[False, True]]


def test_b6_clean_step_over_direct_direct_only_positive():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        clean_step_over_sequence_update,
    )

    active = torch.tensor([False])
    first = torch.tensor([False])
    failed = torch.tensor([False])
    paid = torch.tensor([False])

    # stair1 -> stair2 DIRECT: no reward yet.
    active, first, failed, paid, signal = (
        clean_step_over_sequence_update(
            active,
            first,
            failed,
            paid,
            torch.tensor([True]),
            torch.tensor([0]),
            torch.tensor([1]),
        )
    )

    assert signal.item() == 0
    assert first.item()
    assert not failed.item()
    assert not paid.item()

    # stair2 -> stair3 DIRECT: exactly one clean completion.
    active, first, failed, paid, signal = (
        clean_step_over_sequence_update(
            active,
            first,
            failed,
            paid,
            torch.tensor([True]),
            torch.tensor([1]),
            torch.tensor([1]),
        )
    )

    assert signal.item() == 1
    assert not active.item()
    assert not first.item()
    assert not failed.item()
    assert paid.item()


def test_b6_sequence_failure_is_latched_and_cannot_recover():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        clean_step_over_sequence_update,
    )

    active = torch.tensor([False])
    first = torch.tensor([False])
    failed = torch.tensor([False])
    paid = torch.tensor([False])

    # Transition 1 failure.
    active, first, failed, paid, signal = (
        clean_step_over_sequence_update(
            active,
            first,
            failed,
            paid,
            torch.tensor([True]),
            torch.tensor([0]),
            torch.tensor([3]),
        )
    )

    assert signal.item() == -1
    assert failed.item()

    # Later DIRECT must never recover.
    active, first, failed, paid, signal = (
        clean_step_over_sequence_update(
            active,
            first,
            failed,
            paid,
            torch.tensor([True]),
            torch.tensor([1]),
            torch.tensor([1]),
        )
    )

    assert signal.item() == 0
    assert failed.item()
    assert not paid.item()


def test_b6_first_direct_then_failure_cannot_later_pay():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        clean_step_over_sequence_update,
    )

    active = torch.tensor([False])
    first = torch.tensor([False])
    failed = torch.tensor([False])
    paid = torch.tensor([False])

    active, first, failed, paid, signal = (
        clean_step_over_sequence_update(
            active,
            first,
            failed,
            paid,
            torch.tensor([True]),
            torch.tensor([0]),
            torch.tensor([1]),
        )
    )
    assert signal.item() == 0
    assert first.item()

    active, first, failed, paid, signal = (
        clean_step_over_sequence_update(
            active,
            first,
            failed,
            paid,
            torch.tensor([True]),
            torch.tensor([1]),
            torch.tensor([2]),
        )
    )
    assert signal.item() == -1
    assert failed.item()
    assert not first.item()

    active, first, failed, paid, signal = (
        clean_step_over_sequence_update(
            active,
            first,
            failed,
            paid,
            torch.tensor([True]),
            torch.tensor([1]),
            torch.tensor([1]),
        )
    )
    assert signal.item() == 0
    assert failed.item()
    assert not paid.item()


def test_b6_transition_classifier_priority():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        classify_clean_step_over_transition,
    )

    # direct, replant-over-direct, wrong, unresolved, ambiguous-over-replant
    code = classify_clean_step_over_transition(
        transition_active=torch.tensor(
            [True, True, True, True, True]
        ),
        acquisition_event=torch.tensor(
            [True, True, True, False, True]
        ),
        unique_acquisition=torch.tensor(
            [True, True, True, True, False]
        ),
        preferred_acquisition=torch.tensor(
            [True, True, False, False, False]
        ),
        old_tread_replant=torch.tensor(
            [False, True, False, False, True]
        ),
        episode_end=torch.tensor(
            [False, False, False, True, False]
        ),
    )

    assert code.tolist() == [1, 2, 3, 4, 5]


def _b6_test_step(state, target, event_code):
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        clean_step_over_sequence_reward,
        clean_step_over_sequence_update,
    )

    active, first, failed, paid = state

    (
        active,
        first,
        failed,
        paid,
        signal,
    ) = clean_step_over_sequence_update(
        active,
        first,
        failed,
        paid,
        torch.tensor([True]),
        torch.tensor([target]),
        torch.tensor([event_code]),
    )

    reward = clean_step_over_sequence_reward(
        signal,
        torch.tensor([event_code]),
        success_value=3.0,
        replant_value=-0.10,
        wrong_foot_value=-0.10,
        unresolved_value=-0.10,
    )

    return (active, first, failed, paid), reward.item()


def _b6_initial_state():
    import torch

    return (
        torch.tensor([False]),  # sequence_active
        torch.tensor([False]),  # first_direct
        torch.tensor([False]),  # sequence_failed
        torch.tensor([False]),  # clean_paid
    )


def test_b6_direct_direct_pays_three_once_only():
    state = _b6_initial_state()

    state, r1 = _b6_test_step(state, 0, 1)
    state, r2 = _b6_test_step(state, 1, 1)
    state, r3 = _b6_test_step(state, 1, 1)

    assert r1 == 0.0
    assert r2 == 3.0
    assert r3 == 0.0

    active, first, failed, paid = state
    assert not active.item()
    assert not first.item()
    assert not failed.item()
    assert paid.item()


def test_b6_first_transition_replant_latches_failure():
    state = _b6_initial_state()

    state, r1 = _b6_test_step(state, 0, 2)
    state, r2 = _b6_test_step(state, 0, 1)
    state, r3 = _b6_test_step(state, 1, 1)

    assert abs(r1 - (-0.10)) < 1.0e-6
    assert r2 == 0.0
    assert r3 == 0.0

    assert state[2].item()
    assert not state[3].item()


def test_b6_wrong_foot_first_transition_latches_failure():
    state = _b6_initial_state()

    state, reward = _b6_test_step(state, 0, 3)

    assert abs(reward - (-0.10)) < 1.0e-6
    assert state[2].item()
    assert not state[3].item()


def test_b6_ambiguous_acquisition_latches_failure():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        classify_clean_step_over_transition,
    )

    code = classify_clean_step_over_transition(
        transition_active=torch.tensor([True]),
        acquisition_event=torch.tensor([True]),
        unique_acquisition=torch.tensor([False]),
        preferred_acquisition=torch.tensor([False]),
        old_tread_replant=torch.tensor([False]),
        episode_end=torch.tensor([False]),
    )

    assert code.item() == 5

    state = _b6_initial_state()
    state, reward = _b6_test_step(
        state,
        0,
        code.item(),
    )

    assert abs(reward - (-0.10)) < 1.0e-6
    assert state[2].item()
    assert not state[3].item()


def test_b6_direct_then_canonical_replant_never_pays_clean():
    state = _b6_initial_state()

    # First transition is genuinely DIRECT.
    state, r1 = _b6_test_step(state, 0, 1)

    # This is REPLANT_OLD only because the canonical classifier
    # has already classified the physical event as such.
    state, r2 = _b6_test_step(state, 1, 2)

    # A later apparently clean acquisition cannot recover.
    state, r3 = _b6_test_step(state, 1, 1)

    assert r1 == 0.0
    assert abs(r2 - (-0.10)) < 1.0e-6
    assert r3 == 0.0

    assert state[2].item()
    assert not state[3].item()


def test_b6_failed_sequence_can_never_restart_same_episode():
    state = _b6_initial_state()

    state, r1 = _b6_test_step(state, 0, 3)

    # Even an apparently perfect later pair cannot restart B6.
    state, r2 = _b6_test_step(state, 0, 1)
    state, r3 = _b6_test_step(state, 1, 1)

    assert abs(r1 - (-0.10)) < 1.0e-6
    assert r2 == 0.0
    assert r3 == 0.0

    assert state[2].item()
    assert not state[3].item()


def test_b6_fresh_episode_state_allows_clean_after_previous_failure():
    # Previous episode fails.
    state = _b6_initial_state()
    state, reward = _b6_test_step(state, 0, 3)

    assert abs(reward - (-0.10)) < 1.0e-6
    assert state[2].item()

    # Environment reset contract: all B6 latches return false.
    state = _b6_initial_state()

    active, first, failed, paid = state
    assert not active.item()
    assert not first.item()
    assert not failed.item()
    assert not paid.item()

    # A new episode may legitimately complete CLEAN_STEP_OVER.
    state, r1 = _b6_test_step(state, 0, 1)
    state, r2 = _b6_test_step(state, 1, 1)

    assert r1 == 0.0
    assert r2 == 3.0
    assert state[3].item()


def test_b6_positive_reward_implies_clean_paid():
    state = _b6_initial_state()

    state, r1 = _b6_test_step(state, 0, 1)
    assert r1 == 0.0
    assert not state[3].item()

    state, r2 = _b6_test_step(state, 1, 1)

    assert r2 == 3.0
    assert state[3].item()


def test_b6_failed_state_can_never_emit_positive_reward():
    state = _b6_initial_state()

    state, _ = _b6_test_step(state, 0, 2)

    assert state[2].item()

    for target, code in (
        (0, 1),
        (1, 1),
        (0, 1),
        (1, 1),
    ):
        state, reward = _b6_test_step(
            state,
            target,
            code,
        )
        assert reward <= 0.0

    assert state[2].item()
    assert not state[3].item()


def test_b7_sequence_potential_exact_values():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        clean_step_over_potential_shaping,
    )

    gamma = 0.99
    p = 0.5

    false = torch.tensor([False])
    true = torch.tensor([True])

    # DIRECT1: Phi 0 -> 0.5
    _, _, value = clean_step_over_potential_shaping(
        false,
        true,
        gamma=gamma,
        potential_value=p,
    )
    assert torch.allclose(value, torch.tensor([0.495]))

    # Waiting: Phi 0.5 -> 0.5
    _, _, value = clean_step_over_potential_shaping(
        true,
        true,
        gamma=gamma,
        potential_value=p,
    )
    assert torch.allclose(value, torch.tensor([-0.005]), atol=1e-7)

    # Failure or CLEAN: Phi 0.5 -> 0
    _, _, value = clean_step_over_potential_shaping(
        true,
        false,
        gamma=gamma,
        potential_value=p,
    )
    assert torch.allclose(value, torch.tensor([-0.5]))

    # Outside the sequence: Phi 0 -> 0
    _, _, value = clean_step_over_potential_shaping(
        false,
        false,
        gamma=gamma,
        potential_value=p,
    )
    assert torch.allclose(value, torch.tensor([0.0]))


def test_b7_sequence_potential_discounted_telescoping():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        clean_step_over_potential_shaping,
    )

    gamma = 0.99
    p = 0.5

    false = torch.tensor([False])
    true = torch.tensor([True])

    _, _, direct1 = clean_step_over_potential_shaping(
        false, true, gamma=gamma, potential_value=p
    )
    _, _, waiting = clean_step_over_potential_shaping(
        true, true, gamma=gamma, potential_value=p
    )
    _, _, failure = clean_step_over_potential_shaping(
        true, false, gamma=gamma, potential_value=p
    )

    discounted = (
        direct1
        + gamma * waiting
        + (gamma ** 2) * failure
    )

    assert torch.allclose(
        discounted,
        torch.zeros_like(discounted),
        atol=1e-7,
    )


def test_b7_zero_potential_reproduces_b6_shaping():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        clean_step_over_potential_shaping,
    )

    false = torch.tensor([False, True, True])
    nxt = torch.tensor([True, True, False])

    _, _, shaping = clean_step_over_potential_shaping(
        false,
        nxt,
        gamma=0.99,
        potential_value=0.0,
    )

    assert torch.count_nonzero(shaping) == 0

