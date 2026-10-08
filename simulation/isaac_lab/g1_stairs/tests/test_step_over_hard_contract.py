import torch

from simulation.isaac_lab.g1_stairs.geometry_episode import (
    EpisodeContract,
    EpisodeTracker,
    Result,
)
from simulation.isaac_lab.g1_stairs.reward_math import (
    step_over_transition_update,
)


def test_prelift_old_tread_consolidation_is_failure():
    yes = torch.tensor([True])
    no = torch.tensor([False])

    lifted, cleared, next_seen, resolved, reward, code = (
        step_over_transition_update(
            active=yes,
            lifted=no,
            cleared=no,
            next_contact_seen=no,
            resolved=no,
            clear_lift=no,
            clear_riser=no,
            old_tread_contact=no,
            next_tread_contact=no,
            strike_event=no,
            forward_excursion_violation=no,
            acquisition_event=no,
            preferred_acquisition=no,
            single_acquisition=yes,
            strike_severity=torch.tensor([0.0]),
            prelift_old_tread_support=yes,
        )
    )

    assert not lifted.item()
    assert not cleared.item()
    assert not next_seen.item()
    assert resolved.item()
    assert reward.item() == -1.0
    assert code.item() == 7


def test_lower_support_before_committed_lift_is_allowed():
    yes = torch.tensor([True])
    no = torch.tensor([False])

    _, _, _, resolved, reward, code = step_over_transition_update(
        active=yes,
        lifted=no,
        cleared=no,
        next_contact_seen=no,
        resolved=no,
        clear_lift=no,
        clear_riser=no,
        old_tread_contact=no,
        next_tread_contact=no,
        strike_event=no,
        forward_excursion_violation=no,
        acquisition_event=no,
        preferred_acquisition=no,
        single_acquisition=yes,
        strike_severity=torch.tensor([0.0]),
        prelift_old_tread_support=no,
    )

    assert not resolved.item()
    assert reward.item() == 0.0
    assert code.item() == 0


def test_step_over_contract_violation_is_native_terminal():
    tracker = EpisodeTracker(
        1,
        torch.device("cpu"),
        EpisodeContract(),
        first_riser=1.0,
        tread=0.25,
        step_count=3,
        course_start=-1.0,
        course_end=4.0,
        half_width=1.0,
        dt=0.02,
    )

    reason = tracker.update(
        position=torch.tensor([[0.0, 0.0, 0.8]]),
        roll_pitch=torch.zeros((1, 2)),
        riser_height=torch.tensor([0.2]),
        foot_position=torch.tensor(
            [[[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]]
        ),
        foot_force_z=torch.zeros((1, 2)),
        forbidden_contact=torch.tensor([False]),
        invalid_state=torch.tensor([False]),
        joint_limit=torch.tensor([False]),
        velocity_limit=torch.tensor([False]),
        torque_limit=torch.tensor([False]),
        timed_out=torch.tensor([False]),
        step_over_contract_violation=torch.tensor([True]),
    )

    assert reason.item() == Result.STEP_OVER_CONTRACT



def test_hard_gate_starts_after_first_stair():
    from simulation.isaac_lab.g1_stairs.reward_math import (
        step_over_hard_failure_mask,
    )

    event_code = torch.tensor([3, 2, 4, 7, 1])
    target = torch.tensor([0, 0, 1, 2, 2])

    mask = step_over_hard_failure_mask(event_code, target)

    assert mask.tolist() == [
        False,  # first-riser strike remains soft
        False,  # first-riser replant remains soft
        True,   # wrong-foot stair-to-stair acquisition
        True,   # pre-lift same-tread consolidation
        False,  # clean success
    ]



def test_vertical_clearance_progress_is_non_farmable():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        monotonic_progress_credit,
    )

    best = torch.zeros(1)
    eligible = torch.tensor([True])

    best, c1 = monotonic_progress_credit(
        torch.tensor([0.25]),
        best,
        eligible,
        total_value=0.30,
    )
    best, c2 = monotonic_progress_credit(
        torch.tensor([0.10]),
        best,
        eligible,
        total_value=0.30,
    )
    best, c3 = monotonic_progress_credit(
        torch.tensor([0.80]),
        best,
        eligible,
        total_value=0.30,
    )
    best, c4 = monotonic_progress_credit(
        torch.tensor([2.00]),
        best,
        eligible,
        total_value=0.30,
    )

    torch.testing.assert_close(c1, torch.tensor([0.075]))
    torch.testing.assert_close(c2, torch.tensor([0.0]))
    torch.testing.assert_close(c3, torch.tensor([0.165]))
    torch.testing.assert_close(c4, torch.tensor([0.060]))
    torch.testing.assert_close(
        c1 + c2 + c3 + c4,
        torch.tensor([0.30]),
    )
    torch.testing.assert_close(best, torch.tensor([1.0]))


def test_vertical_clearance_credit_is_repaid_on_failure():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        escrowed_transition_reward,
    )

    zeros_b = torch.tensor([False])
    paid = torch.zeros(1)

    first_reward, paid = escrowed_transition_reward(
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        zeros_b,
        zeros_b,
        paid,
        lift_milestone_value=0.0,
        clearance_milestone_value=0.0,
        progress_credit=torch.tensor([0.18]),
    )

    failure_reward, _ = escrowed_transition_reward(
        torch.tensor([-1.0]),
        torch.tensor([2]),
        zeros_b,
        zeros_b,
        paid,
        lift_milestone_value=0.0,
        clearance_milestone_value=0.0,
        progress_credit=torch.tensor([0.12]),
    )

    # Earlier shaping is exactly cancelled at failure.
    torch.testing.assert_close(
        first_reward + failure_reward,
        torch.tensor([-1.0]),
    )


def test_terminal_step_cannot_collect_clearance_credit():
    import torch

    from simulation.isaac_lab.g1_stairs.reward_math import (
        escrowed_transition_reward,
    )

    zeros_b = torch.tensor([False])

    reward, paid = escrowed_transition_reward(
        torch.tensor([-1.0]),
        torch.tensor([2]),
        zeros_b,
        zeros_b,
        torch.zeros(1),
        lift_milestone_value=0.0,
        clearance_milestone_value=0.0,
        progress_credit=torch.tensor([0.30]),
    )

    torch.testing.assert_close(reward, torch.tensor([-1.0]))
    torch.testing.assert_close(paid, torch.tensor([0.0]))
