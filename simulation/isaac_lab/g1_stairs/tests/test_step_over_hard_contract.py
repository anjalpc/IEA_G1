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
