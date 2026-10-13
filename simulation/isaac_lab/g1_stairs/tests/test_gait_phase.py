"""Run directly with Isaac's Python; no simulator startup needed."""

import math
import unittest
import torch

from simulation.isaac_lab.g1_stairs.gait_phase import (
    advance_gait_phase,
    gait_frequency,
    gait_phase_observation,
    phase_swing_contact_loss,
)


class TestGaitPhase(unittest.TestCase):
    def test_frequency_mapping_and_fixed_control(self):
        raw = torch.tensor([-100.0, 0.0, 100.0])
        torch.testing.assert_close(
            gait_frequency(raw), torch.tensor([0.7, 1.0, 1.3])
        )
        torch.testing.assert_close(
            gait_frequency(raw, learned=False), torch.ones(3)
        )
        differentiable = torch.tensor([0.0], requires_grad=True)
        gait_frequency(differentiable).sum().backward()
        torch.testing.assert_close(
            differentiable.grad, torch.tensor([0.3])
        )

    def test_phase_advance_and_half_cycle_offset(self):
        phase = torch.tensor([0.875, 0.125])
        result = advance_gait_phase(
            phase, torch.ones(2), torch.ones(2, dtype=torch.bool),
            torch.tensor([-1, -1]), 0.25,
        )
        torch.testing.assert_close(result, torch.tensor([0.125, 0.375]))
        right = torch.remainder(result + 0.5, 1.0)
        torch.testing.assert_close(
            torch.remainder(right - result, 1.0),
            torch.full((2,), 0.5),
        )

    def test_acquisition_overrides_elapsed_phase_and_realigns(self):
        active = torch.ones(2, dtype=torch.bool)
        first = advance_gait_phase(
            torch.tensor([0.37, 0.81]), torch.ones(2),
            active, torch.tensor([0, 1]), 0.02,
        )
        torch.testing.assert_close(first, torch.tensor([0.0, 0.5]))
        second = advance_gait_phase(
            first, torch.ones(2), active, torch.tensor([1, 0]), 0.02,
        )
        torch.testing.assert_close(second, torch.tensor([0.5, 0.0]))

    def test_inactive_reset_and_completion_mask(self):
        inactive = torch.zeros(2, dtype=torch.bool)
        phase = advance_gait_phase(
            torch.tensor([0.2, 0.7]), torch.ones(2),
            inactive, torch.tensor([-1, 1]), 0.02,
        )
        torch.testing.assert_close(phase, torch.zeros(2))
        torch.testing.assert_close(
            gait_phase_observation(phase, inactive), torch.zeros(2, 2)
        )
        loss = phase_swing_contact_loss(
            torch.tensor([0.25, 0.75]), inactive,
            torch.full((2, 2), 100.0), touchdown_force_n=30.0,
        )
        torch.testing.assert_close(loss, torch.zeros(2))

    def test_left_acquisition_selects_right_swing_and_reverse(self):
        active = torch.ones(2, dtype=torch.bool)
        aligned = advance_gait_phase(
            torch.zeros(2), torch.ones(2), active,
            torch.tensor([0, 1]), 0.02,
        )
        phase = advance_gait_phase(
            aligned, torch.ones(2), active,
            torch.tensor([-1, -1]), 0.25,
        )
        stance_contact = torch.tensor([[100.0, 0.0], [0.0, 100.0]])
        swing_contact = stance_contact.flip(-1)
        torch.testing.assert_close(
            phase_swing_contact_loss(
                phase, active, stance_contact, touchdown_force_n=30.0,
            ), torch.zeros(2),
        )
        torch.testing.assert_close(
            phase_swing_contact_loss(
                phase, active, swing_contact, touchdown_force_n=30.0,
            ), torch.ones(2),
        )

    def test_contact_threshold_and_bound(self):
        phase = torch.tensor([0.25, 0.25])
        loss = phase_swing_contact_loss(
            phase, torch.ones(2, dtype=torch.bool),
            torch.tensor([[0.0, 29.9], [0.0, 30.0]]),
            touchdown_force_n=30.0,
        )
        torch.testing.assert_close(loss, torch.tensor([0.0, 1.0]))

        phase = torch.linspace(0.0, 1.0, 1001)[:-1]
        loss = phase_swing_contact_loss(
            phase, torch.ones(1000, dtype=torch.bool),
            torch.full((1000, 2), 100.0), touchdown_force_n=30.0,
        )
        assert torch.all((loss >= 0.0) & (loss <= 1.0))

    def test_sine_cosine_distinguishes_early_and_late_phase(self):
        features = gait_phase_observation(
            torch.tensor([0.125, 0.375]),
            torch.ones(2, dtype=torch.bool),
        )
        torch.testing.assert_close(features[0, 0], features[1, 0])
        assert features[0, 1] > 0 and features[1, 1] < 0

    def test_invalid_scalar_parameters_fail(self):
        for dt in (0.0, -1.0, math.nan):
            with self.assertRaises(ValueError):
                advance_gait_phase(
                    torch.zeros(1), torch.ones(1),
                    torch.ones(1, dtype=torch.bool),
                    torch.tensor([-1]), dt,
                )
        for threshold in (0.0, -1.0, math.nan):
            with self.assertRaises(ValueError):
                phase_swing_contact_loss(
                    torch.zeros(1), torch.ones(1, dtype=torch.bool),
                    torch.zeros(1, 2), touchdown_force_n=threshold,
                )


if __name__ == "__main__":
    unittest.main()
