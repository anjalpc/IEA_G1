"""Unit tests for paper-inspired, G1-unverified gait phase timing."""
import unittest

import torch

from simulation.isaac_lab.g1_stairs.contracts import FIXED_FORWARD_COMMAND_MPS
from simulation.isaac_lab.g1_stairs.footstep_gait_clock import (
    FootstepGaitClock, NOMINAL_DOUBLE_SUPPORT_S, NOMINAL_SINGLE_SUPPORT_S,
    NOMINAL_STEP_LENGTH_M,
)


class NominalCadenceTests(unittest.TestCase):
    def test_step_length_period_and_command_agree(self):
        period = NOMINAL_SINGLE_SUPPORT_S + NOMINAL_DOUBLE_SUPPORT_S
        implied = NOMINAL_STEP_LENGTH_M / period
        self.assertLess(abs(implied - FIXED_FORWARD_COMMAND_MPS), 0.02)

    def test_step_length_matches_nominal_tread(self):
        # Reciprocal plan: one foot per 0.25 m tread.
        self.assertAlmostEqual(NOMINAL_STEP_LENGTH_M, 0.25)


class GaitClockTests(unittest.TestCase):
    def create(self):
        return FootstepGaitClock(num_envs=2, device="cpu",
                                 single_support_s=0.8, double_support_s=0.2)

    def test_alternates_without_target_acquisition(self):
        clock = self.create()
        assigned = torch.tensor([0, 0])
        valid = torch.tensor([True, True])

        expected = [
            (0.0, [[0., 0.], [0., 0.]]),
            (0.3, [[1., 0.], [1., 0.]]),
            (1.1, [[0., 0.], [0., 0.]]),
            (1.5, [[0., 1.], [0., 1.]]),
        ]

        for time, wanted in expected:
            clock.elapsed_s[:] = time
            _, swing, _ = clock.features(
                target_foot=assigned, valid=valid
            )
            self.assertEqual(swing.tolist(), wanted)

    def test_acquisition_does_not_restart_gait(self):
        clock = self.create()
        clock.update(dt=0.3)
        clock.update(dt=0.1, acquired=torch.tensor([True, False]))
        self.assertAlmostEqual(clock.elapsed_s[0].item(), 0.4, places=6)
        self.assertAlmostEqual(clock.elapsed_s[1].item(), 0.4, places=6)

    def test_clock_is_independent_of_assigned_foot(self):
        clock = self.create()
        phase, _, _ = clock.features(
            target_foot=torch.tensor([0, 1]),
            valid=torch.tensor([True, True]),
        )
        self.assertEqual(phase[0].tolist(), phase[1].tolist())

    def test_expiration_never_claims_acquisition(self):
        clock = self.create()
        clock.update(dt=2.25)
        self.assertAlmostEqual(clock.elapsed_s[0].item(), .25, places=6)

    def test_inactive_zero_and_partial_reset(self):
        clock = self.create()
        clock.update(dt=0.3)
        clock.reset(torch.tensor([0]))
        self.assertEqual(clock.elapsed_s.tolist()[0], 0.)
        self.assertGreater(clock.elapsed_s[1].item(), 0.)
        obs, swing, stance = clock.features(target_foot=torch.tensor([0, 1]),
                                             valid=torch.tensor([True, False]))
        self.assertEqual(obs[1].tolist(), [0., 0.])
        self.assertEqual(swing[1].tolist(), [0., 0.])
        # Plan complete: stand mode requests both feet in stance.
        self.assertEqual(stance[1].tolist(), [1., 1.])

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            FootstepGaitClock(num_envs=1, device="cpu",
                              single_support_s=0., double_support_s=.2)
        clock = self.create()
        with self.assertRaises(ValueError):
            clock.update(dt=0)
        with self.assertRaises(ValueError):
            clock.features(target_foot=torch.tensor([3, 0]),
                           valid=torch.tensor([True, True]))


if __name__ == "__main__":
    unittest.main()
