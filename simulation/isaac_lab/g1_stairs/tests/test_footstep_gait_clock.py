"""Unit tests for paper-inspired, G1-unverified gait phase timing."""
import unittest

import torch

from simulation.isaac_lab.g1_stairs.footstep_gait_clock import FootstepGaitClock


class GaitClockTests(unittest.TestCase):
    def create(self):
        return FootstepGaitClock(num_envs=2, device="cpu",
                                 single_support_s=0.8, double_support_s=0.2)

    def test_double_support_and_assigned_swing(self):
        clock = self.create()
        foot = torch.tensor([0, 1])
        active = torch.tensor([True, True])
        _, swing, stance = clock.features(target_foot=foot, valid=active)
        self.assertEqual(swing.tolist(), [[0., 0.], [0., 0.]])
        self.assertEqual(stance.tolist(), [[1., 1.], [1., 1.]])
        clock.update(dt=0.2)
        _, swing, stance = clock.features(target_foot=foot, valid=active)
        self.assertEqual(swing.tolist(), [[1., 0.], [0., 1.]])
        self.assertEqual(stance.tolist(), [[0., 1.], [1., 0.]])

    def test_touchdown_resynchronizes_only_selected(self):
        clock = self.create()
        clock.update(dt=0.3)
        clock.update(dt=0.1, acquired=torch.tensor([True, False]))
        self.assertAlmostEqual(clock.elapsed_s[0].item(), 0., places=6)
        self.assertAlmostEqual(clock.elapsed_s[1].item(), .4, places=6)

    def test_clock_phase_flips_with_foot(self):
        clock = self.create()
        phase, _, _ = clock.features(target_foot=torch.tensor([0, 1]),
                                     valid=torch.tensor([True, True]))
        self.assertAlmostEqual(phase[0, 1].item(), 1., places=6)
        self.assertAlmostEqual(phase[1, 1].item(), -1., places=6)

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
        self.assertEqual(stance[1].tolist(), [0., 0.])

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
