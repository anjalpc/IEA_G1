import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_tensor import (
    planned_target_batch, footstep_observation, footstep_scores, OBS_WIDTH)


class BatchedFootstepTests(unittest.TestCase):
    def build(self):
        return planned_target_batch(
            first_riser_x=torch.tensor([1., .8]),
            tread_depth=torch.tensor([.25, .3]),
            riser_height=torch.tensor([.20, .10]),
            step_count=3, first_foot=torch.tensor([0, 1]),
            target_index=torch.tensor([0, 4]),
            centre_y=torch.zeros(2), foot_separation=torch.tensor([.16, .20]))

    def test_per_environment_geometry_and_foot(self):
        target, foot, valid = self.build()
        self.assertEqual(target.shape, (2, 2, 4))
        self.assertAlmostEqual(target[0, 0, 0].item(), 1.125)
        self.assertAlmostEqual(target[1, 0, 2].item(), .3)
        self.assertEqual(foot[0, 0].item(), 0)
        self.assertEqual(foot[1, 0].item(), 1)
        self.assertTrue(valid[1, 0].item())
        self.assertFalse(valid[1, 1].item())

    def test_finished_slots_zero_and_shape(self):
        target, foot, valid = self.build()
        obs = footstep_observation(targets=target, feet=foot, valid=valid,
            root_xyz=torch.zeros((2,3)), root_yaw=torch.zeros(2),
            phase=torch.tensor([0., .25]), position_scale=1.)
        self.assertEqual(obs.shape, (2, OBS_WIDTH))
        self.assertTrue(torch.all(obs[1, 12:] == 0))
        self.assertAlmostEqual(obs[0, 5].item(), 1.125)

    def test_invalid_shapes_rejected(self):
        t, f, v = self.build()
        with self.assertRaises(ValueError):
            footstep_observation(targets=t, feet=f, valid=v,
                root_xyz=torch.zeros(2, 2), root_yaw=torch.zeros(2),
                phase=torch.zeros(2), position_scale=1.)

    def test_reward_vectorization_and_limits(self):
        x = torch.tensor([0., .4])
        z = torch.zeros(2)
        o = torch.ones(2)
        scores = footstep_scores(foot_distance=x, midpoint_distance=x,
            measured_velocity=z, commanded_velocity=z, swing_force=z,
            support_force=o*100, swing_speed=o, swing_phase=o,
            distance_scale=.2, velocity_scale=.5, force_scale=100., speed_scale=1.)
        self.assertEqual(set(scores), {"footstep_target", "footstep_root_progress",
           "footstep_velocity_tracking", "footstep_force_phase", "footstep_velocity_phase"})
        self.assertTrue(torch.all(scores['footstep_target'][0] > scores['footstep_target'][1]))
        for value in scores.values():
            self.assertTrue(torch.all((value >= 0) & (value <= 1)))

if __name__ == '__main__':
    unittest.main()
