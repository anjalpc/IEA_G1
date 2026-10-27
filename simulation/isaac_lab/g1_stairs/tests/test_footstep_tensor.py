import math
import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_tensor import (
    planned_target_batch, footstep_observation, footstep_scores, OBS_WIDTH)
from simulation.isaac_lab.g1_stairs.footstep_targets import StairCourse, FootstepProgress, planned_targets
from simulation.isaac_lab.g1_stairs.footstep_learning import (
    FootstepLearningConfig, observation, learning_scores)


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
            phase=torch.tensor([0., .25]), position_scale=1., step_count=3,
            target_index=torch.tensor([0, 4]))
        self.assertEqual(obs.shape, (2, OBS_WIDTH))
        self.assertTrue(torch.all(obs[1, 12:] == 0))
        self.assertAlmostEqual(obs[0, 5].item(), 1.125)

    def test_invalid_shapes_rejected(self):
        t, f, v = self.build()
        with self.assertRaises(ValueError):
            footstep_observation(targets=t, feet=f, valid=v,
                root_xyz=torch.zeros(2, 2), root_yaw=torch.zeros(2),
                phase=torch.zeros(2), position_scale=1., step_count=3,
                target_index=torch.tensor([0, 4]))

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

    def test_observation_matches_scalar_contract_for_multiple_states(self):
        course = StairCourse(.8, .3, .1, 3, centre_y=.02, foot_separation=.20)
        cfg = FootstepLearningConfig(.7, .5, .2, 100., .4)
        indexes = [0, 1, 3, 4, 5]
        first_feet = [0, 1, 0, 1, 0]
        batch = planned_target_batch(
            first_riser_x=torch.full((5,), .8, dtype=torch.float64),
            tread_depth=torch.full((5,), .3, dtype=torch.float64),
            riser_height=torch.full((5,), .1, dtype=torch.float64),
            step_count=3, first_foot=torch.tensor(first_feet),
            target_index=torch.tensor(indexes),
            centre_y=torch.full((5,), .02, dtype=torch.float64),
            foot_separation=torch.full((5,), .2, dtype=torch.float64))
        xyz = torch.tensor([[.2, .01, .8], [.4, -.1, .83], [.7, .2, .91],
                            [1., .0, 1.], [1.3, .0, 1.1]], dtype=torch.float64)
        yaw = torch.tensor([0., math.pi/2, -.4, math.pi, .3], dtype=torch.float64)
        phase = torch.tensor([0., .25, .5, .75, .9], dtype=torch.float64)
        actual = footstep_observation(targets=batch[0], feet=batch[1], valid=batch[2],
            root_xyz=xyz, root_yaw=yaw, phase=phase, position_scale=.7,
            step_count=3, target_index=torch.tensor(indexes))
        for i in range(5):
            progress = FootstepProgress(planned_targets(course, first_feet[i]), next_index=indexes[i])
            expected = observation(progress, root_x=xyz[i, 0].item(),
                root_y=xyz[i, 1].item(), root_z=xyz[i, 2].item(),
                root_yaw=yaw[i].item(), phase_fraction=phase[i].item(), cfg=cfg)
            torch.testing.assert_close(actual[i], torch.tensor(expected, dtype=actual.dtype), atol=1e-12, rtol=1e-12)

    def test_scores_match_scalar_contract(self):
        cfg = FootstepLearningConfig(.7, .5, .2, 100., .4)
        x = torch.tensor([.01, .4, .7], dtype=torch.float64)
        mid = torch.tensor([.3, .1, .8], dtype=torch.float64)
        velocity = torch.tensor([.1, -.2, .3], dtype=torch.float64)
        commanded = torch.tensor([.2, .1, .2], dtype=torch.float64)
        force = torch.tensor([0., 20., 140.], dtype=torch.float64)
        support = torch.tensor([120., 80., 15.], dtype=torch.float64)
        speed = torch.tensor([.4, .1, 1.2], dtype=torch.float64)
        phase = torch.tensor([1., .3, 0.], dtype=torch.float64)
        actual = footstep_scores(foot_distance=x, midpoint_distance=mid,
            measured_velocity=velocity, commanded_velocity=commanded,
            swing_force=force, support_force=support, swing_speed=speed,
            swing_phase=phase, distance_scale=.2, velocity_scale=.5,
            force_scale=100., speed_scale=.4)
        for i in range(3):
            expected = learning_scores(assigned_foot_distance_m=x[i].item(),
                root_midpoint_distance_m=mid[i].item(), measured_forward_mps=velocity[i].item(),
                commanded_forward_mps=commanded[i].item(), swing_foot_force_n=force[i].item(),
                support_foot_force_n=support[i].item(), swing_foot_speed_mps=speed[i].item(),
                phase_swing_fraction=phase[i].item(), cfg=cfg)
            for key, value in expected.items():
                self.assertAlmostEqual(actual[key][i].item(), value, places=12)

    def test_invalid_phase_and_negative_measurements_rejected(self):
        t, f, v = self.build()
        kwargs = dict(targets=t, feet=f, valid=v, root_xyz=torch.zeros(2, 3),
            root_yaw=torch.zeros(2), phase=torch.tensor([0., 1.]),
            position_scale=1., step_count=3, target_index=torch.tensor([0, 4]))
        with self.assertRaises(ValueError):
            footstep_observation(**kwargs)
        kwargs['phase'] = torch.tensor([0., 0.])
        kwargs['step_count'] = 0
        with self.assertRaises(ValueError):
            footstep_observation(**kwargs)
        score_kwargs = dict(foot_distance=torch.tensor([-.01]), midpoint_distance=torch.tensor([0.]),
            measured_velocity=torch.tensor([0.]), commanded_velocity=torch.tensor([0.]),
            swing_force=torch.tensor([0.]), support_force=torch.tensor([100.]),
            swing_speed=torch.tensor([.4]), swing_phase=torch.tensor([1.]),
            distance_scale=.2, velocity_scale=.5, force_scale=100., speed_scale=.4)
        with self.assertRaises(ValueError):
            footstep_scores(**score_kwargs)
        score_kwargs['foot_distance'] = torch.tensor([.01])
        score_kwargs['swing_phase'] = torch.tensor([1.1])
        with self.assertRaises(ValueError):
            footstep_scores(**score_kwargs)

if __name__ == '__main__':
    unittest.main()
