import unittest
from simulation.isaac_lab.g1_stairs.footstep_targets import StairCourse, FootstepProgress, planned_targets
from simulation.isaac_lab.g1_stairs.footstep_learning import FootstepLearningConfig, observation, learning_scores


class FootstepLearningTests(unittest.TestCase):
    def setUp(self):
        self.cfg = FootstepLearningConfig(1., .5, .2, 100., .4)
        self.progress = FootstepProgress(planned_targets(StairCourse(1., .25, .2, 3), 0))

    def test_observation_shape_and_next_foot(self):
        obs = observation(self.progress, root_x=0., root_y=0., root_yaw=0.,
                          root_z=0., phase_fraction=0., cfg=self.cfg)
        self.assertEqual(len(obs), 22)
        self.assertEqual(obs[2:5], (1., 1., 0.))
        self.assertEqual(obs[12:15], (1., 0., 1.))
        self.assertEqual(obs[5], 1.125)

    def test_progress_updates_target_without_hidden_sequence_gate(self):
        self.assertTrue(self.progress.touchdown(foot=0, level=1, within_target=True, supported=True))
        obs = observation(self.progress, root_x=0., root_y=0., root_yaw=0.,
                          root_z=0., phase_fraction=.5, cfg=self.cfg)
        self.assertEqual(obs[2:5], (1., 0., 1.))

    def test_finished_plan_zero_fills_targets(self):
        self.progress.next_index = len(self.progress.targets)
        obs = observation(self.progress, root_x=0., root_y=0., root_yaw=0.,
                          root_z=0., phase_fraction=0., cfg=self.cfg)
        self.assertEqual(obs[2:], (0.,)*20)

    def test_target_score_is_monotone_and_phase_senses_loading(self):
        kwargs = dict(root_midpoint_distance_m=.2, measured_forward_mps=.2,
                      commanded_forward_mps=.3, swing_foot_force_n=0.,
                      support_foot_force_n=100., swing_foot_speed_mps=.4,
                      phase_swing_fraction=1., cfg=self.cfg)
        near = learning_scores(assigned_foot_distance_m=.05, **kwargs)
        far = learning_scores(assigned_foot_distance_m=.3, **kwargs)
        self.assertGreater(near['footstep_target'], far['footstep_target'])
        self.assertGreater(near['footstep_force_phase'], .9)
        self.assertGreater(near['footstep_velocity_phase'], .9)

    def test_invalid_scaling_or_phase_fails(self):
        with self.assertRaises(ValueError):
            FootstepLearningConfig(0., .5, .2, 100., .4)
        with self.assertRaises(ValueError):
            observation(self.progress, root_x=0., root_y=0., root_yaw=0.,
                        root_z=0., phase_fraction=1., cfg=self.cfg)

if __name__ == '__main__':
    unittest.main()
