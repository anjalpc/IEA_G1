import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_runtime import FootstepRuntime

class FootstepRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FootstepRuntime(num_envs=2, device='cpu', step_count=3,
                                       first_foot=torch.tensor([0, 1]))
        self.geo = dict(first_riser_x=torch.tensor([1., 1.]),
                        tread_depth=torch.tensor([.25, .25]),
                        riser_height=torch.tensor([.2, .2]),
                        centre_y=torch.zeros(2), foot_separation=torch.full((2,), .16))

    def inputs(self):
        t, f, v = self.runtime.targets(**self.geo)
        sole = torch.zeros(2, 2, 3)
        sole[torch.arange(2), f[:, 0]] = t[:, 0, :3]
        return t, f, v, sole

    def update(self, supported):
        t, f, v, sole = self.inputs()
        return self.runtime.update(targets=t, feet=f, valid=v,
                                   sole_xyz=sole, supported=supported,
                                   xy_tolerance=.04, z_tolerance=.03)

    def test_landing_advances_only_assigned_foot(self):
        self.assertEqual(self.update(torch.tensor([[False, True], [True, False]])).tolist(), [False, False])
        self.assertEqual(self.update(torch.tensor([[True, True], [True, True]])).tolist(), [True, True])
        self.assertEqual(self.runtime.target_index.tolist(), [1, 1])

    def test_supported_hold_does_not_retrigger(self):
        self.update(torch.tensor([[True, False], [False, True]]))
        self.assertEqual(self.update(torch.tensor([[True, False], [False, True]])).tolist(), [False, False])
        self.assertEqual(self.runtime.completed.tolist(), [1, 1])

    def test_distance_rejects_false_landing(self):
        t, f, v, sole = self.inputs()
        sole[:, :, 0] += .4
        result = self.runtime.update(targets=t, feet=f, valid=v,
                                     sole_xyz=sole, supported=torch.tensor([[True, False], [False, True]]),
                                     xy_tolerance=.04, z_tolerance=.03)
        self.assertEqual(result.tolist(), [False, False])
        self.assertEqual(self.runtime.target_index.tolist(), [0, 0])

    def test_partial_reset_isolated(self):
        self.update(torch.tensor([[True, False], [False, True]]))
        self.runtime.reset(torch.tensor([0]))
        self.assertEqual(self.runtime.target_index.tolist(), [0, 1])
        self.assertEqual(self.runtime.was_supported.tolist(), [[False, False], [False, True]])

    def test_observation_shape(self):
        t, f, v, sole = self.inputs()
        out = self.runtime.observation(targets=t, feet=f, valid=v,
                                       root_xyz=torch.zeros(2, 3), root_yaw=torch.zeros(2), position_scale=1.)
        self.assertEqual(out.shape, (2, 14))

if __name__ == '__main__':
    unittest.main()
