import math
import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_conditioning import encode_next_two, OBS_WIDTH
from simulation.isaac_lab.g1_stairs.footstep_tensor import planned_target_batch


class CompactFootstepConditioningTests(unittest.TestCase):
    def inputs(self):
        targets, feet, valid = planned_target_batch(
            first_riser_x=torch.tensor([1., 1., 1.]),
            tread_depth=torch.tensor([.25, .25, .25]),
            riser_height=torch.tensor([.2, .2, .2]), step_count=3,
            first_foot=torch.tensor([0, 1, 0]),
            target_index=torch.tensor([0, 4, 5]),
            centre_y=torch.zeros(3), foot_separation=torch.full((3,), .16))
        return targets, feet, valid

    def test_shape_identity_and_finished_slots(self):
        t, f, v = self.inputs()
        out = encode_next_two(targets=t, feet=f, valid=v,
                              root_xyz=torch.zeros(3, 3), root_yaw=torch.zeros(3),
                              position_scale=1.)
        self.assertEqual(out.shape, (3, OBS_WIDTH))
        self.assertEqual(out[0, :3].tolist(), [1., 1., 0.])
        self.assertEqual(out[0, 7:10].tolist(), [1., 0., 1.])
        self.assertTrue(torch.all(out[1, 7:] == 0))
        self.assertTrue(torch.all(out[2] == 0))

    def test_relative_coordinates_with_rotated_root(self):
        t = torch.tensor([[[1., 2., .2, 0.], [2., 2., .4, 0.]]])
        out = encode_next_two(targets=t, feet=torch.tensor([[0, 1]]),
                              valid=torch.ones(1, 2, dtype=torch.bool),
                              root_xyz=torch.tensor([[1., 1., .1]]),
                              root_yaw=torch.tensor([math.pi/2]), position_scale=1.)
        self.assertAlmostEqual(out[0, 3].item(), 1., places=6)
        self.assertAlmostEqual(out[0, 4].item(), 0., places=6)
        self.assertAlmostEqual(out[0, 5].item(), .1, places=6)
        self.assertAlmostEqual(out[0, 6].item(), -.5, places=6)

    def test_reject_invalid_foot_and_scale(self):
        t, f, v = self.inputs()
        f[0, 0] = 7
        args = dict(targets=t, feet=f, valid=v, root_xyz=torch.zeros(3, 3),
                    root_yaw=torch.zeros(3), position_scale=1.)
        with self.assertRaises(ValueError):
            encode_next_two(**args)
        f[0, 0] = 0
        with self.assertRaises(ValueError):
            encode_next_two(**(args | {"position_scale": 0.}))

    def test_independent_per_environment_targets(self):
        t, f, v = self.inputs()
        out = encode_next_two(targets=t, feet=f, valid=v,
                              root_xyz=torch.zeros(3, 3), root_yaw=torch.zeros(3),
                              position_scale=1.)
        self.assertNotEqual(out[0, 3].item(), out[1, 3].item())
        self.assertEqual(out[1, 0].item(), 1.)
        self.assertEqual(out[1, 2].item(), 1.)


if __name__ == '__main__':
    unittest.main()
