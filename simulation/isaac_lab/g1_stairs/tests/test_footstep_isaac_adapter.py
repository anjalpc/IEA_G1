"""Adapter tests use a mocked Isaac scene; no simulation claims."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch

from simulation.isaac_lab.g1_stairs.footstep_runtime import FootstepRuntime
from simulation.isaac_lab.g1_stairs.footstep_isaac_adapter import (
    course_root_pose, course_tensors, target_observation,
)


class FakeScene:
    def __init__(self):
        self.env_origins = torch.tensor([[10., 0., 0.], [20., 1., 0.]])
        self.robot = NS(data=NS(
            root_pos_w=torch.tensor([[10., 0., 0.], [20., 1., 0.]]),
            root_quat_w=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        ))
    def __getitem__(self, key):
        assert key == 'robot'
        return self.robot


class IsaacAdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = NS(scene=FakeScene(), num_envs=2)
        self.cfg = NS(step_count=3, tread_depth=.25, start_clearance=1.)
        self.runtime = FootstepRuntime(num_envs=2, device='cpu', step_count=3,
                                       first_foot=torch.tensor([0, 1]))

    def patched(self):
        return patch.multiple(
            'simulation.isaac_lab.g1_stairs.footstep_isaac_adapter',
            stair_cfg=lambda env: self.cfg,
            riser_height_per_env=lambda env: torch.tensor([.2, .1]),
        )

    def test_geometry_uses_resolved_env_height(self):
        with self.patched():
            args = course_tensors(self.env, foot_separation_m=.16)
        torch.testing.assert_close(args['riser_height'], torch.tensor([.2, .1]))
        torch.testing.assert_close(args['first_riser_x'], torch.ones(2))

    def test_course_frame_subtracts_each_env_origin(self):
        xyz, yaw = course_root_pose(self.env)
        torch.testing.assert_close(xyz, torch.zeros(2, 3))
        torch.testing.assert_close(yaw, torch.zeros(2))

    def test_read_only_target_observation(self):
        with self.patched():
            obs = target_observation(self.env, self.runtime,
                                     position_scale_m=1., foot_separation_m=.16)
        self.assertEqual(tuple(obs.shape), (2, 14))
        self.assertAlmostEqual(obs[0, 3].item(), 1.125)
        self.assertAlmostEqual(obs[1, 5].item(), .1)
        self.assertEqual(self.runtime.target_index.tolist(), [0, 0])

    def test_mismatched_stair_count_fails(self):
        self.cfg.step_count = 4
        with self.patched(), self.assertRaises(ValueError):
            target_observation(self.env, self.runtime,
                               position_scale_m=1., foot_separation_m=.16)


if __name__ == '__main__':
    unittest.main()
