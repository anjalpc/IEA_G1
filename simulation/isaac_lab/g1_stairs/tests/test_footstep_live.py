"""Regression: footstep observation is read-only and resets remain isolated."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from simulation.isaac_lab.g1_stairs import footstep_live


class LiveFootstepsTests(unittest.TestCase):
    def make_env(self):
        return SimpleNamespace(num_envs=2, device=torch.device("cpu"))

    @patch.object(footstep_live, "stair_cfg", return_value=SimpleNamespace(step_count=3))
    def test_lazy_runtime_and_reset_isolation(self, _):
        env = self.make_env()
        runtime = footstep_live.ensure_runtime(env)
        self.assertIs(runtime, footstep_live.ensure_runtime(env))
        runtime.target_index[:] = torch.tensor([2, 1])
        footstep_live.reset_footsteps(env, torch.tensor([0]))
        self.assertEqual(runtime.target_index.tolist(), [0, 1])

    @patch.object(footstep_live, "target_observation")
    @patch.object(footstep_live, "stair_cfg", return_value=SimpleNamespace(step_count=3))
    def test_observation_does_not_update_progress(self, _, observation):
        env = self.make_env()
        runtime = footstep_live.ensure_runtime(env)
        runtime.target_index[:] = torch.tensor([1, 2])
        observation.return_value = torch.zeros(2, 14)
        result = footstep_live.planned_footstep_targets(env)
        self.assertEqual(tuple(result.shape), (2, 14))
        self.assertEqual(runtime.target_index.tolist(), [1, 2])
        observation.assert_called_once()

    @patch.object(footstep_live, "stair_cfg", return_value=SimpleNamespace(step_count=4))
    def test_geometry_mismatch_fails_closed(self, _):
        env = self.make_env()
        env._g1_footstep_runtime = footstep_live.FootstepRuntime(
            num_envs=2, device="cpu", step_count=3,
            first_foot=torch.zeros(2, dtype=torch.long))
        with self.assertRaises(RuntimeError):
            footstep_live.ensure_runtime(env)


if __name__ == "__main__":
    unittest.main()
