import unittest
from types import SimpleNamespace

import torch

from simulation.isaac_lab.g1_stairs.geometry_locomotion_rewards import (
    actual_gait_scores,
    forward_stall,
)


class ForwardStallTests(unittest.TestCase):
    def make_env(self, speed, command):
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_lin_vel_w=torch.tensor([[speed, 0.0, 0.0]])
            )
        )
        return SimpleNamespace(
            scene={"robot": robot},
            command_manager=SimpleNamespace(
                get_command=lambda name: torch.tensor(
                    [[command, 0.0, 0.0]]
                )
            ),
        )

    def test_stopped_robot_receives_full_cost(self):
        env = self.make_env(0.0, 0.35)
        value = forward_stall(
            env,
            asset_cfg=SimpleNamespace(name="robot"),
        )
        self.assertAlmostEqual(value.item(), 1.0)

    def test_half_speed_receives_half_cost(self):
        env = self.make_env(0.05, 0.35)
        value = forward_stall(
            env,
            asset_cfg=SimpleNamespace(name="robot"),
        )
        self.assertAlmostEqual(value.item(), 0.5)

    def test_no_forward_command_receives_no_cost(self):
        env = self.make_env(0.0, 0.0)
        value = forward_stall(
            env,
            asset_cfg=SimpleNamespace(name="robot"),
        )
        self.assertEqual(value.item(), 0.0)


class ActualGaitScoreTests(unittest.TestCase):
    def score(self, foot_x, force, speed=0.5):
        return actual_gait_scores(
            torch.tensor([foot_x]),
            torch.tensor([force]),
            torch.tensor([speed]),
            minimum_stride_m=0.10,
            target_stride_m=0.20,
            maximum_stride_m=0.30,
            support_force_n=80.0,
            unload_force_n=20.0,
            full_speed_mps=0.35,
        )

    def test_target_stride_with_forward_support_scores_one(self):
        stride, support, lead = self.score([0.20, 0.0], [100.0, 0.0])
        self.assertAlmostEqual(stride.item(), 1.0)
        self.assertAlmostEqual(support.item(), 1.0)
        self.assertEqual(lead.item(), 0)

    def test_static_double_support_cannot_farm_credit(self):
        stride, support, lead = self.score(
            [0.20, 0.0], [100.0, 100.0], speed=0.0
        )
        self.assertEqual(stride.item(), 0.0)
        self.assertEqual(support.item(), 0.0)
        self.assertEqual(lead.item(), -1)

    def test_oversized_stride_scores_zero(self):
        stride, support, lead = self.score([0.31, 0.0], [100.0, 0.0])
        self.assertEqual(stride.item(), 0.0)
        self.assertEqual(support.item(), 0.0)
        self.assertEqual(lead.item(), -1)


if __name__ == "__main__":
    unittest.main()


class AlbertSingleStepConfigTests(unittest.TestCase):
    def test_resolved_config_matches_training_and_evaluation_contracts(self):
        import tempfile
        import yaml
        from pathlib import Path

        from simulation.isaac_lab.g1_stairs.reward_runtime import (
            active_reward_weights,
            load_resolved_config,
        )
        from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
            resolve_config,
        )

        source = Path(
            "experiments/g1_stairs/configs/albert_single04_smoke.yaml"
        )
        resolved, _ = resolve_config(source)

        stage = resolved["reward_experiment"]["active_stage"]
        self.assertTrue(stage.startswith("geometry_locomotion"))
        self.assertFalse(resolved["episode_contract"]["enabled"])
        self.assertEqual(
            resolved["policy_initialization"]["stair_observations"], 485
        )
        self.assertEqual(resolved["policy_initialization"]["actions"], 29)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.yaml"
            path.write_text(yaml.safe_dump(resolved, sort_keys=False))
            runtime = load_resolved_config(path)
            weights = active_reward_weights(runtime)

        self.assertEqual(len(weights), 17)
        self.assertEqual(weights["forward_stall"], -1.0)
