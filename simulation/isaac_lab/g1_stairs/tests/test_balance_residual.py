from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from simulation.isaac_lab.g1_stairs.balance_residual import (
    compose_balance_residual_action,
    load_frozen_rsl_actor,
    residual_scale_vector,
)
from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS


class BalanceResidualTests(unittest.TestCase):
    def test_scale_map_requires_exact_joint_contract(self):
        mapping = {name: 0.1 for name in BODY_JOINTS}
        scale = residual_scale_vector(mapping, device="cpu")
        self.assertEqual(tuple(scale.shape), (29,))
        self.assertTrue(torch.allclose(scale, torch.full((29,), 0.1)))

        broken = dict(mapping)
        broken.pop(BODY_JOINTS[0])
        with self.assertRaises(ValueError):
            residual_scale_vector(broken, device="cpu")

    def test_zero_residual_exactly_preserves_teacher(self):
        teacher = torch.linspace(-0.7, 0.7, 58).reshape(2, 29)
        residual = torch.zeros_like(teacher)
        scale = torch.ones(29)
        executed = compose_balance_residual_action(
            teacher,
            residual,
            residual_scale=scale,
            residual_clip=1.0,
            executed_clip=5.0,
        )
        self.assertTrue(torch.equal(executed, teacher))

    def test_residual_is_clipped_before_joint_scaling(self):
        teacher = torch.zeros((1, 29))
        residual = torch.full((1, 29), 10.0)
        scale = torch.linspace(0.0, 1.4, 29)
        executed = compose_balance_residual_action(
            teacher,
            residual,
            residual_scale=scale,
            residual_clip=1.0,
            executed_clip=5.0,
        )
        self.assertTrue(torch.allclose(executed[0], scale))

    def test_executed_action_has_final_safety_clip(self):
        teacher = torch.full((1, 29), 4.9)
        residual = torch.ones((1, 29))
        scale = torch.ones(29)
        executed = compose_balance_residual_action(
            teacher,
            residual,
            residual_scale=scale,
            residual_clip=1.0,
            executed_clip=5.0,
        )
        self.assertEqual(float(executed.max()), 5.0)

    def test_load_frozen_actor_reconstructs_checkpoint_network(self):
        actor = torch.nn.Sequential(
            torch.nn.Linear(98, 16),
            torch.nn.ELU(),
            torch.nn.Linear(16, 29),
        )
        state = {
            f"actor.{key}": value.detach().clone()
            for key, value in actor.state_dict().items()
        }
        state["critic.0.weight"] = torch.zeros((8, 8))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"model_state_dict": state, "iter": 1500}, path)
            frozen, metadata = load_frozen_rsl_actor(path, device="cpu")

            x = torch.randn((4, 98))
            with torch.no_grad():
                expected = actor(x)
                actual = frozen(x)

            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            self.assertEqual(metadata["observation_dim"], 98)
            self.assertEqual(metadata["action_dim"], 29)
            self.assertEqual(metadata["source_iteration"], 1500)
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))


if __name__ == "__main__":
    unittest.main()


class BalanceResidualConfigTests(unittest.TestCase):
    def test_v01_config_resolves_teacher_and_zero_residual_actor(self):
        from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
            REPO_ROOT,
            resolve_config,
        )

        resolved, _ = resolve_config(
            REPO_ROOT
            / "experiments/g1_stairs/configs/g1_stair_v10.yaml"
        )
        self.assertEqual(
            resolved["policy_initialization"]["type"],
            "zero_residual_actor",
        )
        cfg = resolved["balance_residual_control"]
        self.assertTrue(cfg["enabled"])
        self.assertTrue(cfg["teacher_checkpoint"].endswith("model_1500.pt"))
        self.assertEqual(set(cfg["scale_by_joint"]), set(BODY_JOINTS))
        self.assertEqual(resolved["terrain"]["riser_height_m"], 0.06)
        self.assertEqual(resolved["terrain"]["step_count"], 1)
        self.assertEqual(
            resolved["observations"]["proprioception_history_steps"],
            1,
        )
