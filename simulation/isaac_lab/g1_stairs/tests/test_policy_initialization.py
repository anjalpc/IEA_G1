from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from simulation.isaac_lab.g1_stairs.policy_initialization import (
    LOCOMOTION_OBS_DIM,
    TERRAIN_OBS_DIM,
    initialize_actor_from_rsl_rl_checkpoint,
    initialize_actor_from_unitree_onnx,
)


def _make_actor() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(667, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 29),
    )


def _make_critic() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(682, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 1),
    )


def _make_fake_onnx(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)

    tensors = {
        "actor.0.weight": rng.normal(size=(512, 480)).astype(np.float32),
        "actor.0.bias": rng.normal(size=(512,)).astype(np.float32),
        "actor.2.weight": rng.normal(size=(256, 512)).astype(np.float32),
        "actor.2.bias": rng.normal(size=(256,)).astype(np.float32),
        "actor.4.weight": rng.normal(size=(128, 256)).astype(np.float32),
        "actor.4.bias": rng.normal(size=(128,)).astype(np.float32),
        "actor.6.weight": rng.normal(size=(29, 128)).astype(np.float32),
        "actor.6.bias": rng.normal(size=(29,)).astype(np.float32),
    }

    initializers = [
        numpy_helper.from_array(value, name=name)
        for name, value in tensors.items()
    ]

    graph = helper.make_graph(
        nodes=[],
        name="test_actor",
        inputs=[],
        outputs=[],
        initializer=initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, str(path))

    return tensors


class PolicyInitializationTest(unittest.TestCase):
    def test_unitree_onnx_transfer(self) -> None:
        actor = _make_actor()
        critic = _make_critic()

        critic_before = {
            key: value.detach().clone()
            for key, value in critic.state_dict().items()
        }

        policy = SimpleNamespace(
            actor=actor,
            critic=critic,
        )

        runner = SimpleNamespace(
            alg=SimpleNamespace(
                policy=policy,
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "policy.onnx"
            source = _make_fake_onnx(model_path)

            metadata = initialize_actor_from_unitree_onnx(
                runner=runner,
                onnx_path=model_path,
            )

        np.testing.assert_allclose(
            actor[0].weight[:, :LOCOMOTION_OBS_DIM].detach().cpu().numpy(),
            source["actor.0.weight"],
        )

        self.assertEqual(
            float(
                actor[0]
                .weight[:, LOCOMOTION_OBS_DIM:]
                .detach()
                .abs()
                .max()
            ),
            0.0,
        )

        self.assertEqual(
            actor[0].weight.shape[1],
            LOCOMOTION_OBS_DIM + TERRAIN_OBS_DIM,
        )

        for index in (2, 4, 6):
            np.testing.assert_allclose(
                actor[index].weight.detach().cpu().numpy(),
                source[f"actor.{index}.weight"],
            )
            np.testing.assert_allclose(
                actor[index].bias.detach().cpu().numpy(),
                source[f"actor.{index}.bias"],
            )

        for key, before in critic_before.items():
            torch.testing.assert_close(
                critic.state_dict()[key],
                before,
            )

        self.assertEqual(
            metadata["type"],
            "unitree_velocity_onnx_transfer",
        )
        self.assertEqual(
            metadata["new_columns_max_abs"],
            0.0,
        )

    def test_rsl_rl_checkpoint_actor_transfer_preserves_critic(self) -> None:
        class FakePolicy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.actor = torch.nn.Sequential(
                    torch.nn.Linear(98, 512),
                    torch.nn.ELU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 29),
                )
                self.critic = torch.nn.Sequential(
                    torch.nn.Linear(288, 512),
                    torch.nn.ELU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 1),
                )
                self.std = torch.nn.Parameter(torch.full((29,), 0.5))

        source_policy = FakePolicy()
        target_policy = FakePolicy()

        with torch.no_grad():
            for parameter in source_policy.actor.parameters():
                parameter.add_(1.0)
            source_policy.std.fill_(0.23)

        critic_before = {
            key: value.detach().clone()
            for key, value in target_policy.critic.state_dict().items()
        }

        runner = SimpleNamespace(
            alg=SimpleNamespace(policy=target_policy)
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model_950.pt"
            torch.save(
                {
                    "model_state_dict": source_policy.state_dict(),
                    "optimizer_state_dict": {"state": {}, "param_groups": []},
                    "iter": 950,
                    "infos": None,
                },
                checkpoint,
            )

            metadata = initialize_actor_from_rsl_rl_checkpoint(
                runner=runner,
                checkpoint_path=checkpoint,
            )

        target_state = target_policy.state_dict()
        source_state = source_policy.state_dict()

        for key in source_state:
            if key == "std" or key.startswith("actor."):
                torch.testing.assert_close(
                    target_state[key],
                    source_state[key],
                )

        for key, before in critic_before.items():
            torch.testing.assert_close(
                target_policy.critic.state_dict()[key],
                before,
            )

        self.assertEqual(metadata["source_iteration"], 950)
        self.assertEqual(metadata["critic_initialization"], "fresh")
        self.assertEqual(metadata["optimizer_initialization"], "fresh")


class ZeroResidualActorTests(unittest.TestCase):
    def test_zero_residual_initializer_zeros_only_actor_output_layer(self):
        from types import SimpleNamespace
        from simulation.isaac_lab.g1_stairs.policy_initialization import (
            initialize_zero_residual_actor,
        )

        actor = torch.nn.Sequential(
            torch.nn.Linear(98, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 29),
        )
        critic = torch.nn.Sequential(torch.nn.Linear(288, 1))
        hidden_before = actor[0].weight.detach().clone()
        critic_before = critic[0].weight.detach().clone()
        policy = SimpleNamespace(actor=actor, critic=critic)
        runner = SimpleNamespace(alg=SimpleNamespace(policy=policy))

        meta = initialize_zero_residual_actor(runner)

        self.assertTrue(torch.equal(actor[0].weight, hidden_before))
        self.assertTrue(torch.equal(critic[0].weight, critic_before))
        self.assertTrue(torch.count_nonzero(actor[2].weight) == 0)
        self.assertTrue(torch.count_nonzero(actor[2].bias) == 0)
        self.assertEqual(meta["type"], "zero_residual_actor")


if __name__ == "__main__":
    unittest.main()
