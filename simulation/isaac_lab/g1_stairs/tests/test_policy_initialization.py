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


if __name__ == "__main__":
    unittest.main()
