"""Pure actor-expansion checks for C7 preparation."""

import unittest

import torch

from simulation.isaac_lab.g1_stairs.scripts.training.prepare_c7 import (
    ACTIONS,
    SOURCE_OBS,
    TARGET_OBS,
    expand_actor_observations,
    matched_output_check,
)


class PrepareC7Tests(unittest.TestCase):
    def state(self):
        return {
            "actor.0.weight": 0.02 * torch.randn(512, SOURCE_OBS),
            "actor.0.bias": 0.02 * torch.randn(512),
            "actor.2.weight": 0.02 * torch.randn(256, 512),
            "actor.2.bias": 0.02 * torch.randn(256),
            "actor.4.weight": 0.02 * torch.randn(128, 256),
            "actor.4.bias": 0.02 * torch.randn(128),
            "actor.6.weight": 0.02 * torch.randn(ACTIONS, 128),
            "actor.6.bias": 0.02 * torch.randn(ACTIONS),
            "std": torch.full((ACTIONS,), 0.2),
        }

    def test_expansion_preserves_actor_exactly(self):
        state = self.state()
        expanded, keys = expand_actor_observations(state)
        self.assertEqual(expanded["actor.0.weight"].shape, (512, TARGET_OBS))
        self.assertEqual(
            torch.count_nonzero(expanded["actor.0.weight"][:, SOURCE_OBS:]).item(),
            0,
        )
        self.assertLessEqual(matched_output_check(state, expanded, keys), 1e-5)

    def test_wrong_source_dimension_is_rejected(self):
        state = self.state()
        state["actor.0.weight"] = 0.02 * torch.randn(512, SOURCE_OBS - 1)
        with self.assertRaises(ValueError):
            expand_actor_observations(state)


    def test_c7_resolved_reward_contract(self):
        import yaml
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from simulation.isaac_lab.g1_stairs.scripts.training.prepare_c7 import (
            leaf_config,
            validate_leaf,
        )

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            for smoke in (True, False):
                path = tmp / (
                    "c7_smoke.yaml" if smoke else "c7_train.yaml"
                )
                cfg = leaf_config(
                    name=path.stem,
                    output_hash="0" * 64,
                    smoke=smoke,
                )

                # validate_leaf expects a repo-relative inheritance path,
                # so materialize the generated config under the repo tree.
                repo_path = Path(
                    "experiments/g1_stairs/configs/"
                    + ("_test_c7_smoke.yaml" if smoke else "_test_c7_train.yaml")
                )
                try:
                    repo_path.write_text(
                        yaml.safe_dump(cfg, sort_keys=False)
                    )
                    validate_leaf(repo_path, smoke=smoke)
                finally:
                    repo_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
