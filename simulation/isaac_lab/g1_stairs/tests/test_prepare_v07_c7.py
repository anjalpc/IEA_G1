"""Pure checks for behavior-preserving V07 -> 492/30 actor expansion."""

import unittest
from pathlib import Path

import torch
import yaml

from simulation.isaac_lab.g1_stairs.scripts.training.prepare_v07_c7 import (
    FREQUENCY_STD,
    SOURCE_ACTIONS,
    SOURCE_OBS,
    TARGET_ACTIONS,
    TARGET_OBS,
    control_config,
    expand_v07_actor,
    matched_output_check,
    validate_control,
)


class PrepareV07C7Tests(unittest.TestCase):
    def state(self):
        return {
            "actor.0.weight": 0.02 * torch.randn(512, SOURCE_OBS),
            "actor.0.bias": 0.02 * torch.randn(512),
            "actor.2.weight": 0.02 * torch.randn(256, 512),
            "actor.2.bias": 0.02 * torch.randn(256),
            "actor.4.weight": 0.02 * torch.randn(128, 256),
            "actor.4.bias": 0.02 * torch.randn(128),
            "actor.6.weight": 0.02 * torch.randn(SOURCE_ACTIONS, 128),
            "actor.6.bias": 0.02 * torch.randn(SOURCE_ACTIONS),
            "std": torch.full((SOURCE_ACTIONS,), 0.2),
        }

    def test_expansion_preserves_physical_actor_and_zeroes_new_contract(self):
        state = self.state()
        expanded, keys = expand_v07_actor(state)

        self.assertEqual(expanded["actor.0.weight"].shape, (512, TARGET_OBS))
        self.assertEqual(expanded["actor.6.weight"].shape, (TARGET_ACTIONS, 128))
        self.assertEqual(expanded["std"].shape, (TARGET_ACTIONS,))
        self.assertEqual(
            torch.count_nonzero(expanded["actor.0.weight"][:, SOURCE_OBS:]).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(expanded["actor.6.weight"][SOURCE_ACTIONS:]).item(),
            0,
        )
        self.assertEqual(expanded["actor.6.bias"][SOURCE_ACTIONS].item(), 0.0)
        self.assertAlmostEqual(expanded["std"][SOURCE_ACTIONS].item(), FREQUENCY_STD, places=7)
        self.assertLessEqual(matched_output_check(state, expanded, keys), 1e-5)

    def test_wrong_v07_source_dimension_is_rejected(self):
        state = self.state()
        state["actor.0.weight"] = 0.02 * torch.randn(512, SOURCE_OBS - 1)
        with self.assertRaises(ValueError):
            expand_v07_actor(state)

    def test_control_keeps_v07_task_contract_and_fixed_cadence(self):
        repo_path = Path(
            "experiments/g1_stairs/configs/_test_v07_model2130_expanded492_control.yaml"
        )
        try:
            repo_path.write_text(
                yaml.safe_dump(control_config(output_hash="0" * 64), sort_keys=False),
                encoding="utf-8",
            )
            resolved = validate_control(repo_path)
            self.assertEqual(resolved["gait"]["mode"], "fixed")
            self.assertFalse(resolved["gait"]["coordinated_motion"])
            self.assertFalse(
                resolved.get("episode_contract", {}).get("strict_stair_gait", False)
            )
        finally:
            repo_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
