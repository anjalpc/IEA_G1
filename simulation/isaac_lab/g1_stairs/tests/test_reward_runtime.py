"""Tests for experiment-driven reward-stage selection."""

import unittest
from pathlib import Path

from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import resolve_config


class RewardRuntimeTests(unittest.TestCase):
    def test_experiment_b_resolves_exact_runtime_rewards(self):
        root = Path(__file__).resolve().parents[4]
        config = root / "experiments/g1_stairs/configs/003_nominal_reward_debug.yaml"
        resolved, _ = resolve_config(config)

        self.assertEqual(
            resolved["reward_experiment"]["active_stage"],
            "B_add_upright",
        )
        self.assertEqual(
            active_reward_weights(resolved),
            {
                "forward_progress": 1.0,
                "stair_progress": 1.0,
                "upright": -1.0,
            },
        )

    def test_unknown_stage_is_rejected(self):
        resolved = {
            "reward_experiment": {"active_stage": "does_not_exist"},
            "rewards": {
                "stage_plan": {
                    "A_progress_only": {
                        "enabled_terms": {"forward_progress_raw": 1.0}
                    }
                }
            },
        }

        with self.assertRaisesRegex(ValueError, "unknown active reward stage"):
            active_reward_weights(resolved)

    def test_unimplemented_active_reward_is_rejected(self):
        resolved = {
            "reward_experiment": {"active_stage": "C_add_completion"},
            "rewards": {
                "stage_plan": {
                    "A_progress_only": {
                        "enabled_terms": {"forward_progress_raw": 1.0}
                    },
                    "C_add_completion": {
                        "enabled_terms": {"completion_raw": 3.0}
                    },
                }
            },
        }

        with self.assertRaisesRegex(ValueError, "no runtime implementation"):
            active_reward_weights(resolved)


if __name__ == "__main__":
    unittest.main()
