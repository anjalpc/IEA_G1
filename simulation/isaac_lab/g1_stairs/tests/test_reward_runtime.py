"""Tests for experiment-driven reward-stage selection."""

import unittest
from pathlib import Path

from simulation.isaac_lab.g1_stairs.reward_runtime import (
    active_reward_weights,
    apply_resolved_rewards,
)
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

    def test_experiment_c_resolves_exact_runtime_rewards(self):
        root = Path(__file__).resolve().parents[4]
        config = (
            root
            / "experiments/g1_stairs/configs/"
            "005_experiment_c_completion.yaml"
        )
        resolved, sources = resolve_config(config)

        self.assertEqual(
            sources,
            [
                "experiments/g1_stairs/configs/nominal_stairs_base.yaml",
                "experiments/g1_stairs/configs/005_experiment_c_completion.yaml",
            ],
        )
        self.assertEqual(
            resolved["reward_experiment"]["active_stage"],
            "C_add_completion",
        )
        self.assertEqual(
            active_reward_weights(resolved),
            {
                "forward_progress": 1.0,
                "stair_progress": 1.0,
                "upright": -1.0,
                "platform_entry": 1.0,
                "task_completion": 3.0,
            },
        )


    def test_strict_runtime_set_prunes_undeclared_rewards(self):
        class _Term:
            def __init__(self, weight):
                self.func = object()
                self.weight = weight

        class _Rewards:
            upright = _Term(-1.0)
            action_rate = _Term(-0.01)
            legacy_extra = _Term(7.0)

        class _EnvCfg:
            rewards = _Rewards()

        resolved = {
            "reward_experiment": {
                "active_stage": "foundation",
                "strict_runtime_set": True,
            },
            "rewards": {
                "terms": {
                    "upright_raw": {"active_weight": -2.0},
                    "action_rate_raw": {"active_weight": -0.02},
                },
                "stage_plan": {
                    "foundation": {
                        "enabled_terms": {
                            "upright_raw": -2.0,
                            "action_rate_raw": -0.02,
                        }
                    }
                },
            },
        }

        weights = apply_resolved_rewards(_EnvCfg(), resolved)

        self.assertEqual(
            weights,
            {"upright": -2.0, "action_rate": -0.02},
        )
        self.assertEqual(_EnvCfg.rewards.upright.weight, -2.0)
        self.assertEqual(_EnvCfg.rewards.action_rate.weight, -0.02)
        self.assertIsNone(_EnvCfg.rewards.legacy_extra)

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
                        "enabled_terms": {"unknown_reward_raw": 3.0}
                    },
                }
            },
        }

        with self.assertRaisesRegex(ValueError, "no runtime implementation"):
            active_reward_weights(resolved)


if __name__ == "__main__":
    unittest.main()
