"""Tests for configuration-driven reward-stage selection."""

import unittest
from pathlib import Path

from simulation.isaac_lab.g1_stairs.reward_registry import (
    materialize_legacy_reward_config,
)
from simulation.isaac_lab.g1_stairs.reward_runtime import (
    active_reward_weights,
    apply_resolved_rewards,
)
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import resolve_config


class RewardRuntimeTests(unittest.TestCase):
    def test_specialist_smoke_resolves_exact_foundation_rewards(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / "experiments/g1_stairs/configs/specialist_smoke.yaml"
        )
        materialize_legacy_reward_config(resolved)
        self.assertEqual(
            active_reward_weights(resolved),
            {
                "upright": -2.0,
                "action_rate": -0.01,
                "joint_acc": -2.5e-7,
                "dof_pos_limits": -5.0,
                "undesired_contacts": -1.0,
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
        self.assertEqual(weights, {"upright": -2.0, "action_rate": -0.02})
        self.assertEqual(_EnvCfg.rewards.upright.weight, -2.0)
        self.assertEqual(_EnvCfg.rewards.action_rate.weight, -0.02)
        self.assertIsNone(_EnvCfg.rewards.legacy_extra)

    def test_archived_runtime_sections_fail_closed(self):
        resolved = {
            "physical_step": {"enabled": True},
            "reward_experiment": {"active_stage": "foundation"},
            "rewards": {
                "terms": {"upright_raw": {}},
                "stage_plan": {
                    "foundation": {"enabled_terms": {"upright_raw": -1.0}}
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "archived pre-specialist runtime"):
            active_reward_weights(resolved)

    def test_unknown_stage_is_rejected(self):
        resolved = {
            "reward_experiment": {"active_stage": "does_not_exist"},
            "rewards": {
                "terms": {"upright_raw": {}},
                "stage_plan": {
                    "foundation": {"enabled_terms": {"upright_raw": -1.0}}
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "unknown active reward stage"):
            active_reward_weights(resolved)

    def test_unimplemented_active_reward_is_rejected(self):
        resolved = {
            "reward_experiment": {"active_stage": "foundation"},
            "rewards": {
                "terms": {},
                "stage_plan": {
                    "foundation": {
                        "enabled_terms": {"unknown_reward_raw": 3.0}
                    }
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "no runtime implementation"):
            active_reward_weights(resolved)


if __name__ == "__main__":
    unittest.main()
