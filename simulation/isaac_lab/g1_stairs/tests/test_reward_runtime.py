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
                "alive": 0.15,
                "upright": -5.0,
                "base_angular_velocity": -0.05,
                "feet_slide": -0.2,
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


    def test_single_step_config_resolves_exact_specialist_objective(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / "experiments/g1_stairs/configs/specialist_single_step_smoke.yaml"
        )
        materialize_legacy_reward_config(resolved)
        self.assertEqual(
            active_reward_weights(resolved),
            {
                "alive": 0.15,
                "upright": -5.0,
                "base_angular_velocity": -0.05,
                "feet_slide": -0.2,
                "action_rate": -0.01,
                "joint_acc": -2.5e-7,
                "dof_pos_limits": -5.0,
                "undesired_contacts": -1.0,
                "failure_termination_event": -4.0,
                "approach_progress": 1.0,
                "swing_clearance_progress": 2.5,
                "upper_foothold_progress": 2.5,
                "body_transfer_progress": 3.0,
                "trailing_foot_recovery": 2.0,
                "riser_strike_event": -2.0,
                "single_step_success": 6.0,
            },
        )


    def test_single_step_acquisition_staging_is_fixed_nominal(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / "experiments/g1_stairs/configs/specialist_single_step_smoke.yaml"
        )
        staging = resolved["specialist_stair"]["staging"]
        self.assertEqual(staging["distance_before_riser_m"], [0.30, 0.30])
        self.assertEqual(staging["lateral_offset_m"], [0.0, 0.0])
        self.assertEqual(staging["yaw_offset_rad"], [0.0, 0.0])
        self.assertEqual(
            resolved["reward_parameters"]["approach_progress"][
                "preparation_distance_m"
            ],
            0.10,
        )


    def test_lift_transfer_config_activates_supported_lift_curriculum(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / (
                "experiments/g1_stairs/configs/"
                "specialist_single_step_lift_transfer1000.yaml"
            )
        )
        materialize_legacy_reward_config(resolved)
        weights = active_reward_weights(resolved)
        self.assertEqual(weights["alive"], 0.0)
        self.assertEqual(weights["approach_progress"], 0.25)
        self.assertEqual(weights["swing_lift_progress"], 3.0)
        self.assertEqual(weights["swing_clearance_progress"], 2.5)
        self.assertEqual(weights["upper_foothold_progress"], 2.5)
        self.assertEqual(weights["single_step_success"], 6.0)
        self.assertEqual(
            resolved["policy_initialization"]["type"],
            "rsl_rl_checkpoint_actor_transfer",
        )
        self.assertTrue(
            resolved["policy_initialization"]["source"]["path"].endswith(
                "specialist_single_step_transfer3000_20260926_112515/model_1500.pt"
            )
        )

    def test_v03_directional_swing_replaces_unconstrained_lift_terms(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v03.yaml"
        )
        materialize_legacy_reward_config(resolved)
        weights = active_reward_weights(resolved)
        self.assertEqual(weights["alive"], 0.0)
        self.assertEqual(weights["approach_progress"], 0.10)
        self.assertEqual(weights["swing_target_progress"], 4.0)
        self.assertEqual(weights["swing_motion_guard"], -1.0)
        self.assertEqual(weights["body_heading_guard"], -0.25)
        self.assertNotIn("swing_lift_progress", weights)
        self.assertNotIn("swing_clearance_progress", weights)
        self.assertIsNone(
            resolved["reward_parameters"]["swing_clearance_progress"]
        )
        self.assertTrue(
            resolved["reward_parameters"]["swing_target_progress"]["swing_left"]
        )
        self.assertEqual(
            resolved["policy_initialization"]["type"],
            "rsl_rl_checkpoint_actor_transfer",
        )
        self.assertTrue(
            resolved["policy_initialization"]["source"]["path"].endswith(
                "specialist_single_step_transfer3000_20260926_112515/model_1500.pt"
            )
        )

    def test_v04_waypoint_arc_replaces_vertical_only_swing_phase(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v04.yaml"
        )
        materialize_legacy_reward_config(resolved)
        weights = active_reward_weights(resolved)
        self.assertEqual(weights["alive"], 0.0)
        self.assertEqual(weights["approach_progress"], 0.05)
        self.assertEqual(weights["swing_waypoint_progress"], 4.0)
        self.assertEqual(weights["swing_waypoint_guard"], -1.5)
        self.assertEqual(weights["posture_envelope_guard"], -0.25)
        self.assertNotIn("swing_target_progress", weights)
        self.assertNotIn("swing_motion_guard", weights)
        self.assertNotIn("swing_lift_progress", weights)
        self.assertNotIn("swing_clearance_progress", weights)
        self.assertIsNone(resolved["reward_parameters"]["swing_clearance_progress"])
        params = resolved["reward_parameters"]["swing_waypoint_progress"]
        self.assertTrue(params["swing_left"])
        self.assertEqual(params["early_x_fraction"], 0.35)
        self.assertEqual(params["clearance_x_fraction"], 0.60)
        self.assertEqual(params["maximum_touchdown_vertical_speed_mps"], 0.25)
        self.assertTrue(
            resolved["policy_initialization"]["source"]["path"].endswith(
                "specialist_single_step_transfer3000_20260926_112515/model_1500.pt"
            )
        )

    def test_v05_support_locked_phase0_is_bounded_and_noncompensatory(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v05.yaml"
        )
        materialize_legacy_reward_config(resolved)
        weights = active_reward_weights(resolved)
        self.assertEqual(weights["approach_progress"], 0.0)
        self.assertEqual(weights["swing_waypoint_progress"], 4.0)
        self.assertEqual(weights["swing_waypoint_guard"], -1.5)
        progress = resolved["reward_parameters"]["swing_waypoint_progress"]
        guard = resolved["reward_parameters"]["swing_waypoint_guard"]
        self.assertTrue(progress["phase0_bottleneck"])
        self.assertTrue(progress["phase0_require_grounded_swing"])
        self.assertTrue(guard["enforce_phase0_support"])
        self.assertTrue(
            resolved["policy_initialization"]["source"]["path"].endswith(
                "g1_stair_v04_20260926_135458/model_50.pt"
            )
        )

    def test_v06_continues_v05_actor_with_only_body_advance_added(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v06.yaml"
        )
        materialize_legacy_reward_config(resolved)
        weights = active_reward_weights(resolved)
        self.assertEqual(weights["swing_waypoint_progress"], 4.0)
        self.assertEqual(weights["swing_waypoint_guard"], -1.5)
        self.assertEqual(weights["swing_body_advance_progress"], 1.5)
        self.assertEqual(weights["body_transfer_progress"], 3.0)
        self.assertEqual(resolved["execution"]["environments"], 2048)
        self.assertEqual(resolved["execution"]["maximum_iterations"], 500)
        self.assertEqual(resolved["ppo"]["num_mini_batches"], 32)
        self.assertEqual(resolved["ppo"]["learning_rate"], 0.00005)
        self.assertTrue(
            resolved["policy_initialization"]["source"]["path"].endswith(
                "g1_stair_v05_20260926_144601/model_3999.pt"
            )
        )
        params = resolved["reward_parameters"]["swing_body_advance_progress"]
        self.assertEqual(params["target_forward_from_support_m"], 0.080)
        self.assertEqual(params["clearance_x_fraction"], 0.60)
        self.assertEqual(params["support_anchor_limit_m"], 0.050)

    def test_reward_parameter_overrides_are_active_term_scoped(self):
        class _Term:
            def __init__(self, weight, **params):
                self.func = object()
                self.weight = weight
                self.params = params

        class _Rewards:
            approach_progress = _Term(1.0, preparation_distance_m=0.18)
            legacy_extra = _Term(7.0, unused=1.0)

        class _EnvCfg:
            rewards = _Rewards()

        resolved = {
            "reward_experiment": {
                "active_stage": "single",
                "strict_runtime_set": True,
            },
            "rewards": {
                "terms": {
                    "approach_progress_raw": {"active_weight": 2.0},
                },
                "stage_plan": {
                    "single": {
                        "enabled_terms": {
                            "approach_progress_raw": 2.0,
                        }
                    }
                },
            },
            "reward_parameters": {
                "approach_progress": {
                    "preparation_distance_m": 0.22,
                }
            },
        }

        weights = apply_resolved_rewards(_EnvCfg(), resolved)
        self.assertEqual(weights, {"approach_progress": 2.0})
        self.assertAlmostEqual(
            _EnvCfg.rewards.approach_progress.params["preparation_distance_m"],
            0.22,
        )
        self.assertIsNone(_EnvCfg.rewards.legacy_extra)

    def test_reward_parameter_null_removes_inherited_inactive_override(self):
        class _Term:
            def __init__(self):
                self.func = object()
                self.weight = 1.0
                self.params = {"x": 1.0}

        class _Rewards:
            upright = _Term()
            approach_progress = _Term()

        class _EnvCfg:
            rewards = _Rewards()

        resolved = {
            "reward_experiment": {
                "active_stage": "foundation",
                "strict_runtime_set": True,
            },
            "rewards": {
                "terms": {"upright_raw": {}},
                "stage_plan": {
                    "foundation": {
                        "enabled_terms": {"upright_raw": -1.0}
                    }
                },
            },
            "reward_parameters": {
                "approach_progress": None,
            },
        }

        weights = apply_resolved_rewards(_EnvCfg(), resolved)
        self.assertEqual(weights, {"upright": -1.0})
        self.assertIsNone(_EnvCfg.rewards.approach_progress)

    def test_reward_parameter_override_rejects_inactive_term(self):
        class _Term:
            def __init__(self):
                self.func = object()
                self.weight = 1.0
                self.params = {"x": 1.0}

        class _Rewards:
            upright = _Term()
            approach_progress = _Term()

        class _EnvCfg:
            rewards = _Rewards()

        resolved = {
            "reward_experiment": {
                "active_stage": "foundation",
                "strict_runtime_set": True,
            },
            "rewards": {
                "terms": {"upright_raw": {}},
                "stage_plan": {
                    "foundation": {
                        "enabled_terms": {"upright_raw": -1.0}
                    }
                },
            },
            "reward_parameters": {
                "approach_progress": {"x": 2.0}
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "only override active rewards",
        ):
            apply_resolved_rewards(_EnvCfg(), resolved)

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
