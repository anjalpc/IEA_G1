"""Tests for configuration-driven reward-stage selection."""

import ast
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
    def test_geometry_locomotion_v01_resolves_exact_contract(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / (
                "experiments/g1_stairs/configs/"
                "geometry_locomotion_v01_t0.yaml"
            )
        )
        materialize_legacy_reward_config(resolved)

        self.assertEqual(
            resolved["robot"]["asset"],
            "G1_GEOMETRY_LOCOMOTION_ROBOT_CFG",
        )
        self.assertEqual(
            resolved["command"]["forward_velocity_mps"],
            0.35,
        )
        self.assertEqual(
            resolved["terrain"]["riser_height_range_m"],
            [0.12, 0.12],
        )

        observations = resolved["observations"]
        self.assertIs(observations["actor_height_scan"], False)
        self.assertIs(observations["critic_height_scan"], True)
        self.assertIs(observations["terrain_scan_in_actor"], False)
        self.assertIs(observations["terrain_scan_in_critic"], True)
        self.assertIs(
            observations["additions"]["local_terrain_height_scan"][
                "actor_input"
            ],
            False,
        )

        initialization = resolved["policy_initialization"]
        self.assertEqual(initialization["locomotion_observations"], 480)
        self.assertEqual(initialization["terrain_observations"], 0)
        self.assertEqual(initialization["actor_height_scan_observations"], 0)
        self.assertEqual(initialization["explicit_geometry_observations"], 5)
        self.assertEqual(initialization["stair_observations"], 485)
        self.assertEqual(initialization["actions"], 29)

        self.assertIs(
            resolved["reward_experiment"]["strict_runtime_set"],
            True,
        )
        self.assertEqual(
            active_reward_weights(resolved),
            {
                "track_lin_vel_xy": 1.0,
                "track_ang_vel_z": 0.5,
                "alive": 0.15,
                "upright": -5.0,
                "base_angular_velocity": -0.05,
                "joint_acc": -2.5e-7,
                "action_rate": -0.05,
                "dof_pos_limits": -5.0,
                "feet_slide": -0.2,
                "undesired_contacts": -5.0,
            },
        )

        specialist = resolved.get("specialist_stair")
        self.assertFalse(
            isinstance(specialist, dict)
            and specialist.get("enabled", False)
        )

    def test_geometry_locomotion_rewards_exist_in_runtime_catalog(self):
        root = Path(__file__).resolve().parents[4]
        resolved, _ = resolve_config(
            root / (
                "experiments/g1_stairs/configs/"
                "geometry_locomotion_v01_t0.yaml"
            )
        )
        materialize_legacy_reward_config(resolved)
        active = set(active_reward_weights(resolved))

        source = (
            root / "simulation/isaac_lab/g1_stairs/env_cfg.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        catalog = set()
        for node in tree.body:
            if not (
                isinstance(node, ast.ClassDef)
                and node.name == "StairRewardsCfg"
            ):
                continue
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            catalog.add(target.id)
                elif isinstance(statement, ast.AnnAssign):
                    if isinstance(statement.target, ast.Name):
                        catalog.add(statement.target.id)

        self.assertEqual(active - catalog, set())

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

    def test_v07_closes_phase4_hover_loophole_without_changing_v06_defaults(self):
        root = Path(__file__).resolve().parents[4]
        v06, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v06.yaml"
        )
        v07, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v07.yaml"
        )
        materialize_legacy_reward_config(v06)
        materialize_legacy_reward_config(v07)

        v06_waypoint = v06["reward_parameters"]["swing_waypoint_progress"]
        v06_body = v06["reward_parameters"]["swing_body_advance_progress"]
        self.assertFalse(v06_waypoint.get("reversible_touchdown_credit", False))
        self.assertFalse(
            v06_body.get("require_live_touchdown_commitment", False)
        )

        v07_waypoint = v07["reward_parameters"]["swing_waypoint_progress"]
        v07_body = v07["reward_parameters"]["swing_body_advance_progress"]
        self.assertTrue(v07_waypoint["reversible_touchdown_credit"])
        self.assertTrue(v07_body["require_live_touchdown_commitment"])
        self.assertEqual(v07_body["touchdown_containment_minimum"], 0.75)
        self.assertEqual(v07_body["upper_support_minimum"], 0.60)

        weights = active_reward_weights(v07)
        self.assertEqual(weights["swing_waypoint_progress"], 4.0)
        self.assertEqual(weights["swing_body_advance_progress"], 1.5)
        self.assertEqual(weights["body_transfer_progress"], 3.0)
        self.assertEqual(v07["specialist_stair"]["staging"]["distance_before_riser_m"], [0.30, 0.30])
        self.assertEqual(v07["execution"]["maximum_iterations"], 300)
        self.assertTrue(
            v07["policy_initialization"]["source"]["path"].endswith(
                "g1_stair_v06_overnight1000_20260926_200718/model_999.pt"
            )
        )

    def test_v08_adds_xy_support_lock_and_continuous_touchdown(self):
        root = Path(__file__).resolve().parents[4]
        v07, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v07.yaml"
        )
        v08, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v08.yaml"
        )
        materialize_legacy_reward_config(v07)
        materialize_legacy_reward_config(v08)

        v07_weights = active_reward_weights(v07)
        v08_weights = active_reward_weights(v08)
        self.assertNotIn("support_foot_anchor_guard", v07_weights)
        self.assertEqual(v08_weights["support_foot_anchor_guard"], -1.0)
        self.assertEqual(v08_weights["swing_waypoint_progress"], 4.0)
        self.assertEqual(v08_weights["swing_body_advance_progress"], 1.5)

        waypoint = v08["reward_parameters"]["swing_waypoint_progress"]
        body = v08["reward_parameters"]["swing_body_advance_progress"]
        support = v08["reward_parameters"]["support_foot_anchor_guard"]
        self.assertTrue(waypoint["reversible_touchdown_credit"])
        self.assertTrue(waypoint["continuous_touchdown_credit"])
        self.assertTrue(body["support_anchor_xy"])
        self.assertTrue(body["require_live_touchdown_commitment"])
        self.assertEqual(support["anchor_free_m"], 0.015)
        self.assertEqual(support["anchor_zero_m"], 0.060)
        self.assertEqual(support["speed_zero_mps"], 0.20)

        self.assertEqual(
            v08["specialist_stair"]["staging"]["distance_before_riser_m"],
            [0.30, 0.30],
        )
        self.assertEqual(v08["execution"]["maximum_iterations"], 300)
        self.assertTrue(
            v08["policy_initialization"]["source"]["path"].endswith(
                "g1_stair_v06_overnight1000_20260926_200718/model_999.pt"
            )
        )

    def test_v09_closes_support_lock_recontact_gate_loophole(self):
        root = Path(__file__).resolve().parents[4]
        v08, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v08.yaml"
        )
        v09, _ = resolve_config(
            root / "experiments/g1_stairs/configs/g1_stair_v09.yaml"
        )
        materialize_legacy_reward_config(v08)
        materialize_legacy_reward_config(v09)

        v08_support = v08["reward_parameters"]["support_foot_anchor_guard"]
        v09_support = v09["reward_parameters"]["support_foot_anchor_guard"]
        self.assertNotIn("anchor_phase_gate_only", v08_support)
        self.assertNotIn("latch_anchor_on_active_phase", v08_support)
        self.assertTrue(v09_support["anchor_phase_gate_only"])
        self.assertTrue(v09_support["latch_anchor_on_active_phase"])

        v09_weights = active_reward_weights(v09)
        self.assertEqual(v09_weights["support_foot_anchor_guard"], -1.0)
        self.assertEqual(v09_weights["swing_waypoint_progress"], 4.0)
        self.assertEqual(v09_weights["swing_body_advance_progress"], 1.5)
        self.assertEqual(
            v09["specialist_stair"]["staging"]["distance_before_riser_m"],
            [0.30, 0.30],
        )
        self.assertEqual(v09["execution"]["maximum_iterations"], 300)
        self.assertTrue(
            v09["policy_initialization"]["source"]["path"].endswith(
                "g1_stair_v06_overnight1000_20260926_200718/model_999.pt"
            )
        )

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
