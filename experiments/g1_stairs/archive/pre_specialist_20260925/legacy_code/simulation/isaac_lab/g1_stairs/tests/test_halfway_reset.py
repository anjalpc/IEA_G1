"""Tests for the deterministic Experiment 021 halfway fixture."""

from dataclasses import replace
import ast
import importlib.util
import os
from pathlib import Path
import unittest

from simulation.isaac_lab.g1_stairs.halfway_reset import (
    HalfwayResetConfig,
    halfway_reset_event_params,
    parse_halfway_reset_config,
    solve_leading_leg_pose,
)
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
)


class HalfwayResetTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("IEA_G1_ISAAC_APP_TESTS") == "1"
        and importlib.util.find_spec("isaaclab")
        and importlib.util.find_spec("unitree_rl_lab"),
        "requires an explicitly launched Isaac application context",
    )
    def test_normal_stair_env_cfg_instantiates_without_halfway_reset(self):
        from simulation.isaac_lab.g1_stairs.env_cfg import StairEnvCfg

        cfg = StairEnvCfg()
        self.assertIsNotNone(cfg.events.reset_base)
        self.assertIsNotNone(cfg.events.reset_robot_joints)

    def test_021_config_parses_and_preserves_onnx_initialization(self):
        root = Path(__file__).resolve().parents[4]
        path = root / "experiments/g1_stairs/configs/021_halfway_support_05m_smoke.yaml"
        resolved, _ = resolve_config(path)
        cfg = parse_halfway_reset_config(resolved)

        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.leading_leg, "left")
        self.assertEqual(resolved["execution"]["environments"], 32)
        self.assertEqual(resolved["execution"]["maximum_iterations"], 1)
        self.assertEqual(
            resolved["policy_initialization"]["type"],
            "unitree_velocity_onnx_transfer",
        )
        self.assertEqual(
            active_reward_weights(resolved),
            {
                "track_lin_vel_xy": 1.0,
                "upright": -5.0,
                "action_rate": -0.05,
                "waist_deviation": -1.0,
                "base_above_transition": 1.0,
                "both_feet_supported_transition": 2.0,
                "completed_transition": 3.0,
            },
        )

    def test_pose_solver_changes_only_sagittal_leading_leg(self):
        hip, knee, ankle = solve_leading_leg_pose(HalfwayResetConfig())
        self.assertAlmostEqual(hip + knee + ankle, 0.0)
        self.assertGreater(knee, 0.30)

    def test_manager_params_contain_only_plain_configuration_data(self):
        params = halfway_reset_event_params(HalfwayResetConfig())
        self.assertIsInstance(params, dict)
        self.assertIsInstance(params["config"], dict)
        self.assertNotIsInstance(params["config"], HalfwayResetConfig)

    def test_transition_term_defers_tensor_state_until_runtime_call(self):
        root = Path(__file__).resolve().parents[4]
        source = (
            root / "simulation/isaac_lab/g1_stairs/rewards.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        transition = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "halfway_step_transition"
        )
        initializer = next(
            node
            for node in transition.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(initializer)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("torch.full", calls)
        self.assertNotIn("torch.zeros", calls)

    def test_default_config_does_not_enable_halfway_reset(self):
        root = Path(__file__).resolve().parents[4]
        path = root / "experiments/g1_stairs/configs/nominal_stairs_base.yaml"
        resolved, _ = resolve_config(path)
        self.assertIsNone(parse_halfway_reset_config(resolved))

    def test_invalid_halfway_pose_is_rejected(self):
        invalid = replace(
            HalfwayResetConfig(),
            leading_foot_x_from_root_m=1.0,
        )
        with self.assertRaisesRegex(ValueError, "outside leg workspace"):
            solve_leading_leg_pose(invalid)


if __name__ == "__main__":
    unittest.main()
