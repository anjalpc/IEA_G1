"""Tests for the deterministic Experiment 021 halfway fixture."""

from dataclasses import replace
from pathlib import Path
import unittest

from simulation.isaac_lab.g1_stairs.halfway_reset import (
    HalfwayResetConfig,
    parse_halfway_reset_config,
    solve_leading_leg_pose,
)
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
)


class HalfwayResetTest(unittest.TestCase):
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

    def test_invalid_halfway_pose_is_rejected(self):
        invalid = replace(
            HalfwayResetConfig(),
            leading_foot_x_from_root_m=1.0,
        )
        with self.assertRaisesRegex(ValueError, "outside leg workspace"):
            solve_leading_leg_pose(invalid)


if __name__ == "__main__":
    unittest.main()
