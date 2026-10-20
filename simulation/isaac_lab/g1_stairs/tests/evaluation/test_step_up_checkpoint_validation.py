"""Tests for checkpoint step-up calibration lifecycle."""

import tempfile
import unittest
from pathlib import Path


from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
    AppliedReplayConfig,
    apply_resolved_replay_agent_config,
    calibration_ready,
    load_replay_resolved_config,
    resolved_replay_riser_height,
    stair_outcome_config_from_replay,
)


class StepUpCheckpointCalibrationTest(unittest.TestCase):
    def test_calibration_finishes_as_soon_as_both_populations_are_ready(self):
        forces = [100.0] * 5
        heights = [0.78] * 5

        self.assertTrue(calibration_ready(forces, heights))

    def test_calibration_requires_both_measurement_populations(self):
        self.assertFalse(
            calibration_ready([100.0] * 5, [0.78] * 4)
        )
        self.assertFalse(
            calibration_ready([100.0] * 4, [0.78] * 5)
        )

    def test_calibration_rejects_invalid_minimum(self):
        with self.assertRaises(ValueError):
            calibration_ready([], [], minimum_samples=0)


    def test_single_recorded_height_overrides_inherited_scalar(self):
        self.assertEqual(
            resolved_replay_riser_height(
                {
                    "riser_height_m": 0.20,
                    "riser_heights_m": [0.04],
                }
            ),
            0.04,
        )

    def test_multi_height_replay_keeps_scalar_without_explicit_override(self):
        self.assertEqual(
            resolved_replay_riser_height(
                {
                    "riser_height_m": 0.20,
                    "riser_heights_m": [0.04, 0.08],
                }
            ),
            0.20,
        )

    def test_empty_recorded_height_list_is_rejected(self):
        with self.assertRaises(ValueError):
            resolved_replay_riser_height(
                {
                    "riser_height_m": 0.20,
                    "riser_heights_m": [],
                }
            )


    def test_replay_loader_normalizes_specialist_reward_schema(self):
        payload = """
reward_experiment:
  active_stage: specialist_foundation
  strict_runtime_set: true
  rewards:
    stage_plan:
      specialist_foundation:
        upright: -2.0
        action_rate: -0.01
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved_config.yaml"
            path.write_text(payload, encoding="utf-8")
            resolved = load_replay_resolved_config(path)

        self.assertIn("rewards", resolved)
        self.assertIn("stage_plan", resolved["rewards"])
        self.assertEqual(
            resolved["rewards"]["stage_plan"][
                "specialist_foundation"
            ]["enabled_terms"]["upright_raw"],
            -2.0,
        )
        self.assertEqual(
            resolved["rewards"]["stage_plan"][
                "specialist_foundation"
            ]["enabled_terms"]["action_rate_raw"],
            -0.01,
        )


    def test_replay_outcome_geometry_uses_recorded_specialist_frame(self):
        config = stair_outcome_config_from_replay(
            AppliedReplayConfig(
                riser_height_m=0.06,
                tread_depth_m=0.25,
                step_count=1,
                forward_velocity_mps=0.0,
                start_x_m=0.0,
                first_riser_x_m=0.30,
                required_platform_continuation_m=0.40,
            )
        )
        self.assertEqual(config.start_x_m, 0.0)
        self.assertEqual(config.first_riser_x_m, 0.30)
        self.assertEqual(config.step_count, 1)
        self.assertEqual(config.required_platform_continuation_m, 0.40)

    def test_replay_agent_uses_recorded_action_clip(self):
        class AgentCfg:
            clip_actions = 100.0

        agent_cfg = AgentCfg()
        apply_resolved_replay_agent_config(
            agent_cfg,
            {"ppo": {"action_clip": 5.0}},
        )
        self.assertEqual(agent_cfg.clip_actions, 5.0)

    def test_replay_agent_rejects_invalid_action_clip(self):
        class AgentCfg:
            clip_actions = 100.0

        with self.assertRaises(ValueError):
            apply_resolved_replay_agent_config(
                AgentCfg(),
                {"ppo": {"action_clip": 0.0}},
            )


if __name__ == "__main__":
    unittest.main()
