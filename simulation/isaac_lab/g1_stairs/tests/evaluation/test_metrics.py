"""Tests for machine-readable RL metrics."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from simulation.isaac_lab.g1_stairs.evaluation.metrics import (
    RunMetricsWriter,
    wilson_interval,
)
from simulation.isaac_lab.g1_stairs.contact_sensor_contract import (
    resolve_available_body_indices,
)
from simulation.isaac_lab.g1_stairs.evaluation.isaac_training_metrics import (
    POSTURE_DIAGNOSTIC_NAMES,
    STAIR_DIAGNOSTIC_NAMES,
    recover_raw_reward_step,
)
from simulation.isaac_lab.g1_stairs.evaluation.stair_diagnostics import (
    NamedTensorStatsAccumulator,
    TensorStatsAccumulator,
    physical_diagnostic_matrix,
)
from simulation.isaac_lab.g1_stairs.evaluation.stair_outcomes import (
    EpisodeResult,
    Outcome,
)


def result(
    outcome: Outcome,
    *,
    success: bool,
) -> EpisodeResult:
    return EpisodeResult(
        outcome=outcome,
        success=success,
        termination_time_s=5.0,
        primary_reason=outcome.value,
        secondary_flags=(),
        metrics={
            "maximum_x_m": 4.8 if success else 2.2,
            "maximum_base_height_m": 1.4,
            "maximum_step_reached": 3 if success else 1,
            "approach_completed": True,
            "platform_reached": success,
            "platform_continuation_m": (
                2.05 if success else 0.0
            ),
            "upright_dwell_s": 1.1 if success else 0.0,
        },
    )


class MetricsTest(unittest.TestCase):

    def test_posture_diagnostics_cover_arm_default_error_and_action(self):
        expected = {
            "posture_joint_pos_rad_left_shoulder_pitch",
            "posture_joint_default_rad_left_shoulder_pitch",
            "posture_joint_error_rad_left_shoulder_pitch",
            "posture_joint_vel_radps_left_shoulder_pitch",
            "posture_policy_action_left_shoulder_pitch",
            "posture_joint_pos_rad_right_elbow",
            "posture_policy_action_right_wrist_yaw",
            "posture_arms_error_rms_rad",
            "posture_arms_action_rms",
        }
        self.assertTrue(expected.issubset(set(POSTURE_DIAGNOSTIC_NAMES)))
        self.assertEqual(
            len(POSTURE_DIAGNOSTIC_NAMES),
            len(set(POSTURE_DIAGNOSTIC_NAMES)),
        )

    def test_directional_swing_diagnostics_cover_orientation_and_lane_failures(self):
        expected = {
            "motion_phase_index",
            "motion_phase_progress",
            "motion_phase_is_0",
            "motion_phase_is_1",
            "motion_phase_is_2",
            "motion_phase_is_3",
            "motion_phase_is_4",
            "motion_phase_is_complete",
            "swing_target_x_m",
            "swing_target_z_m",
            "swing_target_x_error_m",
            "swing_target_z_error_m",
            "swing_lateral_error_m",
            "support_anchor_error_m",
            "swing_foot_speed_mps",
            "swing_lower_floor_recontact",
            "root_to_support_foot_y_m",
            "swing_foot_force_share",
            "support_foot_force_share",
            "both_feet_airborne",
            "swing_overheight_m",
            "support_foot_lift_m",
            "swing_backward_error_m",
            "swing_forward_overshoot_m",
            "swing_foot_vertical_speed_mps",
            "swing_foot_roll_rad",
            "swing_foot_pitch_rad",
            "swing_foot_yaw_rad",
            "support_foot_roll_rad",
            "support_foot_pitch_rad",
            "support_foot_yaw_rad",
        }
        self.assertTrue(expected.issubset(set(STAIR_DIAGNOSTIC_NAMES)))
        self.assertEqual(
            len(STAIR_DIAGNOSTIC_NAMES), len(set(STAIR_DIAGNOSTIC_NAMES))
        )

    def test_forbidden_contact_indices_follow_runtime_sensor_inventory(self):
        available_names = (
            "pelvis",
            "left_knee_link",
            "right_knee_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_ankle_roll_link",
        )

        indices = resolve_available_body_indices(
            available_names,
            (
                "pelvis",
                "left_knee_link",
                "right_knee_link",
                "left_elbow_link",
                "right_elbow_link",
                "left_rubber_hand",
                "right_rubber_hand",
            ),
        )

        self.assertEqual(
            indices,
            {
                "pelvis": 0,
                "left_knee_link": 1,
                "right_knee_link": 2,
                "left_elbow_link": 3,
                "right_elbow_link": 4,
            },
        )

    def test_tensor_stats_accumulator_tracks_distribution(self):
        import torch

        stats = TensorStatsAccumulator()

        stats.update(
            torch.tensor([-2.0, 0.0, 1.0])
        )
        stats.update(
            torch.tensor([3.0])
        )

        snapshot = stats.snapshot()

        self.assertEqual(
            snapshot["sample_count"],
            4,
        )
        self.assertAlmostEqual(
            snapshot["mean"],
            0.5,
        )
        self.assertAlmostEqual(
            snapshot["min"],
            -2.0,
        )
        self.assertAlmostEqual(
            snapshot["max"],
            3.0,
        )
        self.assertAlmostEqual(
            snapshot["nonzero_fraction"],
            0.75,
        )
        self.assertAlmostEqual(
            snapshot["positive_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            snapshot["negative_fraction"],
            0.25,
        )

    def test_named_tensor_stats_accumulator_tracks_columns(self):
        import torch

        stats = NamedTensorStatsAccumulator(
            ("first", "second")
        )

        stats.update(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [3.0, -2.0],
                ]
            )
        )

        stats.update(
            torch.tensor(
                [
                    [5.0, 4.0],
                ]
            )
        )

        snapshot = stats.snapshot()

        self.assertEqual(
            snapshot["first"]["sample_count"],
            3,
        )

        self.assertAlmostEqual(
            snapshot["first"]["mean"],
            3.0,
        )

        self.assertAlmostEqual(
            snapshot["first"]["min"],
            1.0,
        )

        self.assertAlmostEqual(
            snapshot["first"]["max"],
            5.0,
        )

        self.assertAlmostEqual(
            snapshot["second"][
                "nonzero_fraction"
            ],
            2.0 / 3.0,
        )

        self.assertAlmostEqual(
            snapshot["second"][
                "positive_fraction"
            ],
            1.0 / 3.0,
        )

        self.assertAlmostEqual(
            snapshot["second"][
                "negative_fraction"
            ],
            1.0 / 3.0,
        )

        self.assertAlmostEqual(
            snapshot["second"]["mean_abs"],
            2.0,
        )

    def test_physical_diagnostic_matrix_keeps_feet_separate(self):
        import torch

        num_envs = 2

        diagnostics = {
            "root": torch.tensor(
                [
                    [0.1, 0.0, 0.8],
                    [0.2, 0.0, 0.9],
                ]
            ),
            "foot_x_min": torch.tensor(
                [
                    [0.3, 0.4],
                    [0.5, 0.6],
                ]
            ),
            "foot_x_max": torch.tensor(
                [
                    [0.4, 0.5],
                    [0.6, 0.7],
                ]
            ),
            "sole_bottom_min": torch.zeros(
                num_envs,
                2,
            ),
            "sole_bottom_max": torch.ones(
                num_envs,
                2,
            ) * 0.01,
            "foot_normal_z": torch.ones(
                num_envs,
                2,
            ),
            "vertical_force": torch.tensor(
                [
                    [150.0, 75.0],
                    [300.0, 150.0],
                ]
            ),
            "bodyweight": torch.tensor(
                [300.0, 300.0]
            ),
            "loaded": torch.tensor(
                [
                    [True, False],
                    [True, True],
                ]
            ),
            "slow_translation": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "slow_rotation": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "lateral_contained": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "support_quality": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "lower_geometry": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "upper_geometry": torch.zeros(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "landing_rectangle": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "stable_lower": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "stable_upper": torch.zeros(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "swing_zone": torch.zeros(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "swing_ok": torch.zeros(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "clearance_value": torch.tensor(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]
            ),
            "distance_to_riser_face": torch.tensor(
                [
                    [0.6, 0.5],
                    [0.4, 0.3],
                ]
            ),
            "contact": torch.ones(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "contact_onset": torch.zeros(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "near_upper": torch.zeros(
                num_envs,
                2,
                dtype=torch.bool,
            ),
            "precontact_velocity": torch.zeros(
                num_envs,
                2,
                3,
            ),
            "foot_velocity": torch.zeros(
                num_envs,
                2,
                3,
            ),
            "foot_angular_velocity": torch.zeros(
                num_envs,
                2,
                3,
            ),
            "upright": torch.tensor(
                [True, True]
            ),
            "safe": torch.tensor(
                [True, False]
            ),
            "forbidden": torch.tensor(
                [False, True]
            ),
        }

        names, values = physical_diagnostic_matrix(
            diagnostics
        )

        self.assertEqual(
            values.shape[0],
            num_envs,
        )

        self.assertEqual(
            values.shape[1],
            len(names),
        )

        self.assertIn(
            "left_distance_to_riser_face_m",
            names,
        )

        self.assertIn(
            "right_distance_to_riser_face_m",
            names,
        )

        left_force = names.index(
            "left_vertical_force_bodyweight_fraction"
        )

        right_force = names.index(
            "right_vertical_force_bodyweight_fraction"
        )

        self.assertAlmostEqual(
            float(values[0, left_force]),
            0.5,
        )

        self.assertAlmostEqual(
            float(values[0, right_force]),
            0.25,
        )

        self.assertNotEqual(
            names.index(
                "left_foot_x_max_m"
            ),
            names.index(
                "right_foot_x_max_m"
            ),
        )

    def test_raw_reward_recovery_matches_isaac_step_reward_semantics(self):
        # RewardManager._step_reward stores raw_term * weight, not the
        # dt-scaled contribution. A raw term of 3.0 at weight 2.0 therefore
        # appears as 6.0 regardless of the 0.02 s control step.
        raw_step = recover_raw_reward_step(
            6.0,
            dt=0.02,
            weight=2.0,
        )

        self.assertAlmostEqual(raw_step, 3.0)

    def test_zero_weight_reward_raw_value_is_not_reconstructed(self):
        class ArithmeticMustNotRun:
            def __mul__(self, other):
                raise AssertionError("zero-weight term was multiplied")

        raw_step = recover_raw_reward_step(
            ArithmeticMustNotRun(),
            dt=0.02,
            weight=0.0,
        )

        self.assertIsNone(raw_step)

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(95, 100)

        self.assertLess(lower, 0.95)
        self.assertGreater(upper, 0.95)

    def test_training_and_rewards_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RunMetricsWriter(
                Path(directory),
                "test_run",
            )

            writer.record_training_snapshot(
                {
                    "iteration": 10,
                    "total_simulated_steps": 30720,
                    "wall_time_s": 12.5,
                    "episode_return_mean": 4.2,
                    "policy_loss": -0.01,
                    "custom_debug_value": 7.0,
                },
                {
                    "forward_progress": {
                        "raw": 0.4,
                        "weight": 1.0,
                    },
                    "action_rate": {
                        "raw": 0.2,
                        "weight": -0.05,
                    },
                },
            )

            with writer.training_path.open(
                encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["iteration"], "10")
            self.assertIn(
                "custom_debug_value",
                rows[0]["extra_json"],
            )

            with writer.reward_path.open(
                encoding="utf-8"
            ) as stream:
                rewards = list(csv.DictReader(stream))

            self.assertEqual(len(rewards), 2)

            action_rate = next(
                row
                for row in rewards
                if row["name"] == "action_rate"
            )

            self.assertAlmostEqual(
                float(action_rate["weighted_value"]),
                -0.01,
            )

            writer.record_reward_diagnostics(
                iteration=10,
                total_simulated_steps=30720,
                diagnostics={
                    "action_rate": {
                        "sample_count": 100,
                        "mean": 0.2,
                        "std": 0.05,
                        "min": 0.0,
                        "max": 0.8,
                        "nonzero_fraction": 0.7,
                        "positive_fraction": 0.7,
                        "negative_fraction": 0.0,
                        "weight": -0.05,
                        "weighted_mean": -0.01,
                        "weighted_abs_mean": 0.01,
                    },
                },
            )

            with writer.reward_diagnostic_path.open(
                encoding="utf-8"
            ) as stream:
                diagnostics = list(
                    csv.DictReader(stream)
                )

            self.assertEqual(
                len(diagnostics),
                1,
            )
            self.assertEqual(
                diagnostics[0]["name"],
                "action_rate",
            )
            self.assertEqual(
                diagnostics[0]["sample_count"],
                "100",
            )

            writer.record_stair_diagnostics(
                iteration=10,
                total_simulated_steps=30720,
                diagnostics={
                    "touchdown_horizontal_speed_mps": {
                        "sample_count": 8,
                        "mean": 0.3,
                        "std": 0.1,
                        "min": 0.1,
                        "max": 0.6,
                    },
                },
            )

            with writer.stair_diagnostic_path.open(
                encoding="utf-8"
            ) as stream:
                stair_diagnostics = list(
                    csv.DictReader(stream)
                )

            self.assertEqual(
                len(stair_diagnostics),
                1,
            )
            self.assertEqual(
                stair_diagnostics[0]["name"],
                "touchdown_horizontal_speed_mps",
            )

    def test_stair_event_counts_are_written_with_explicit_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RunMetricsWriter(
                Path(directory),
                "stair_event_test",
            )

            writer.record_stair_event_counts(
                iteration=4,
                total_simulated_steps=12345,
                counts={
                    "near_riser_entry_count": 3,
                    "left_swing_entry_count": 0,
                    "right_swing_entry_count": 2,
                },
            )

            with writer.stair_event_path.open(
                encoding="utf-8"
            ) as stream:
                rows = list(
                    csv.DictReader(stream)
                )

            self.assertEqual(
                len(rows),
                3,
            )

            by_name = {
                row["name"]: row
                for row in rows
            }

            self.assertEqual(
                by_name[
                    "left_swing_entry_count"
                ]["count"],
                "0",
            )

            self.assertEqual(
                by_name[
                    "near_riser_entry_count"
                ]["count"],
                "3",
            )

            self.assertEqual(
                by_name[
                    "right_swing_entry_count"
                ]["count"],
                "2",
            )

    def test_stair_episode_counts_are_written_with_explicit_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RunMetricsWriter(
                Path(directory),
                "stair_episode_test",
            )

            writer.record_stair_episode_counts(
                iteration=7,
                total_simulated_steps=54321,
                counts={
                    "episodes_finished_count": 4,
                    "episodes_swing_count": 2,
                    "episodes_completion_count": 0,
                    (
                        "terminal_physical_failure_"
                        "after_swing_count"
                    ): 1,
                },
            )

            with writer.stair_episode_path.open(
                encoding="utf-8"
            ) as stream:
                rows = list(
                    csv.DictReader(stream)
                )

            self.assertEqual(
                len(rows),
                4,
            )

            by_name = {
                row["name"]: row
                for row in rows
            }

            self.assertEqual(
                by_name[
                    "episodes_finished_count"
                ]["count"],
                "4",
            )

            self.assertEqual(
                by_name[
                    "episodes_swing_count"
                ]["count"],
                "2",
            )

            self.assertEqual(
                by_name[
                    "episodes_completion_count"
                ]["count"],
                "0",
            )

            self.assertEqual(
                by_name[
                    (
                        "terminal_physical_failure_"
                        "after_swing_count"
                    )
                ]["count"],
                "1",
            )

    def test_episode_summary_uses_independent_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RunMetricsWriter(
                Path(directory),
                "evaluation_test",
            )

            writer.record_episode(
                result(
                    Outcome.SUCCESS,
                    success=True,
                ),
                checkpoint="model_100.pt",
                evaluation_seed=1001,
                episode_index=0,
            )

            writer.record_episode(
                result(
                    Outcome.FALL,
                    success=False,
                ),
                checkpoint="model_100.pt",
                evaluation_seed=1002,
                episode_index=1,
            )

            summary = writer.write_summary()

            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(
                summary["outcome_counts"]["fall"],
                1,
            )
            self.assertEqual(
                summary["outcome_counts"]["success"],
                1,
            )

            saved = json.loads(
                writer.summary_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                saved["run_id"],
                "evaluation_test",
            )

    def test_invalid_reward_does_not_write_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RunMetricsWriter(
                Path(directory),
                "atomic_snapshot_test",
            )

            with self.assertRaises(ValueError):
                writer.record_training_snapshot(
                    {
                        "iteration": 1,
                        "total_simulated_steps": 100,
                        "wall_time_s": 1.0,
                    },
                    {
                        "invalid_reward": {
                            "raw": float("nan"),
                            "weight": 1.0,
                        },
                    },
                )

            self.assertFalse(writer.training_path.exists())
            self.assertFalse(writer.reward_path.exists())

    def test_nonfinite_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RunMetricsWriter(
                Path(directory),
                "invalid_test",
            )

            with self.assertRaises(ValueError):
                writer.record_training_snapshot(
                    {
                        "iteration": 1,
                        "total_simulated_steps": 100,
                        "wall_time_s": float("nan"),
                    },
                    {},
                )

            self.assertFalse(writer.training_path.exists())
            self.assertFalse(writer.reward_path.exists())


if __name__ == "__main__":
    unittest.main()
