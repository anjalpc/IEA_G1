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
from simulation.isaac_lab.g1_stairs.evaluation.isaac_training_metrics import (
    recover_raw_reward_step,
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
