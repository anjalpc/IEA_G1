import csv
import json
import tempfile
import unittest
from pathlib import Path

from simulation.isaac_lab.g1_stairs.scripts.diagnostics.build_run_diagnostic_report import (
    build_summary,
    render_markdown,
)


def write_csv(path, fieldnames, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


class DiagnosticReportTest(unittest.TestCase):
    def test_detects_touchdown_to_support_bottleneck(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)

            write_csv(
                run / "training_metrics.csv",
                [
                    "iteration",
                    "total_simulated_steps",
                    "episode_return_mean",
                    "extra_json",
                ],
                [
                    {
                        "iteration": 10,
                        "total_simulated_steps": 10000,
                        "episode_return_mean": 1.5,
                        "extra_json": json.dumps(
                            {
                                "approach_teacher_loss": 0.01,
                            }
                        ),
                    }
                ],
            )

            write_csv(
                run / "reward_diagnostics.csv",
                [
                    "iteration",
                    "total_simulated_steps",
                    "name",
                    "sample_count",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "nonzero_fraction",
                    "positive_fraction",
                    "negative_fraction",
                    "weight",
                    "weighted_mean",
                    "weighted_abs_mean",
                ],
                [
                    {
                        "iteration": 10,
                        "total_simulated_steps": 10000,
                        "name": "step043_clearance",
                        "sample_count": 100,
                        "mean": 0.2,
                        "std": 0.1,
                        "min": 0,
                        "max": 1,
                        "nonzero_fraction": 0.5,
                        "positive_fraction": 0.5,
                        "negative_fraction": 0,
                        "weight": 0.25,
                        "weighted_mean": 0.05,
                        "weighted_abs_mean": 0.05,
                    },
                    {
                        "iteration": 10,
                        "total_simulated_steps": 10000,
                        "name": "step043_complete",
                        "sample_count": 100,
                        "mean": 0,
                        "std": 0,
                        "min": 0,
                        "max": 0,
                        "nonzero_fraction": 0,
                        "positive_fraction": 0,
                        "negative_fraction": 0,
                        "weight": 5,
                        "weighted_mean": 0,
                        "weighted_abs_mean": 0,
                    },
                ],
            )

            write_csv(
                run / "stair_diagnostics.csv",
                [
                    "iteration",
                    "total_simulated_steps",
                    "name",
                    "sample_count",
                    "mean",
                    "std",
                    "min",
                    "max",
                ],
                [],
            )

            write_csv(
                run / "stair_event_counts.csv",
                [
                    "iteration",
                    "total_simulated_steps",
                    "name",
                    "count",
                ],
                [
                    {
                        "iteration": 10,
                        "total_simulated_steps": 10000,
                        "name": "left_upper_touchdown_count",
                        "count": 8,
                    }
                ],
            )

            funnel = {
                "episodes_finished_count": 10,
                "episodes_near_riser_count": 10,
                "episodes_swing_count": 9,
                "episodes_riser_crossing_count": 8,
                "episodes_upper_touchdown_count": 8,
                "episodes_stable_upper_count": 2,
                "episodes_first_support_count": 2,
                "episodes_base_transfer_count": 1,
                "episodes_both_feet_count": 0,
                "episodes_completion_count": 0,
            }

            episode_rows = [
                {
                    "iteration": 10,
                    "total_simulated_steps": 10000,
                    "name": name,
                    "count": value,
                }
                for name, value in funnel.items()
            ]

            episode_rows.extend(
                [
                    {
                        "iteration": 10,
                        "total_simulated_steps": 10000,
                        "name": "episodes_premature_high_lift_count",
                        "count": 3,
                    },
                    {
                        "iteration": 10,
                        "total_simulated_steps": 10000,
                        "name": "terminal_success_count",
                        "count": 0,
                    },
                    {
                        "iteration": 10,
                        "total_simulated_steps": 10000,
                        "name": "terminal_physical_failure_after_upper_touchdown_count",
                        "count": 10,
                    },
                ]
            )

            write_csv(
                run / "stair_episode_counts.csv",
                [
                    "iteration",
                    "total_simulated_steps",
                    "name",
                    "count",
                ],
                episode_rows,
            )

            summary = build_summary(run)

            bottleneck = summary[
                "episodes"
            ]["bottleneck"]

            self.assertEqual(
                bottleneck["from"],
                "episodes_upper_touchdown_count",
            )

            self.assertEqual(
                bottleneck["to"],
                "episodes_stable_upper_count",
            )

            self.assertIn(
                "step043_complete",
                summary[
                    "rewards"
                ]["dead_terms"],
            )

            self.assertEqual(
                summary[
                    "reward_physics_alignment"
                ]["status"],
                "no_verified_completion",
            )

            report = render_markdown(
                summary
            )

            self.assertIn(
                "upper touchdown",
                report,
            )

            self.assertIn(
                "stable upper support",
                report,
            )

    def test_zero_finished_episodes_do_not_invent_bottleneck(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)

            write_csv(
                run / "stair_episode_counts.csv",
                [
                    "iteration",
                    "total_simulated_steps",
                    "name",
                    "count",
                ],
                [
                    {
                        "iteration": 0,
                        "total_simulated_steps": 768,
                        "name": name,
                        "count": 0,
                    }
                    for name in FUNNEL_NAMES
                ],
            )

            summary = build_summary(run)

            self.assertEqual(
                summary[
                    "episodes"
                ]["bottleneck"]["status"],
                "insufficient_completed_episodes",
            )


FUNNEL_NAMES = (
    "episodes_finished_count",
    "episodes_near_riser_count",
    "episodes_swing_count",
    "episodes_riser_crossing_count",
    "episodes_upper_touchdown_count",
    "episodes_stable_upper_count",
    "episodes_first_support_count",
    "episodes_base_transfer_count",
    "episodes_both_feet_count",
    "episodes_completion_count",
)


if __name__ == "__main__":
    unittest.main()
