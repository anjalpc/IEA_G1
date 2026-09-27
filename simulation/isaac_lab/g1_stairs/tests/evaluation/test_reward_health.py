"""Tests for the read-only mid-run reward health card."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / "simulation/isaac_lab/g1_stairs/scripts/diagnostics/reward_health.py"


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RewardHealthTest(unittest.TestCase):
    def test_health_card_reports_support_and_downstream_state(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "g1_stair_v08_test"
            run.mkdir()

            reward_fields = (
                "iteration", "name", "nonzero_fraction", "weighted_mean"
            )
            reward_rows = []
            for it in range(10):
                reward_rows.extend(
                    [
                        {
                            "iteration": it,
                            "name": "support_foot_anchor_guard",
                            "nonzero_fraction": 0.3,
                            "weighted_mean": -0.08 + 0.006 * it,
                        },
                        {
                            "iteration": it,
                            "name": "feet_slide",
                            "nonzero_fraction": 0.2,
                            "weighted_mean": -0.01,
                        },
                        {
                            "iteration": it,
                            "name": "swing_waypoint_progress",
                            "nonzero_fraction": 0.4,
                            "weighted_mean": 0.10,
                        },
                        {
                            "iteration": it,
                            "name": "swing_body_advance_progress",
                            "nonzero_fraction": 0.02 if it >= 7 else 0.0,
                            "weighted_mean": 0.01 if it >= 7 else 0.0,
                        },
                        {
                            "iteration": it,
                            "name": "upper_foothold_progress",
                            "nonzero_fraction": 0.001 if it == 9 else 0.0,
                            "weighted_mean": 0.0,
                        },
                        {
                            "iteration": it,
                            "name": "body_transfer_progress",
                            "nonzero_fraction": 0.0,
                            "weighted_mean": 0.0,
                        },
                        {
                            "iteration": it,
                            "name": "single_step_success",
                            "nonzero_fraction": 0.0,
                            "weighted_mean": 0.0,
                        },
                        {
                            "iteration": it,
                            "name": "dof_pos_limits",
                            "nonzero_fraction": 0.05,
                            "weighted_mean": -0.01,
                        },
                    ]
                )
            write_csv(run / "reward_diagnostics.csv", reward_fields, reward_rows)

            diag_fields = ("iteration", "name", "mean")
            diag_rows = []
            for it in range(10):
                diag_rows.extend(
                    [
                        {"iteration": it, "name": "support_lock_anchor_error_m", "mean": 0.055 - 0.004 * it},
                        {"iteration": it, "name": "support_lock_planar_speed_mps", "mean": 0.15 - 0.01 * it},
                        {"iteration": it, "name": "support_lock_gate", "mean": 0.35},
                        {"iteration": it, "name": "swing_lateral_abs_error_m", "mean": 0.05 - 0.002 * it},
                        {"iteration": it, "name": "swing_lower_floor_recontact", "mean": 0.08},
                        {"iteration": it, "name": "motion_phase_is_0", "mean": 0.05},
                        {"iteration": it, "name": "motion_phase_is_1", "mean": 0.10},
                        {"iteration": it, "name": "motion_phase_is_2", "mean": 0.20},
                        {"iteration": it, "name": "motion_phase_is_3", "mean": 0.55},
                        {"iteration": it, "name": "motion_phase_is_4", "mean": 0.10},
                        {"iteration": it, "name": "motion_phase_is_complete", "mean": 0.0},
                    ]
                )
            write_csv(run / "stair_diagnostics.csv", diag_fields, diag_rows)

            metric_fields = (
                "iteration", "episode_return_mean", "fall_rate", "action_rate_mean",
                "mechanical_power_mean", "learning_rate", "entropy", "value_loss",
                "policy_loss", "approximate_kl", "clip_fraction", "nan_count",
                "maximum_x_any_m", "maximum_step_reached_any", "platform_reached_rate",
            )
            metric_rows = [
                {
                    "iteration": it,
                    "episode_return_mean": -5 + 0.1 * it,
                    "fall_rate": 0.0,
                    "action_rate_mean": 1.0,
                    "mechanical_power_mean": 210.0,
                    "learning_rate": 5e-5,
                    "entropy": 22.0,
                    "value_loss": 0.2,
                    "policy_loss": -0.005,
                    "approximate_kl": "",
                    "clip_fraction": "",
                    "nan_count": 0,
                    "maximum_x_any_m": 0.2,
                    "maximum_step_reached_any": 0,
                    "platform_reached_rate": 0.0,
                }
                for it in range(10)
            ]
            write_csv(run / "training_metrics.csv", metric_fields, metric_rows)

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(run)],
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout
            self.assertIn("G1 STAIR REWARD HEALTH", output)
            self.assertIn("SUPPORT LOCK", output)
            self.assertIn("improving", output)
            self.assertIn("upper foothold", output)
            self.assertIn("EMERGING", output)
            self.assertIn("PPO / SAFETY", output)

    def test_json_mode_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "empty_run"
            run.mkdir()
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(run), "--json"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"support"', result.stdout)
            self.assertIn('"ppo"', result.stdout)


if __name__ == "__main__":
    unittest.main()


def test_health_reads_balance_residual_telemetry(tmp_path):
    import csv
    import json
    from simulation.isaac_lab.g1_stairs.scripts.diagnostics.reward_health import build_health

    path = tmp_path / "training_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["iteration", "extra_json"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "iteration": 3,
                "extra_json": json.dumps(
                    {
                        "balance_teacher_action_abs_mean": 0.42,
                        "balance_residual_action_abs_mean": 0.08,
                        "balance_executed_action_abs_mean": 0.44,
                    }
                ),
            }
        )

    health = build_health(tmp_path)
    residual = health["residual_control"]
    assert residual["active"] is True
    assert residual["teacher_action_abs_mean"] == 0.42
    assert residual["residual_action_abs_mean"] == 0.08
    assert residual["executed_action_abs_mean"] == 0.44
