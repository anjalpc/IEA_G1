"""Reward-independent stateful stair-event diagnostic tests."""

import importlib.util
import unittest

from simulation.isaac_lab.g1_stairs.evaluation.stair_diagnostics import (
    StairEventDiagnostics,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(
    HAS_TORCH,
    "CPU PyTorch unavailable",
)
class StairEventDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

        self.events = StairEventDiagnostics(
            2,
            "cpu",
            first_riser_m=1.0,
            tread_m=0.25,
            height_m=0.06,
            takeoff_distance_m=0.32,
        )

        self.payload = {
            "foot_x_min": torch.tensor(
                [
                    [0.20, 0.20],
                    [0.20, 0.20],
                ]
            ),
            "foot_x_max": torch.tensor(
                [
                    [0.30, 0.30],
                    [0.30, 0.30],
                ]
            ),
            "sole_bottom_min": torch.zeros(
                2,
                2,
            ),
            "sole_bottom_max": torch.zeros(
                2,
                2,
            ),
            "foot_normal_z": torch.ones(
                2,
                2,
            ),
            "stable_upper": torch.zeros(
                2,
                2,
                dtype=torch.bool,
            ),
            "swing_ok": torch.zeros(
                2,
                2,
                dtype=torch.bool,
            ),
            "distance_to_riser_face": torch.tensor(
                [
                    [0.70, 0.70],
                    [0.70, 0.70],
                ]
            ),
            "contact_onset": torch.zeros(
                2,
                2,
                dtype=torch.bool,
            ),
            "near_upper": torch.zeros(
                2,
                2,
                dtype=torch.bool,
            ),
            "precontact_velocity": torch.zeros(
                2,
                2,
                3,
            ),
        }

    def update(self):
        self.events.update(
            self.payload,
            dt=0.02,
        )

    def test_funnel_records_premature_lift_swing_crossing_touchdown_support(self):
        # Baseline far approach.
        self.update()

        # Environment 0, left foot: premature high lift while 0.40 m away.
        self.payload[
            "distance_to_riser_face"
        ][0, 0] = 0.40

        self.payload[
            "sole_bottom_max"
        ][0, 0] = 0.10

        self.update()

        # Enter foot-based takeoff region and valid swing.
        self.payload[
            "distance_to_riser_face"
        ][0, 0] = 0.25

        self.payload[
            "swing_ok"
        ][0, 0] = True

        self.payload[
            "sole_bottom_min"
        ][0, 0] = 0.07

        self.update()

        # Cross the riser with 3 cm physical sole clearance.
        self.payload[
            "foot_x_max"
        ][0, 0] = 1.01

        self.payload[
            "sole_bottom_min"
        ][0, 0] = 0.09

        self.payload[
            "sole_bottom_max"
        ][0, 0] = 0.10

        self.update()

        # First upper touchdown.
        self.payload[
            "contact_onset"
        ][0, 0] = True

        self.payload[
            "near_upper"
        ][0, 0] = True

        self.payload[
            "foot_x_min"
        ][0, 0] = 1.02

        self.payload[
            "foot_x_max"
        ][0, 0] = 1.18

        self.payload[
            "foot_normal_z"
        ][0, 0] = 0.97

        self.payload[
            "precontact_velocity"
        ][0, 0] = self.torch.tensor(
            [0.30, 0.40, -0.35]
        )

        self.update()

        # Contact onset is an event, not a persistent state.
        self.payload[
            "contact_onset"
        ][0, 0] = False

        # Five 20 ms steps after touchdown.
        for _ in range(5):
            self.update()

        # Sixth step establishes stable upper support:
        # touchdown -> support latency should be 0.12 s.
        self.payload[
            "stable_upper"
        ][0, 0] = True

        self.update()

        counts, stats = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            counts["near_riser_entry_count"],
            1,
        )

        self.assertEqual(
            counts[
                "left_premature_high_lift_count"
            ],
            1,
        )

        self.assertEqual(
            counts["left_swing_entry_count"],
            1,
        )

        self.assertEqual(
            counts["left_riser_crossing_count"],
            1,
        )

        self.assertEqual(
            counts["left_upper_touchdown_count"],
            1,
        )

        self.assertEqual(
            counts[
                "left_stable_upper_entry_count"
            ],
            1,
        )

        self.assertAlmostEqual(
            stats[
                "left_premature_high_lift_distance_m"
            ]["mean"],
            0.40,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_swing_entry_distance_m"
            ]["mean"],
            0.25,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_riser_crossing_clearance_m"
            ]["mean"],
            0.03,
            places=5,
        )

        # sqrt(0.3^2 + 0.4^2) = 0.5 m/s
        self.assertAlmostEqual(
            stats[
                "left_touchdown_precontact_horizontal_speed_mps"
            ]["mean"],
            0.50,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_touchdown_precontact_vertical_speed_mps"
            ]["mean"],
            -0.35,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_touchdown_rear_margin_m"
            ]["mean"],
            0.02,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_touchdown_front_margin_m"
            ]["mean"],
            0.07,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_support_acquisition_latency_s"
            ]["mean"],
            0.12,
            places=5,
        )

    def test_iteration_consume_does_not_rearm_episode_events(self):
        self.payload[
            "distance_to_riser_face"
        ][0, 0] = 0.20

        self.payload[
            "swing_ok"
        ][0, 0] = True

        self.update()

        first_counts, _ = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            first_counts[
                "near_riser_entry_count"
            ],
            1,
        )

        self.assertEqual(
            first_counts[
                "left_swing_entry_count"
            ],
            1,
        )

        # Same physical episode, next PPO iteration.
        self.update()

        second_counts, second_stats = (
            self.events.consume_iteration()
        )

        # Event keys remain explicit with zero counts. This keeps the
        # downstream training/report schema stable across iterations while
        # proving that consuming iteration statistics did not rearm any
        # episode-local first-occurrence event.
        self.assertTrue(
            second_counts,
        )

        self.assertTrue(
            all(
                value == 0
                for value in second_counts.values()
            )
        )

        self.assertEqual(
            second_stats,
            {},
        )

    def test_episode_reset_rearms_only_selected_environment(self):
        self.payload[
            "distance_to_riser_face"
        ][:, 0] = 0.20

        self.update()

        counts, _ = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            counts[
                "near_riser_entry_count"
            ],
            2,
        )

        # Reset only environment 0.
        done = self.torch.tensor(
            [True, False]
        )

        self.events.reset(done)

        # Move both away and return near.
        self.payload[
            "distance_to_riser_face"
        ][:, 0] = 0.70

        self.update()

        self.payload[
            "distance_to_riser_face"
        ][:, 0] = 0.20

        self.update()

        counts, _ = (
            self.events.consume_iteration()
        )

        # Only reset environment 0 may mint another first-entry event.
        self.assertEqual(
            counts[
                "near_riser_entry_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
