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
            "foot_velocity": torch.zeros(
                2,
                2,
                3,
            ),
            "root": torch.tensor(
                [
                    [0.50, 0.00, 0.80],
                    [0.50, 0.00, 0.80],
                ]
            ),
            "upright": torch.tensor(
                [True, True]
            ),
            "safe": torch.tensor(
                [True, True]
            ),
            "forbidden": torch.tensor(
                [False, False]
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

        self.payload[
            "foot_x_min"
        ][0, 0] = 0.84

        self.payload[
            "foot_normal_z"
        ][0, 0] = 0.98

        self.payload[
            "foot_velocity"
        ][0, 0] = self.torch.tensor(
            [0.30, 0.40, 0.20]
        )

        self.payload[
            "root"
        ][0, 2] = 0.82

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

        counts, stats, episode_counts = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            episode_counts,
            {},
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

        self.assertAlmostEqual(
            stats[
                "left_crossing_root_height_m"
            ]["mean"],
            0.82,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_crossing_upright_fraction"
            ]["mean"],
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_crossing_safe_fraction"
            ]["mean"],
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_crossing_forbidden_fraction"
            ]["mean"],
            0.0,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_crossing_foot_normal_z"
            ]["mean"],
            0.98,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_crossing_rear_margin_m"
            ]["mean"],
            -0.16,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_crossing_front_overshoot_m"
            ]["mean"],
            0.01,
            places=5,
        )

        # sqrt(0.3^2 + 0.4^2) = 0.5 m/s
        self.assertAlmostEqual(
            stats[
                "left_crossing_horizontal_speed_mps"
            ]["mean"],
            0.50,
            places=5,
        )

        self.assertAlmostEqual(
            stats[
                "left_crossing_vertical_speed_mps"
            ]["mean"],
            0.20,
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

        first_counts, _, first_episode_counts = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            first_episode_counts,
            {},
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

        (
            second_counts,
            second_stats,
            second_episode_counts,
        ) = self.events.consume_iteration()

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

        self.assertEqual(
            second_episode_counts,
            {},
        )

    def test_terminal_funnel_and_failure_progress_are_classified_before_reset(self):
        # Environment 0 reaches the near-riser and swing milestones.
        self.payload[
            "distance_to_riser_face"
        ][0, 0] = 0.20

        self.payload[
            "swing_ok"
        ][0, 0] = True

        self.update()

        # Environment 0 then physically fails. Environment 1 times out
        # without reaching any stair milestone.
        dones = self.torch.tensor(
            [True, True]
        )

        physical_failure = self.torch.tensor(
            [True, False]
        )

        physical_success = self.torch.tensor(
            [False, False]
        )

        stage = self.torch.tensor(
            [0, 0],
            dtype=self.torch.long,
        )

        timeouts = self.torch.tensor(
            [False, True]
        )

        self.events.observe_terminations(
            dones,
            physical_failure=physical_failure,
            physical_success=physical_success,
            stage=stage,
            timeouts=timeouts,
        )

        # Classification must happen before episode reset.
        self.events.reset(dones)

        _, _, episode_counts = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            episode_counts[
                "episodes_finished_count"
            ],
            2,
        )

        self.assertEqual(
            episode_counts[
                "episodes_near_riser_count"
            ],
            1,
        )

        self.assertEqual(
            episode_counts[
                "episodes_swing_count"
            ],
            1,
        )

        self.assertEqual(
            episode_counts[
                "episodes_riser_crossing_count"
            ],
            0,
        )

        self.assertEqual(
            episode_counts[
                "terminal_physical_failure_after_swing_count"
            ],
            1,
        )

        self.assertEqual(
            episode_counts[
                "terminal_timeout_after_approach_only_count"
            ],
            1,
        )

        self.assertEqual(
            episode_counts[
                "terminal_success_count"
            ],
            0,
        )

        # Cause/progress bins must account for every finished,
        # non-success episode exactly once.
        classified = 0

        for cause in self.events.TERMINAL_CAUSES:
            for progress in self.events.PROGRESS_LABELS:
                classified += episode_counts[
                    (
                        f"terminal_{cause}_after_"
                        f"{progress}_count"
                    )
                ]

        self.assertEqual(
            classified,
            2,
        )

    def test_full_length_integer_done_vector_is_rejected_by_reset(self):
        dones = self.torch.tensor(
            [1, 0],
            dtype=self.torch.long,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "integer 0/1 reset vector",
        ):
            self.events.reset(dones)

    def test_boolean_done_mask_rearms_only_completed_environment(self):
        self.payload[
            "swing_ok"
        ][:, 0] = True

        self.update()

        counts, _, _ = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            counts[
                "left_swing_entry_count"
            ],
            2,
        )

        # Runtime RSL-RL dones are long, but the bridge must convert
        # them to bool before reset.
        runtime_dones = self.torch.tensor(
            [1, 0],
            dtype=self.torch.long,
        )

        self.events.reset(
            runtime_dones.to(
                dtype=self.torch.bool
            )
        )

        # Leave swing, then enter it again.
        self.payload[
            "swing_ok"
        ][:, 0] = False

        self.update()

        self.payload[
            "swing_ok"
        ][:, 0] = True

        self.update()

        counts, _, _ = (
            self.events.consume_iteration()
        )

        # Only environment 0 was rearmed. Environment 1 must retain
        # its existing episode latch.
        self.assertEqual(
            counts[
                "left_swing_entry_count"
            ],
            1,
        )

    def test_episode_reset_rearms_only_selected_environment(self):
        self.payload[
            "distance_to_riser_face"
        ][:, 0] = 0.20

        self.update()

        counts, _, episode_counts = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            episode_counts,
            {},
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

        counts, _, episode_counts = (
            self.events.consume_iteration()
        )

        self.assertEqual(
            episode_counts,
            {},
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
