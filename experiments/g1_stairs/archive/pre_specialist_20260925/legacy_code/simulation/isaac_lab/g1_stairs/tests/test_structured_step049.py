"""Adversarial contracts for 049 clearance-first stair acquisition."""

import unittest
import torch

from simulation.isaac_lab.g1_stairs import structured_step044_runtime as runtime
from simulation.isaac_lab.g1_stairs.structured_step044_config import (
    StructuredStep044Config,
)

MODE5_WEIGHTS = {
    "step044_approach": 0.05,
    "step044_lift_advance": 0.75,
    "step044_maintained_clearance_advance": 0.70,
    "step044_meaningful_lift": 0.20,
    "step044_riser_ready": 0.70,
    "step044_valid_crossing": 2.50,
    "step044_tread_insertion": 1.00,
    "step044_good_touchdown": 2.00,
    "step044_stable_upper_support": 4.00,
    "step044_supported_body_rise": 1.00,
    "step044_body_transfer": 4.00,
    "step044_both_feet_supported": 3.00,
    "step044_complete": 5.00,
    "step044_failure": -3.00,
}


class StructuredStep049StateTests(unittest.TestCase):

    def setUp(self):
        self.c = StructuredStep044Config(
            reward_mode=5,
            ascent_launch_distance_m=0.30,
            terminal_stability_dwell_s=0.06,
        )
        self.state = runtime.StructuredStep044State(
            1,
            "cpu",
            self.c,
            reward_weights=MODE5_WEIGHTS,
        )
        self.frame = {
            "root": torch.tensor([[0.0, 0.0, 0.80]], dtype=torch.float32),
            "gravity": torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            "x_min": torch.tensor([[-0.055, -0.055]], dtype=torch.float32),
            "x_max": torch.tensor([[0.125, 0.125]], dtype=torch.float32),
            "sole_bottom": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "normal_z": torch.ones((1, 2), dtype=torch.float32),
            "loaded": torch.ones((1, 2), dtype=torch.bool),
            "stable_lower": torch.ones((1, 2), dtype=torch.bool),
            "stable_upper": torch.zeros((1, 2), dtype=torch.bool),
            "upper_geometry": torch.zeros((1, 2), dtype=torch.bool),
            "landing_rectangle": torch.ones((1, 2), dtype=torch.bool),
            "contact": torch.ones((1, 2), dtype=torch.bool),
            "foot_velocity": torch.zeros((1, 2, 3), dtype=torch.float32),
            "forbidden": torch.zeros(1, dtype=torch.bool),
            "forbidden_contacts": torch.zeros((1, 8), dtype=torch.bool),
            "root_linear_velocity": torch.zeros((1, 3), dtype=torch.float32),
            "root_angular_velocity": torch.zeros((1, 3), dtype=torch.float32),
            "root_yaw": torch.zeros(1, dtype=torch.float32),
            "foot_lateral_max": torch.full((1, 2), 0.20, dtype=torch.float32),
        }

    def step(self):
        return self.state.update(0.02, **self.frame)

    def launch(self, baseline_left=0.0):
        self.step()  # safe calibration
        self.frame["root"][0, 0] = 0.70
        self.frame["root_linear_velocity"][0, 0] = 0.12
        self.frame["sole_bottom"][0, 0] = baseline_left
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["ascent_start_event"][0]))
        return result

    def make_left_swing(self):
        self.frame["loaded"][0, 0] = False
        self.frame["contact"][0, 0] = False
        self.frame["stable_lower"][0, 0] = False
        self.frame["stable_lower"][0, 1] = True
        return self.step()

    def test_prelaunch_high_foot_is_baselined_not_back_credited(self):
        baseline = 0.070
        self.launch(baseline_left=baseline)
        self.make_left_swing()
        same_height = self.step()
        self.assertAlmostEqual(
            float(same_height["components"]["lift_advance"][0]),
            0.0,
            places=7,
        )
        expected = min(baseline / self.c.required_clearance_m, 1.0)
        self.assertAlmostEqual(
            float(self.state.clearance_baseline_fraction[0, 0]),
            expected,
            places=6,
        )

    def test_low_forward_motion_after_launch_has_no_positive_stair_progress(self):
        self.launch()
        self.make_left_swing()

        # A small NEW vertical improvement is allowed to earn clearance credit.
        self.frame["x_max"][0, 0] = 0.82
        self.frame["x_min"][0, 0] = 0.64
        self.frame["sole_bottom"][0, 0] = 0.01
        vertical = self.step()
        self.assertGreater(float(vertical["components"]["lift_advance"][0]), 0.0)

        # Moving the same still-low foot forward must add no task credit.
        self.frame["x_max"][0, 0] = 0.90
        self.frame["x_min"][0, 0] = 0.72
        result = self.step()
        self.assertEqual(float(result["components"]["approach"][0]), 0.0)
        self.assertEqual(float(result["components"]["lift_advance"][0]), 0.0)
        self.assertEqual(
            float(result["components"]["maintained_clearance_advance"][0]),
            0.0,
        )

    def test_vertical_clearance_potential_is_bounded_and_not_repeatable(self):
        self.launch()
        self.make_left_swing()

        self.frame["sole_bottom"][0, 0] = 0.040
        first = self.step()
        first_credit = float(first["components"]["lift_advance"][0])
        self.assertGreater(first_credit, 0.0)

        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        second = self.step()
        second_credit = float(second["components"]["lift_advance"][0])
        self.assertGreater(second_credit, 0.0)
        self.assertLessEqual(first_credit + second_credit, 1.0 + 1.0e-6)

        self.frame["sole_bottom"][0, 0] = 0.02
        self.step()
        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        replay = self.step()
        self.assertAlmostEqual(
            float(replay["components"]["lift_advance"][0]),
            0.0,
            places=7,
        )

    def test_full_clearance_event_unlocks_only_future_maintained_advance(self):
        self.launch()
        self.make_left_swing()
        self.frame["sole_bottom"][0, 0] = 0.040
        self.step()  # SWING -> LIFTED

        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        self.frame["x_max"][0, 0] = 0.82
        acquired = self.step()
        self.assertTrue(
            bool(acquired["diagnostics"]["clearance_acquired_event"][0, 0])
        )
        self.assertEqual(
            float(acquired["components"]["maintained_clearance_advance"][0]),
            0.0,
        )

        self.frame["x_max"][0, 0] = 0.84
        advanced = self.step()
        self.assertGreater(
            float(advanced["components"]["maintained_clearance_advance"][0]),
            0.0,
        )

    def test_low_crossing_is_immediate_structured_failure(self):
        self.launch()
        self.make_left_swing()
        self.state.foot_stage[0, 0] = runtime.SWING
        self.state.has_previous[0] = True
        self.state.previous_x_max[0, 0] = 0.99
        self.frame["x_max"][0, 0] = 1.001
        self.frame["x_min"][0, 0] = 0.82
        self.frame["sole_bottom"][0, 0] = 0.02
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["low_crossing_event"][0, 0]))
        self.assertTrue(bool(result["diagnostics"]["failure_low_crossing"][0]))
        self.assertTrue(bool(result["failure"][0]))
        self.assertFalse(bool(result["success"][0]))

    def test_lift_then_low_crossing_remains_net_negative_with_clawback(self):
        self.launch()
        self.make_left_swing()
        self.state.positive_stair_credit[:] = 1.0
        self.state.foot_stage[0, 0] = runtime.SWING
        self.state.has_previous[0] = True
        self.state.previous_x_max[0, 0] = 0.99
        self.frame["x_max"][0, 0] = 1.001
        self.frame["sole_bottom"][0, 0] = 0.02
        result = self.step()
        raw_failure = float(result["components"]["failure"][0])
        weighted_failure = raw_failure * MODE5_WEIGHTS["step044_failure"]
        self.assertLess(weighted_failure, -4.0)

    def test_forbidden_contact_attribution_uses_actual_body_groups(self):
        self.launch()
        # order: pelvis, torso, L/R knee, L/R elbow, L/R hand
        self.frame["forbidden_contacts"][0, 2] = True
        self.frame["forbidden"][0] = True
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["failure_forbidden_knee"][0]))
        self.assertFalse(bool(result["diagnostics"]["failure_forbidden_pelvis"][0]))
        self.assertFalse(bool(result["diagnostics"]["failure_forbidden_hand"][0]))

    def test_reset_clears_049_clearance_state(self):
        self.launch()
        self.state.clearance_baseline_fraction[:] = 0.4
        self.state.clearance_acquired_seen[:] = True
        self.state.reset(torch.tensor([0], dtype=torch.long))
        self.assertTrue(
            torch.equal(
                self.state.clearance_baseline_fraction,
                torch.zeros_like(self.state.clearance_baseline_fraction),
            )
        )
        self.assertFalse(bool(self.state.clearance_acquired_seen.any()))


if __name__ == "__main__":
    unittest.main()
