"""Adversarial contracts for 050 locomotion-preserving clearance acquisition."""

import unittest
import torch

from simulation.isaac_lab.g1_stairs import structured_step044_runtime as runtime
from simulation.isaac_lab.g1_stairs.structured_step044_config import (
    StructuredStep044Config,
)

MODE6_WEIGHTS = {
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


class StructuredStep050StateTests(unittest.TestCase):

    def setUp(self):
        self.c = StructuredStep044Config(
            reward_mode=6,
            ascent_launch_distance_m=0.30,
            terminal_stability_dwell_s=0.06,
        )
        self.state = runtime.StructuredStep044State(
            1,
            "cpu",
            self.c,
            reward_weights=MODE6_WEIGHTS,
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

    def launch(self, left_height=0.0):
        self.step()
        self.frame["root"][0, 0] = 0.70
        self.frame["root_linear_velocity"][0, 0] = 0.12
        self.frame["sole_bottom"][0, 0] = left_height
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["ascent_start_event"][0]))
        return result

    def enter_left_swing(self, height=None):
        if height is not None:
            self.frame["sole_bottom"][0, 0] = height
        self.frame["loaded"][0, 0] = False
        self.frame["contact"][0, 0] = False
        self.frame["stable_lower"][0, 0] = False
        self.frame["stable_lower"][0, 1] = True
        result = self.step()
        self.assertEqual(int(self.state.foot_stage[0, 0]), runtime.SWING)
        return result

    def test_clearance_baseline_is_captured_at_swing_entry_not_launch(self):
        self.launch(left_height=0.020)
        # Flat-gait phase changes before this foot actually becomes swing.
        self.frame["sole_bottom"][0, 0] = 0.030
        self.enter_left_swing()
        expected = 0.030 / self.c.required_clearance_m
        self.assertAlmostEqual(
            float(self.state.clearance_baseline_fraction[0, 0]),
            expected,
            places=6,
        )
        same = self.step()
        self.assertAlmostEqual(float(same["components"]["lift_advance"][0]), 0.0, places=7)
        self.frame["sole_bottom"][0, 0] = 0.040
        raised = self.step()
        self.assertGreater(float(raised["components"]["lift_advance"][0]), 0.0)

    def test_stable_dwell_flicker_does_not_block_latched_swing_lift(self):
        self.launch()
        self.enter_left_swing()
        # The stance foot remains physically loaded but no longer satisfies the
        # 0.12 s stable-support dwell on this exact dynamic frame.
        self.frame["stable_lower"][0, 1] = False
        self.frame["loaded"][0, 1] = True
        self.frame["contact"][0, 1] = True
        self.frame["sole_bottom"][0, 0] = 0.040
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["meaningful_lift_event"][0, 0]))
        self.assertEqual(int(self.state.foot_stage[0, 0]), runtime.LIFTED)

    def test_opposite_foot_must_remain_loaded_so_jump_cannot_earn_lift(self):
        self.launch()
        self.enter_left_swing()
        self.frame["stable_lower"][0, 1] = False
        self.frame["loaded"][0, 1] = False
        self.frame["contact"][0, 1] = False
        self.frame["sole_bottom"][0, 0] = 0.040
        result = self.step()
        self.assertFalse(bool(result["diagnostics"]["meaningful_lift_event"][0, 0]))
        self.assertEqual(int(self.state.foot_stage[0, 0]), runtime.SWING)

    def test_full_clearance_can_be_acquired_during_dynamic_loaded_support(self):
        self.launch()
        self.enter_left_swing()
        self.frame["stable_lower"][0, 1] = False
        self.frame["loaded"][0, 1] = True
        self.frame["sole_bottom"][0, 0] = 0.040
        self.step()  # SWING -> LIFTED
        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["clearance_acquired_event"][0, 0]))

    def test_low_crossing_remains_immediate_terminal_failure(self):
        self.launch()
        self.enter_left_swing()
        self.state.has_previous[0] = True
        self.state.previous_x_max[0, 0] = 0.99
        self.frame["x_max"][0, 0] = 1.001
        self.frame["x_min"][0, 0] = 0.82
        self.frame["sole_bottom"][0, 0] = 0.02
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["failure_low_crossing"][0]))
        self.assertTrue(bool(result["failure"][0]))

    def test_mode5_historical_transition_still_requires_stable_support(self):
        c5 = StructuredStep044Config(
            reward_mode=5,
            ascent_launch_distance_m=0.30,
            terminal_stability_dwell_s=0.06,
        )
        s5 = runtime.StructuredStep044State(1, "cpu", c5, reward_weights=MODE6_WEIGHTS)
        # Reuse the same frame contract manually.
        f = {k: v.clone() for k, v in self.frame.items()}
        s5.update(0.02, **f)
        f["root"][0, 0] = 0.70
        f["root_linear_velocity"][0, 0] = 0.12
        s5.update(0.02, **f)
        f["loaded"][0, 0] = False
        f["contact"][0, 0] = False
        f["stable_lower"][0, 0] = False
        f["stable_lower"][0, 1] = True
        s5.update(0.02, **f)  # enter SWING
        f["stable_lower"][0, 1] = False
        f["sole_bottom"][0, 0] = 0.040
        result = s5.update(0.02, **f)
        self.assertFalse(bool(result["diagnostics"]["meaningful_lift_event"][0, 0]))


if __name__ == "__main__":
    unittest.main()
