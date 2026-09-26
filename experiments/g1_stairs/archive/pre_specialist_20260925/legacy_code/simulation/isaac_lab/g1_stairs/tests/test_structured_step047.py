"""Adversarial counterexample tests for anti-exploit stair mode 047."""

import unittest

import torch

from simulation.isaac_lab.g1_stairs.structured_step044_config import (
    StructuredStep044Config,
)
from simulation.isaac_lab.g1_stairs import structured_step044_runtime as runtime


class StructuredStep047AntiExploitTests(unittest.TestCase):

    def setUp(self):
        self.c = StructuredStep044Config(reward_mode=3)
        self.state = runtime.StructuredStep044State(1, "cpu", self.c)
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
        }
        self.step()  # initial double support calibration

    def step(self):
        return self.state.update(0.02, **self.frame)

    def launch(self):
        self.frame["root"][0, 0] = 0.80
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["ascent_started"][0]))
        return result

    def left_swing(self, x_max=0.90, sole=0.04):
        self.frame["loaded"][0, 0] = False
        self.frame["stable_lower"][0, 0] = False
        self.frame["contact"][0, 0] = False
        self.frame["x_max"][0, 0] = x_max
        self.frame["x_min"][0, 0] = x_max - 0.18
        self.frame["sole_bottom"][0, 0] = sole

    def test_far_high_lift_gets_no_stair_discovery_reward(self):
        self.left_swing(x_max=0.60, sole=0.10)
        self.frame["root"][0, 0] = 0.40
        result = self.step()
        self.assertFalse(bool(result["diagnostics"]["ascent_started"][0]))
        self.assertEqual(float(result["components"]["lift_advance"][0]), 0.0)
        self.assertEqual(float(result["components"]["maintained_clearance_advance"][0]), 0.0)
        self.assertEqual(float(result["components"]["meaningful_lift"][0]), 0.0)

    def test_ascent_permission_latches_after_body_launch(self):
        self.launch()
        self.frame["root"][0, 0] = 1.08
        self.frame["stable_lower"][0, 0] = False
        self.frame["stable_lower"][0, 1] = False
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["ascent_started"][0]))

    def test_trailing_foot_can_start_swing_after_root_passes_riser(self):
        self.launch()
        self.state.foot_stage[0, 0] = runtime.UPPER_SUPPORT
        self.state.seen_upper[0, 0] = True
        self.frame["root"][0, 0] = 1.05
        self.frame["loaded"][0, 0] = True
        self.frame["stable_lower"][0, 0] = False
        self.frame["stable_upper"][0, 0] = True
        self.frame["upper_geometry"][0, 0] = True
        self.frame["loaded"][0, 1] = False
        self.frame["stable_lower"][0, 1] = False
        self.frame["contact"][0, 1] = False
        self.frame["x_min"][0, 1] = 0.62
        self.frame["x_max"][0, 1] = 0.80
        self.frame["sole_bottom"][0, 1] = 0.04
        result = self.step()
        self.assertEqual(int(result["foot_stage"][0, 1]), runtime.SWING)

    def test_trailing_high_lift_after_launch_is_not_marked_premature(self):
        self.launch()
        self.frame["root"][0, 0] = 1.05
        self.frame["loaded"][0, 0] = False
        self.frame["stable_lower"][0, 0] = False
        self.frame["contact"][0, 0] = False
        self.frame["x_min"][0, 0] = 0.52
        self.frame["x_max"][0, 0] = 0.70
        self.frame["sole_bottom"][0, 0] = 0.10
        result = self.step()
        self.assertEqual(float(result["components"]["premature_lift"][0]), 0.0)

    def test_contact_assisted_swing_cannot_collect_lift_or_advance(self):
        self.launch()
        self.left_swing(x_max=0.90, sole=self.c.required_clearance_m)
        self.frame["contact"][0, 0] = True
        result = self.step()
        self.assertEqual(int(result["foot_stage"][0, 0]), runtime.LOWER)
        self.assertEqual(float(result["components"]["lift_advance"][0]), 0.0)
        self.assertEqual(float(result["components"]["maintained_clearance_advance"][0]), 0.0)

    def test_high_lift_in_place_earns_no_maintained_advance(self):
        self.launch()
        self.left_swing(x_max=0.90, sole=self.c.required_clearance_m)
        first = self.step()
        second = self.step()
        self.assertGreater(float(first["components"]["lift_advance"][0]), 0.0)
        self.assertEqual(float(first["components"]["maintained_clearance_advance"][0]), 0.0)
        self.assertEqual(float(second["components"]["maintained_clearance_advance"][0]), 0.0)

    def test_forward_progress_with_continuous_clearance_pays(self):
        self.launch()
        self.left_swing(x_max=0.88, sole=self.c.required_clearance_m)
        self.step()  # LOWER -> SWING; establish high-clearance frontier
        self.step()  # SWING -> LIFTED, still no motion
        self.frame["x_max"][0, 0] = 0.90
        self.frame["x_min"][0, 0] = 0.72
        result = self.step()
        self.assertGreater(
            float(result["components"]["maintained_clearance_advance"][0]),
            0.0,
        )

    def test_bad_orientation_or_lateral_escape_cannot_earn_advance(self):
        self.launch()
        self.left_swing(x_max=0.86, sole=self.c.required_clearance_m)
        self.step()
        self.step()
        self.frame["normal_z"][0, 0] = 0.80
        self.frame["x_max"][0, 0] = 0.90
        self.frame["x_min"][0, 0] = 0.72
        bad_normal = self.step()
        self.assertEqual(float(bad_normal["components"]["maintained_clearance_advance"][0]), 0.0)
        self.frame["normal_z"][0, 0] = 1.0
        self.frame["landing_rectangle"][0, 0] = False
        self.frame["x_max"][0, 0] = 0.92
        self.frame["x_min"][0, 0] = 0.74
        outside = self.step()
        self.assertEqual(float(outside["components"]["maintained_clearance_advance"][0]), 0.0)

    def test_low_advance_then_relift_cannot_back_credit(self):
        self.launch()
        self.left_swing(x_max=0.86, sole=self.c.required_clearance_m)
        self.step()
        self.step()
        # Drop, then move forward while low.
        self.frame["sole_bottom"][0, 0] = 0.02
        self.frame["x_max"][0, 0] = 0.94
        self.frame["x_min"][0, 0] = 0.76
        self.step()
        # Re-lift at the farther x. Rising clearance must only rebase frontier.
        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        relift = self.step()
        self.assertEqual(
            float(relift["components"]["maintained_clearance_advance"][0]),
            0.0,
        )
        # Only new progress after the re-lift can pay.
        self.frame["x_max"][0, 0] = 0.95
        self.frame["x_min"][0, 0] = 0.77
        progressed = self.step()
        self.assertGreater(
            float(progressed["components"]["maintained_clearance_advance"][0]),
            0.0,
        )

    def test_backward_forward_oscillation_does_not_repay_old_frontier(self):
        self.launch()
        self.left_swing(x_max=0.86, sole=self.c.required_clearance_m)
        self.step()
        self.step()
        self.frame["x_max"][0, 0] = 0.90
        self.frame["x_min"][0, 0] = 0.72
        earned = self.step()
        self.assertGreater(float(earned["components"]["maintained_clearance_advance"][0]), 0.0)
        self.frame["x_max"][0, 0] = 0.88
        self.frame["x_min"][0, 0] = 0.70
        self.step()
        self.frame["x_max"][0, 0] = 0.90
        self.frame["x_min"][0, 0] = 0.72
        replay = self.step()
        self.assertEqual(float(replay["components"]["maintained_clearance_advance"][0]), 0.0)

    def test_full_height_crossing_before_riser_ready_is_penalized(self):
        self.launch()
        self.left_swing(x_max=0.98, sole=self.c.required_clearance_m)
        self.step()  # LOWER -> SWING
        self.step()  # SWING -> LIFTED, still not RISER_READY
        self.frame["x_min"][0, 0] = 0.825
        self.frame["x_max"][0, 0] = 1.005
        crossed = self.step()
        self.assertEqual(float(crossed["components"]["valid_crossing"][0]), 0.0)
        self.assertEqual(float(crossed["components"]["low_crossing"][0]), 1.0)

    def test_bad_normal_crossing_is_penalized(self):
        self.launch()
        self.left_swing(x_max=0.95, sole=0.04)
        self.step()
        self.frame["x_max"][0, 0] = 0.97
        self.frame["x_min"][0, 0] = 0.79
        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        self.step()
        self.frame["x_max"][0, 0] = 0.995
        self.frame["x_min"][0, 0] = 0.815
        self.step()  # RISER_READY
        self.frame["normal_z"][0, 0] = 0.80
        self.frame["x_max"][0, 0] = 1.005
        self.frame["x_min"][0, 0] = 0.825
        result = self.step()
        self.assertEqual(float(result["components"]["valid_crossing"][0]), 0.0)
        self.assertEqual(float(result["components"]["low_crossing"][0]), 1.0)

    def test_drop_after_full_clearance_before_riser_ready_is_penalized(self):
        self.launch()
        self.left_swing(x_max=0.88, sole=0.04)
        self.step()  # SWING
        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        self.step()  # LIFTED and full-clearance history established
        self.frame["sole_bottom"][0, 0] = 0.02
        dropped = self.step()
        self.assertEqual(float(dropped["components"]["clearance_loss"][0]), 1.0)

    def test_invalid_slow_touchdown_is_not_free(self):
        self.launch()
        self.state.foot_stage[0, 0] = runtime.CROSSED
        self.state.has_previous[:] = True
        self.state.previous_contact[0, 0] = False
        self.state.previous_velocity[0, 0] = 0.0
        self.frame["contact"][0, 0] = True
        self.frame["upper_geometry"][0, 0] = False
        self.frame["landing_rectangle"][0, 0] = False
        result = self.step()
        self.assertEqual(float(result["components"]["good_touchdown"][0]), 0.0)
        self.assertGreaterEqual(float(result["components"]["bad_touchdown"][0]), 1.0)

    def test_both_feet_reward_requires_both_currently_stable(self):
        self.launch()
        self.state.body_transfer_seen[:] = True
        self.state.seen_upper[:] = True
        self.frame["upper_geometry"][:] = True
        self.frame["stable_upper"][0] = torch.tensor([True, False])
        result = self.step()
        self.assertEqual(float(result["components"]["both_feet_supported"][0]), 0.0)
        self.frame["stable_upper"][:] = True
        result = self.step()
        self.assertEqual(float(result["components"]["both_feet_supported"][0]), 1.0)

    def test_unvalidated_upper_contact_cannot_get_body_transfer_or_rise(self):
        self.launch()
        self.frame["root"][0, 0] = 1.05
        self.frame["root"][0, 2] = 0.86
        self.frame["stable_upper"][0, 0] = True
        self.frame["upper_geometry"][0, 0] = True
        result = self.step()
        self.assertEqual(float(result["components"]["supported_body_rise"][0]), 0.0)
        self.assertEqual(float(result["components"]["body_transfer"][0]), 0.0)

    def test_body_transfer_requires_validated_and_stable_same_foot(self):
        self.launch()
        self.state.seen_upper[0, 0] = True
        self.frame["root"][0, 0] = 1.05
        self.frame["root"][0, 2] = 0.86
        # Only the OTHER, unvalidated foot is physically stable upstairs.
        self.frame["stable_upper"][0] = torch.tensor([False, True])
        self.frame["upper_geometry"][0, 1] = True
        result = self.step()
        self.assertEqual(float(result["components"]["supported_body_rise"][0]), 0.0)
        self.assertEqual(float(result["components"]["body_transfer"][0]), 0.0)

    def test_overlong_insertion_outside_landing_rectangle_gets_no_credit(self):
        self.launch()
        self.state.foot_stage[0, 0] = runtime.CROSSED
        self.frame["sole_bottom"][0, 0] = self.c.required_clearance_m
        self.frame["x_min"][0, 0] = 1.08
        self.frame["x_max"][0, 0] = 1.26
        self.frame["landing_rectangle"][0, 0] = False
        result = self.step()
        self.assertEqual(float(result["components"]["tread_insertion"][0]), 0.0)

    def test_physical_failure_is_explicit_reward_component(self):
        self.frame["forbidden"][0] = True
        result = self.step()
        self.assertEqual(float(result["components"]["failure"][0]), 1.0)
        self.assertTrue(bool(result["failure"][0]))

    def test_reset_clears_all_mode3_episode_state(self):
        self.launch()
        self.state.previous_full_clearance[:] = True
        self.state.qualified_x_best[:] = 0.9
        self.state.maintained_advance_total[:] = 0.5
        self.state.reset(torch.tensor([0], dtype=torch.long))
        self.assertFalse(bool(self.state.ascent_started[0]))
        self.assertFalse(bool(self.state.previous_full_clearance[0].any()))
        self.assertEqual(float(self.state.qualified_x_best[0].sum()), 0.0)
        self.assertEqual(float(self.state.maintained_advance_total[0].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
