"""Adversarial contracts for the 048 align -> climb -> stabilize task."""

import unittest
import torch

from simulation.isaac_lab.g1_stairs import structured_step044_runtime as runtime
from simulation.isaac_lab.g1_stairs.structured_step044_config import (
    StructuredStep044Config,
)

MODE4_WEIGHTS = {
    "step044_approach": 0.05,
    "step044_lift_advance": 0.15,
    "step044_maintained_clearance_advance": 1.00,
    "step044_meaningful_lift": 0.15,
    "step044_riser_ready": 1.00,
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

class StructuredStep048StateTests(unittest.TestCase):

    def setUp(self):
        self.c = StructuredStep044Config(
            reward_mode=4,
            terminal_stability_dwell_s=0.06,
        )
        self.state = runtime.StructuredStep044State(
            1,
            "cpu",
            self.c,
            reward_weights=MODE4_WEIGHTS,
        )
        self.frame = {
            "root": torch.tensor([[0.0, 0.0, 0.80]], dtype=torch.float32),
            "gravity": torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            "x_min": torch.tensor([[-0.055, -0.055]], dtype=torch.float32),
            "x_max": torch.tensor([[0.125, 0.125]], dtype=torch.float32),
            "sole_bottom": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "normal_z": torch.ones((1, 2), dtype=torch.float32),
            "loaded": torch.ones((1, 2), dtype=torch.bool),
            # Intentionally false: 048 initialization must not require the old
            # simultaneous 0.12 s double-support dwell.
            "stable_lower": torch.zeros((1, 2), dtype=torch.bool),
            "stable_upper": torch.zeros((1, 2), dtype=torch.bool),
            "upper_geometry": torch.zeros((1, 2), dtype=torch.bool),
            "landing_rectangle": torch.ones((1, 2), dtype=torch.bool),
            "contact": torch.ones((1, 2), dtype=torch.bool),
            "foot_velocity": torch.zeros((1, 2, 3), dtype=torch.float32),
            "forbidden": torch.zeros(1, dtype=torch.bool),
            "root_linear_velocity": torch.zeros((1, 3), dtype=torch.float32),
            "root_angular_velocity": torch.zeros((1, 3), dtype=torch.float32),
            "root_yaw": torch.zeros(1, dtype=torch.float32),
            "foot_lateral_max": torch.full((1, 2), 0.20, dtype=torch.float32),
        }

    def step(self):
        return self.state.update(0.02, **self.frame)

    def test_safe_reset_pose_initializes_without_double_support_dwell(self):
        result = self.step()
        self.assertTrue(bool(self.state.ready[0]))
        self.assertAlmostEqual(float(self.state.base_reference[0]), 0.80, places=6)
        self.assertEqual(float(result["components"]["failure"][0]), 0.0)

    def test_launch_requires_centerline_yaw_speed_and_loaded_support(self):
        self.step()
        self.frame["root"][0, 0] = 0.80

        self.frame["root"][0, 1] = 0.20
        self.assertFalse(bool(self.step()["diagnostics"]["ascent_started"][0]))

        self.frame["root"][0, 1] = 0.0
        self.frame["root_yaw"][0] = 0.25
        self.assertFalse(bool(self.step()["diagnostics"]["ascent_started"][0]))

        self.frame["root_yaw"][0] = 0.0
        self.frame["root_linear_velocity"][0, 0] = 0.30
        self.assertFalse(bool(self.step()["diagnostics"]["ascent_started"][0]))

        self.frame["root_linear_velocity"][0, 0] = 0.12
        self.frame["foot_lateral_max"][:] = 0.40
        self.assertFalse(bool(self.step()["diagnostics"]["ascent_started"][0]))

        self.frame["foot_lateral_max"][:] = 0.20
        self.frame["loaded"][:] = False
        self.assertFalse(bool(self.step()["diagnostics"]["ascent_started"][0]))

        self.frame["loaded"][0, 1] = True
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["ascent_start_event"][0]))
        self.assertTrue(bool(result["diagnostics"]["ascent_started"][0]))

    def test_ascent_latch_survives_body_crossing(self):
        self.step()
        self.frame["root"][0, 0] = 0.80
        self.frame["root_linear_velocity"][0, 0] = 0.12
        self.assertTrue(bool(self.step()["diagnostics"]["ascent_started"][0]))
        self.frame["root"][0, 0] = 1.10
        self.frame["root"][0, 1] = 0.30
        self.frame["root_yaw"][0] = 0.5
        self.assertTrue(bool(self.step()["diagnostics"]["ascent_started"][0]))

    def _arm_terminal_state(self):
        self.step()
        self.state.ascent_started[:] = True
        self.state.body_transfer_seen[:] = True
        self.state.seen_upper[:] = True
        self.state.foot_stage[:] = runtime.UPPER_SUPPORT
        self.frame["root"][:] = torch.tensor([[1.10, 0.0, 0.86]])
        self.frame["stable_upper"][:] = True
        self.frame["upper_geometry"][:] = True
        self.frame["stable_lower"][:] = False
        self.frame["root_linear_velocity"][:] = 0.0
        self.frame["root_angular_velocity"][:] = 0.0
        self.frame["root_yaw"][:] = 0.0

    def test_completion_requires_continuous_stability_not_forward_distance(self):
        self._arm_terminal_state()
        first = self.step()
        second = self.step()
        third = self.step()
        self.assertFalse(bool(first["success"][0]))
        self.assertFalse(bool(second["success"][0]))
        self.assertTrue(bool(third["success"][0]))
        # Root x=1.10 is far below the historical 1.65 m continuation target.
        self.assertLess(float(self.frame["root"][0, 0]), 1.65)

    def test_terminal_motion_resets_dwell_and_cannot_fake_success(self):
        self._arm_terminal_state()
        self.step()
        self.step()
        self.frame["root_linear_velocity"][0, 0] = 0.20
        disturbed = self.step()
        self.assertFalse(bool(disturbed["success"][0]))
        self.assertEqual(float(self.state.terminal_stability_age[0]), 0.0)
        self.frame["root_linear_velocity"][0, 0] = 0.0
        self.assertFalse(bool(self.step()["success"][0]))
        self.assertFalse(bool(self.step()["success"][0]))
        self.assertTrue(bool(self.step()["success"][0]))

    def test_terminal_misalignment_cannot_complete(self):
        self._arm_terminal_state()
        self.frame["root"][0, 1] = 0.20
        for _ in range(5):
            self.assertFalse(bool(self.step()["success"][0]))
        self.assertEqual(float(self.state.terminal_stability_age[0]), 0.0)

    def test_terminal_foot_escape_cannot_complete(self):
        self._arm_terminal_state()
        self.frame["foot_lateral_max"][:] = 0.40
        for _ in range(5):
            self.assertFalse(bool(self.step()["success"][0]))
        self.assertEqual(float(self.state.terminal_stability_age[0]), 0.0)

    def test_terminal_requires_stricter_upright_and_body_height(self):
        self._arm_terminal_state()
        self.frame["gravity"][0, 2] = -0.95
        for _ in range(4):
            self.assertFalse(bool(self.step()["success"][0]))
        self.frame["gravity"][0, 2] = -1.0
        self.frame["root"][0, 2] = 0.85
        for _ in range(4):
            self.assertFalse(bool(self.step()["success"][0]))

    def test_prelaunch_camping_becomes_failure(self):
        self.step()
        steps = int(self.c.prelaunch_timeout_s / 0.02) + 2
        result = None
        for _ in range(steps):
            result = self.step()
            if bool(result["failure"][0]):
                break
        self.assertIsNotNone(result)
        self.assertTrue(
            bool(result["diagnostics"]["failure_prelaunch_timeout"][0])
        )
        self.assertTrue(bool(result["failure"][0]))

    def test_failure_causes_are_explicit(self):
        self.step()
        self.frame["forbidden"][0] = True
        result = self.step()
        self.assertTrue(bool(result["diagnostics"]["failure_forbidden"][0]))
        self.assertFalse(bool(result["diagnostics"]["failure_tilt"][0]))
        self.assertFalse(bool(result["diagnostics"]["failure_low_base"][0]))
        self.assertFalse(bool(result["diagnostics"]["failure_stall"][0]))

    def test_failure_claws_back_accumulated_positive_stair_credit(self):
        self.step()
        self.state.positive_stair_credit[:] = 6.0
        self.frame["forbidden"][0] = True
        result = self.step()
        raw_failure = float(result["components"]["failure"][0])
        weighted_failure = raw_failure * MODE4_WEIGHTS["step044_failure"]
        self.assertLessEqual(
            weighted_failure,
            -3.0 - self.c.failure_clawback_scale * 6.0 + 1.0e-6,
        )

    def test_ascent_stall_becomes_failure_instead_of_timeout_farming(self):
        self.step()
        self.frame["root"][0, 0] = 0.80
        self.frame["root_linear_velocity"][0, 0] = 0.12
        self.assertTrue(bool(self.step()["diagnostics"]["ascent_started"][0]))
        # No swing/support/crossing progress after launch.
        steps = int(self.c.ascent_stall_timeout_s / 0.02) + 2
        result = None
        for _ in range(steps):
            result = self.step()
            if bool(result["failure"][0]):
                break
        self.assertIsNotNone(result)
        self.assertTrue(bool(result["diagnostics"]["failure_stall"][0]))
        self.assertTrue(bool(result["failure"][0]))

    def test_reset_clears_terminal_and_ascent_state(self):
        self._arm_terminal_state()
        self.state.terminal_stability_age[:] = 0.04
        self.state.prelaunch_age[:] = 1.0
        self.state.ascent_progress_age[:] = 1.0
        self.state.positive_stair_credit[:] = 5.0
        self.state.reset(torch.tensor([0], dtype=torch.long))
        self.assertFalse(bool(self.state.ready[0]))
        self.assertFalse(bool(self.state.ascent_started[0]))
        self.assertEqual(float(self.state.terminal_stability_age[0]), 0.0)
        self.assertEqual(float(self.state.prelaunch_age[0]), 0.0)
        self.assertEqual(float(self.state.ascent_progress_age[0]), 0.0)
        self.assertEqual(float(self.state.positive_stair_credit[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
