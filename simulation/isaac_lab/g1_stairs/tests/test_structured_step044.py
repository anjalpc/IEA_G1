"""Pure counterexample tests for the 044 structured state machine."""

import copy
import math
import unittest

import torch

from simulation.isaac_lab.g1_stairs.structured_step044_config import (
    StructuredStep044Config,
)
from simulation.isaac_lab.g1_stairs import structured_step044_runtime as runtime


class StructuredStep044Tests(unittest.TestCase):

    def setUp(self):
        self.c = StructuredStep044Config()
        self.state = runtime.StructuredStep044State(
            1,
            "cpu",
            self.c,
        )

        self.frame = {
            "root":
                torch.tensor(
                    [[0.0, 0.0, 0.80]],
                    dtype=torch.float32,
                ),
            "gravity":
                torch.tensor(
                    [[0.0, 0.0, -1.0]],
                    dtype=torch.float32,
                ),
            "x_min":
                torch.tensor(
                    [[-0.055, -0.055]],
                    dtype=torch.float32,
                ),
            "x_max":
                torch.tensor(
                    [[0.125, 0.125]],
                    dtype=torch.float32,
                ),
            "sole_bottom":
                torch.tensor(
                    [[0.0, 0.0]],
                    dtype=torch.float32,
                ),
            "normal_z":
                torch.ones(
                    (1, 2),
                    dtype=torch.float32,
                ),
            "loaded":
                torch.ones(
                    (1, 2),
                    dtype=torch.bool,
                ),
            "stable_lower":
                torch.ones(
                    (1, 2),
                    dtype=torch.bool,
                ),
            "stable_upper":
                torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
            "upper_geometry":
                torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
            "landing_rectangle":
                torch.ones(
                    (1, 2),
                    dtype=torch.bool,
                ),
            "contact":
                torch.ones(
                    (1, 2),
                    dtype=torch.bool,
                ),
            "foot_velocity":
                torch.zeros(
                    (1, 2, 3),
                    dtype=torch.float32,
                ),
            "forbidden":
                torch.zeros(
                    1,
                    dtype=torch.bool,
                ),
        }

        # Initial double lower support calibrates the detector.
        self.step()

    def step(self):
        return self.state.update(
            0.02,
            **self.frame,
        )

    def make_left_swing(self):
        self.frame["loaded"][0, 0] = False
        self.frame["contact"][0, 0] = False
        self.frame["stable_lower"][0, 0] = False

    def test_forward_motion_at_ground_height_does_not_get_lift_success(self):
        self.make_left_swing()

        self.frame["x_min"][0, 0] = 0.765
        self.frame["x_max"][0, 0] = 0.945
        self.frame["sole_bottom"][0, 0] = 0.0

        result = self.step()

        self.assertEqual(
            float(
                result["components"][
                    "meaningful_lift"
                ][0]
            ),
            0.0,
        )

        self.assertEqual(
            float(
                result["components"][
                    "riser_ready"
                ][0]
            ),
            0.0,
        )

        self.assertEqual(
            float(
                result["components"][
                    "valid_crossing"
                ][0]
            ),
            0.0,
        )

        self.assertEqual(
            int(
                result["foot_stage"][0, 0]
            ),
            runtime.SWING,
        )

    def test_lifting_far_from_stair_does_not_get_near_riser_milestone(self):
        self.make_left_swing()

        self.frame["x_min"][0, 0] = 0.20
        self.frame["x_max"][0, 0] = 0.38
        self.frame["sole_bottom"][0, 0] = 0.10

        result = self.step()

        self.assertEqual(
            float(
                result["components"][
                    "riser_ready"
                ][0]
            ),
            0.0,
        )

        self.assertEqual(
            float(
                result["components"][
                    "valid_crossing"
                ][0]
            ),
            0.0,
        )

    def test_low_plane_crossing_is_penalized_and_not_valid(self):
        self.make_left_swing()

        # First establish history just before the plane.
        self.frame["x_min"][0, 0] = 0.815
        self.frame["x_max"][0, 0] = 0.995
        self.frame["sole_bottom"][0, 0] = 0.0
        self.step()

        # Then cross while still low.
        self.frame["x_min"][0, 0] = 0.825
        self.frame["x_max"][0, 0] = 1.005

        result = self.step()

        self.assertEqual(
            float(
                result["components"][
                    "low_crossing"
                ][0]
            ),
            1.0,
        )

        self.assertEqual(
            float(
                result["components"][
                    "valid_crossing"
                ][0]
            ),
            0.0,
        )

        self.assertLess(
            int(
                result["foot_stage"][0, 0]
            ),
            runtime.CROSSED,
        )

    def test_lift_then_drop_causes_clearance_loss(self):
        self.make_left_swing()

        # Enter lift zone -> SWING.
        self.frame["x_min"][0, 0] = 0.77
        self.frame["x_max"][0, 0] = 0.95
        self.frame["sole_bottom"][0, 0] = 0.04
        self.step()

        # SWING -> LIFTED.
        self.frame["x_min"][0, 0] = 0.79
        self.frame["x_max"][0, 0] = 0.97
        self.frame["sole_bottom"][0, 0] = 0.09
        self.step()

        # LIFTED -> RISER_READY.
        self.frame["x_min"][0, 0] = 0.815
        self.frame["x_max"][0, 0] = 0.995
        self.frame["sole_bottom"][0, 0] = 0.09

        ready = self.step()

        self.assertEqual(
            int(
                ready["foot_stage"][0, 0]
            ),
            runtime.RISER_READY,
        )

        # Drop before crossing.
        self.frame["sole_bottom"][0, 0] = 0.03

        dropped = self.step()

        self.assertEqual(
            float(
                dropped["components"][
                    "clearance_loss"
                ][0]
            ),
            1.0,
        )

        self.assertEqual(
            float(
                dropped["components"][
                    "valid_crossing"
                ][0]
            ),
            0.0,
        )

    def test_valid_crossing_requires_riser_ready_and_clearance(self):
        self.make_left_swing()

        # LOWER -> SWING.
        self.frame["x_min"][0, 0] = 0.77
        self.frame["x_max"][0, 0] = 0.95
        self.frame["sole_bottom"][0, 0] = 0.04
        self.step()

        # SWING -> LIFTED.
        self.frame["x_min"][0, 0] = 0.79
        self.frame["x_max"][0, 0] = 0.97
        self.frame["sole_bottom"][0, 0] = 0.09
        self.step()

        # LIFTED -> RISER_READY.
        self.frame["x_min"][0, 0] = 0.815
        self.frame["x_max"][0, 0] = 0.995
        self.frame["sole_bottom"][0, 0] = 0.09

        ready = self.step()

        self.assertEqual(
            float(
                ready["components"][
                    "riser_ready"
                ][0]
            ),
            1.0,
        )

        # Genuine cleared crossing.
        self.frame["x_min"][0, 0] = 0.825
        self.frame["x_max"][0, 0] = 1.005
        self.frame["sole_bottom"][0, 0] = 0.09

        crossed = self.step()

        self.assertEqual(
            float(
                crossed["components"][
                    "valid_crossing"
                ][0]
            ),
            1.0,
        )

        self.assertEqual(
            float(
                crossed["components"][
                    "low_crossing"
                ][0]
            ),
            0.0,
        )

        self.assertEqual(
            int(
                crossed["foot_stage"][0, 0]
            ),
            runtime.CROSSED,
        )

    def test_toe_only_low_crossing_cannot_get_insertion(self):
        self.make_left_swing()

        self.frame["x_min"][0, 0] = 0.825
        self.frame["x_max"][0, 0] = 1.005
        self.frame["sole_bottom"][0, 0] = 0.0

        result = self.step()

        self.assertEqual(
            float(
                result["components"][
                    "tread_insertion"
                ][0]
            ),
            0.0,
        )

    def test_lift_advance_requires_both_height_and_forward_progress(self):
        self.make_left_swing()

        # In lift zone but low.
        self.frame["x_min"][0, 0] = 0.72
        self.frame["x_max"][0, 0] = 0.90
        self.frame["sole_bottom"][0, 0] = 0.0

        low = self.step()

        self.assertEqual(
            float(
                low["components"][
                    "lift_advance"
                ][0]
            ),
            0.0,
        )

        # Same forward region with useful lift.
        self.frame["sole_bottom"][0, 0] = 0.05

        lifted = self.step()

        self.assertGreater(
            float(
                lifted["components"][
                    "lift_advance"
                ][0]
            ),
            0.0,
        )

        # Identical pose cannot repay.
        repeated = self.step()

        self.assertEqual(
            float(
                repeated["components"][
                    "lift_advance"
                ][0]
            ),
            0.0,
        )

    def test_body_rise_without_upper_support_gets_no_credit(self):
        self.frame["root"][0, 2] = 0.90

        result = self.step()

        self.assertEqual(
            float(
                result["components"][
                    "supported_body_rise"
                ][0]
            ),
            0.0,
        )

        self.assertEqual(
            float(
                result["components"][
                    "body_transfer"
                ][0]
            ),
            0.0,
        )

    def test_nonfinite_input_fails_closed(self):
        self.frame["root"][0, 0] = math.nan

        with self.assertRaises(
            RuntimeError
        ):
            self.step()

    def test_selective_reset_only_clears_selected_environment(self):
        state = runtime.StructuredStep044State(
            2,
            "cpu",
            self.c,
        )

        state.foot_stage[:] = runtime.CROSSED
        state.low_crossing_seen[:] = True

        state.reset(
            torch.tensor(
                [0],
                dtype=torch.long,
            )
        )

        self.assertTrue(
            bool(
                (
                    state.foot_stage[0]
                    == runtime.LOWER
                ).all()
            )
        )

        self.assertTrue(
            bool(
                (
                    state.foot_stage[1]
                    == runtime.CROSSED
                ).all()
            )
        )

        self.assertFalse(
            bool(
                state.low_crossing_seen[0]
                .any()
            )
        )

        self.assertTrue(
            bool(
                state.low_crossing_seen[1]
                .all()
            )
        )

    def test_exact_clearance_boundary_uses_explicit_tolerance(self):
        self.make_left_swing()

        self.frame["x_min"][0, 0] = 0.815
        self.frame["x_max"][0, 0] = 0.995

        # Enter SWING first.
        self.frame["sole_bottom"][0, 0] = 0.04
        self.step()

        # Float32 representation at nominal 0.085 m must be accepted.
        self.frame["sole_bottom"][0, 0] = (
            self.c.required_clearance_m
        )

        self.frame["x_min"][0, 0] = 0.79
        self.frame["x_max"][0, 0] = 0.97
        self.step()

        self.frame["x_min"][0, 0] = 0.815
        self.frame["x_max"][0, 0] = 0.995

        result = self.step()

        self.assertEqual(
            int(
                result["foot_stage"][0, 0]
            ),
            runtime.RISER_READY,
        )


if __name__ == "__main__":
    unittest.main()
