"""Resolution and reward-budget contracts for experiment 047."""

import unittest
from pathlib import Path

import torch

from simulation.isaac_lab.g1_stairs.approach_teacher import (
    parse_approach_teacher_config,
)
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.reward_math import (
    exp_tracking_score,
)
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
)


ROOT = Path("experiments/g1_stairs/configs")


class StructuredStep047ConfigTests(unittest.TestCase):

    def resolve(self, name):
        return resolve_config(ROOT / name)[0]

    def test_047_smoke_resolves_exact_control_phase(self):
        resolved = self.resolve("047a_anti_exploit_single_step_smoke.yaml")
        self.assertEqual(resolved["structured_step"]["reward_mode"], 3)
        self.assertEqual(resolved["structured_step"]["ascent_launch_distance_m"], 0.25)
        teacher = parse_approach_teacher_config(resolved)
        self.assertIsNotNone(teacher)
        self.assertTrue(teacher.enforce_action_blend)
        self.assertTrue(teacher.gate_task_actor_loss)
        self.assertEqual(teacher.release_near_m, 0.25)
        self.assertEqual(teacher.release_far_m, 0.30)
        self.assertEqual(resolved["execution"]["environments"], 32)
        self.assertEqual(resolved["execution"]["maximum_iterations"], 1)

    def test_047_reward_set_has_no_legacy_leakage(self):
        weights = active_reward_weights(
            self.resolve("047a_anti_exploit_single_step_smoke.yaml")
        )
        expected_structured = {
            "step044_approach",
            "step044_lift_advance",
            "step044_maintained_clearance_advance",
            "step044_meaningful_lift",
            "step044_riser_ready",
            "step044_valid_crossing",
            "step044_tread_insertion",
            "step044_good_touchdown",
            "step044_stable_upper_support",
            "step044_supported_body_rise",
            "step044_body_transfer",
            "step044_both_feet_supported",
            "step044_complete",
            "step044_premature_lift",
            "step044_riser_risk",
            "step044_low_crossing",
            "step044_clearance_loss",
            "step044_bad_touchdown",
            "step044_failure",
        }
        actual_structured = {
            name for name in weights if name.startswith("step044_")
        }
        self.assertEqual(actual_structured, expected_structured)
        self.assertFalse(any(name.startswith("step043_") for name in weights))
        self.assertEqual(weights["track_ang_vel_z"], 0.0)

    def test_pre_crossing_reward_budget_is_dominated_by_failure_modes(self):
        w = active_reward_weights(
            self.resolve("047a_anti_exploit_single_step_smoke.yaml")
        )
        # Maximum bounded/event shaping available to the first foot before a
        # valid crossing. The velocity/upright regularizers are intentionally
        # excluded because they are not stair-milestone payouts.
        pre_cross_budget = (
            w["step044_approach"]
            + w["step044_lift_advance"]
            + w["step044_meaningful_lift"]
            + w["step044_maintained_clearance_advance"]
            + w["step044_riser_ready"]
        )
        self.assertLess(pre_cross_budget, abs(w["step044_low_crossing"]))
        self.assertLess(pre_cross_budget, abs(w["step044_clearance_loss"]))
        self.assertLess(pre_cross_budget, abs(w["step044_failure"]))

    def test_historical_044_and_045_modes_are_unchanged(self):
        r044 = self.resolve("044_structured_single_step_base.yaml")
        r045 = self.resolve("045_body_gated_clearance_diag200.yaml")
        self.assertEqual(r044["structured_step"].get("reward_mode", 1), 1)
        self.assertEqual(r045["structured_step"]["reward_mode"], 2)
        t045 = parse_approach_teacher_config(r045)
        self.assertFalse(t045.enforce_action_blend)
        self.assertFalse(t045.gate_task_actor_loss)
        self.assertEqual(t045.release_far_m, 0.40)

    def test_zero_centered_velocity_tracking_has_no_survival_bonus(self):
        error = torch.tensor([0.0, 0.15 ** 2])
        zero_centered = exp_tracking_score(
            error,
            0.15 ** 2,
            zero_centered=True,
        )
        historical = exp_tracking_score(
            error,
            0.15 ** 2,
            zero_centered=False,
        )
        self.assertEqual(float(zero_centered[0]), 0.0)
        self.assertLess(float(zero_centered[1]), 0.0)
        self.assertEqual(float(historical[0]), 1.0)
        self.assertGreater(float(historical[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
