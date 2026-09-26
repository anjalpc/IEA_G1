"""Resolved configuration contracts for experiment 048."""

import unittest
from pathlib import Path

from simulation.isaac_lab.g1_stairs.approach_teacher import (
    parse_approach_teacher_config,
)
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
)

ROOT = Path("experiments/g1_stairs/configs")


class StructuredStep048ConfigTests(unittest.TestCase):

    def resolve(self, name):
        return resolve_config(ROOT / name)[0]

    def test_048_smoke_resolves_task_contract(self):
        r = self.resolve("048a_task_aligned_stable_smoke.yaml")
        self.assertEqual(r["structured_step"]["reward_mode"], 4)
        self.assertEqual(r["structured_step"]["ascent_launch_distance_m"], 0.25)
        self.assertEqual(r["structured_step"]["terminal_stability_dwell_s"], 0.50)
        self.assertEqual(r["structured_step"]["terminal_upright_cos"], 0.97)
        self.assertEqual(
            r["structured_step"]["terminal_base_height_gain_fraction"],
            0.90,
        )
        self.assertEqual(r["structured_step"]["prelaunch_timeout_s"], 6.00)
        self.assertEqual(r["structured_step"]["ascent_stall_timeout_s"], 2.50)
        self.assertEqual(r["structured_step"]["failure_clawback_scale"], 1.50)
        self.assertEqual(r["command"]["forward_velocity_mps"], 0.35)
        sched = r["command"]["stair_velocity_schedule"]
        self.assertEqual(sched["approach_velocity_mps"], 0.35)
        self.assertEqual(sched["stair_velocity_mps"], 0.12)
        self.assertTrue(sched["stop_after_ascent_start"])
        self.assertEqual(sched["platform_velocity_mps"], 0.0)
        self.assertEqual(sched["centerline_gain_s"], 1.0)
        self.assertEqual(sched["max_lateral_velocity_mps"], 0.12)
        self.assertEqual(sched["heading_gain_s"], 1.5)
        self.assertEqual(sched["max_yaw_rate_radps"], 0.35)
        self.assertEqual(r["execution"]["environments"], 32)
        self.assertEqual(r["execution"]["maximum_iterations"], 1)

    def test_teacher_is_training_only_but_task_gradient_is_gated(self):
        r = self.resolve("048a_task_aligned_stable_smoke.yaml")
        teacher = parse_approach_teacher_config(r)
        self.assertIsNotNone(teacher)
        self.assertFalse(teacher.enforce_action_blend)
        self.assertTrue(teacher.gate_task_actor_loss)
        self.assertEqual(teacher.release_near_m, 0.25)
        self.assertEqual(teacher.release_far_m, 0.40)

    def test_reward_set_contains_alignment_and_no_legacy_leakage(self):
        w = active_reward_weights(
            self.resolve("048a_task_aligned_stable_smoke.yaml")
        )
        self.assertEqual(w["stair_centerline_alignment"], -1.0)
        self.assertEqual(w["stair_heading_alignment"], -1.0)
        self.assertEqual(w["track_stair_velocity_world"], 0.15)
        self.assertFalse(any(name.startswith("step043_") for name in w))
        self.assertIn("step044_complete", w)
        self.assertIn("step044_failure", w)

    def test_pre_crossing_positive_budget_remains_below_major_failures(self):
        w = active_reward_weights(
            self.resolve("048a_task_aligned_stable_smoke.yaml")
        )
        budget = (
            w["step044_approach"]
            + w["step044_lift_advance"]
            + w["step044_meaningful_lift"]
            + w["step044_maintained_clearance_advance"]
            + w["step044_riser_ready"]
        )
        self.assertLess(budget, abs(w["step044_low_crossing"]))
        self.assertLess(budget, abs(w["step044_clearance_loss"]))
        self.assertLess(budget, abs(w["step044_failure"]))

    def test_047_historical_control_contract_is_unchanged(self):
        r = self.resolve("047a_anti_exploit_single_step_smoke.yaml")
        self.assertEqual(r["structured_step"]["reward_mode"], 3)
        teacher = parse_approach_teacher_config(r)
        self.assertTrue(teacher.enforce_action_blend)
        self.assertTrue(teacher.gate_task_actor_loss)
        self.assertEqual(teacher.release_far_m, 0.30)


if __name__ == "__main__":
    unittest.main()
