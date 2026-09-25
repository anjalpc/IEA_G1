"""Resolved configuration contracts for experiment 049."""

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


class StructuredStep049ConfigTests(unittest.TestCase):

    def resolve(self, name):
        return resolve_config(ROOT / name)[0]

    def test_049_smoke_resolves_clearance_first_contract(self):
        r = self.resolve("049a_clearance_acquisition_smoke.yaml")
        self.assertEqual(r["structured_step"]["reward_mode"], 5)
        self.assertEqual(r["structured_step"]["ascent_launch_distance_m"], 0.30)
        self.assertEqual(r["execution"]["environments"], 32)
        self.assertEqual(r["execution"]["maximum_iterations"], 1)

    def test_teacher_remains_training_only(self):
        r = self.resolve("049a_clearance_acquisition_smoke.yaml")
        teacher = parse_approach_teacher_config(r)
        self.assertIsNotNone(teacher)
        self.assertFalse(teacher.enforce_action_blend)
        self.assertTrue(teacher.gate_task_actor_loss)
        self.assertEqual(teacher.release_near_m, 0.25)
        self.assertEqual(teacher.release_far_m, 0.40)

    def test_pre_crossing_budget_remains_below_major_failure_costs(self):
        w = active_reward_weights(
            self.resolve("049a_clearance_acquisition_smoke.yaml")
        )
        budget = (
            w["step044_approach"]
            + w["step044_lift_advance"]
            + w["step044_meaningful_lift"]
            + w["step044_maintained_clearance_advance"]
            + w["step044_riser_ready"]
        )
        self.assertAlmostEqual(budget, 2.40, places=6)
        self.assertLess(budget, abs(w["step044_low_crossing"]))
        self.assertLess(budget, abs(w["step044_clearance_loss"]))
        self.assertLess(budget, abs(w["step044_failure"]))

    def test_049_changes_only_pre_crossing_weights_from_048(self):
        w48 = active_reward_weights(
            self.resolve("048a_task_aligned_stable_smoke.yaml")
        )
        w49 = active_reward_weights(
            self.resolve("049a_clearance_acquisition_smoke.yaml")
        )
        changed = {
            "step044_lift_advance",
            "step044_meaningful_lift",
            "step044_maintained_clearance_advance",
            "step044_riser_ready",
        }
        for name in w48:
            if name in changed:
                self.assertNotEqual(w48[name], w49[name])
            else:
                self.assertEqual(w48[name], w49[name], name)

    def test_048_contract_remains_historical(self):
        r = self.resolve("048a_task_aligned_stable_smoke.yaml")
        self.assertEqual(r["structured_step"]["reward_mode"], 4)
        self.assertEqual(r["structured_step"]["ascent_launch_distance_m"], 0.25)


if __name__ == "__main__":
    unittest.main()
