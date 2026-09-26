"""Resolved configuration contracts for experiment 050."""

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


class StructuredStep050ConfigTests(unittest.TestCase):

    def resolve(self, name):
        return resolve_config(ROOT / name)[0]

    def test_050_smoke_resolves_mode6_contract(self):
        r = self.resolve("050a_locomotion_preserving_clearance_smoke.yaml")
        self.assertEqual(r["structured_step"]["reward_mode"], 6)
        self.assertEqual(r["structured_step"]["ascent_launch_distance_m"], 0.30)
        self.assertEqual(r["execution"]["environments"], 32)
        self.assertEqual(r["execution"]["maximum_iterations"], 1)

    def test_ascent_keeps_low_speed_locomotion_command_context(self):
        r = self.resolve("050a_locomotion_preserving_clearance_smoke.yaml")
        schedule = r["command"]["stair_velocity_schedule"]
        self.assertFalse(schedule["stop_after_ascent_start"])
        self.assertAlmostEqual(schedule["stair_velocity_mps"], 0.12)
        # The task velocity reward is separately disabled after ascent by
        # reward_mode 6, so this command cannot become a forward reward.

    def test_teacher_remains_training_only(self):
        r = self.resolve("050a_locomotion_preserving_clearance_smoke.yaml")
        teacher = parse_approach_teacher_config(r)
        self.assertIsNotNone(teacher)
        self.assertFalse(teacher.enforce_action_blend)
        self.assertTrue(teacher.gate_task_actor_loss)

    def test_050_preserves_049_reward_budget_exactly(self):
        w49 = active_reward_weights(
            self.resolve("049a_clearance_acquisition_smoke.yaml")
        )
        w50 = active_reward_weights(
            self.resolve("050a_locomotion_preserving_clearance_smoke.yaml")
        )
        self.assertEqual(w49, w50)

    def test_049_contract_remains_historical(self):
        r = self.resolve("049a_clearance_acquisition_smoke.yaml")
        self.assertEqual(r["structured_step"]["reward_mode"], 5)
        self.assertTrue(r["command"]["stair_velocity_schedule"]["stop_after_ascent_start"])


if __name__ == "__main__":
    unittest.main()
