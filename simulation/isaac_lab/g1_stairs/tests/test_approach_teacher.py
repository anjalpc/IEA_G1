from __future__ import annotations

import unittest

import torch

from simulation.isaac_lab.g1_stairs.approach_teacher import (
    parse_approach_teacher_config,
    smooth_l1_per_environment,
    smooth_release_gate,
    teacher_observation,
)


class ApproachTeacherTest(unittest.TestCase):
    def test_release_gate_boundaries_and_midpoint(self) -> None:
        distance = torch.tensor([0.20, 0.25, 0.325, 0.40, 0.50])
        gate = smooth_release_gate(distance, 0.25, 0.40)
        torch.testing.assert_close(
            gate,
            torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]),
        )

    def test_teacher_observation_requires_exact_contract(self) -> None:
        value = torch.zeros(3, 480)
        self.assertIs(teacher_observation({"approach_teacher": value}), value)
        with self.assertRaisesRegex(RuntimeError, "480"):
            teacher_observation({"approach_teacher": torch.zeros(3, 485)})

    def test_loss_is_zero_for_matching_actions(self) -> None:
        action = torch.zeros(4, 29)
        loss = smooth_l1_per_environment(action, action, 0.25)
        torch.testing.assert_close(loss, torch.zeros(4))

    def test_invalid_release_interval_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "near < far"):
            parse_approach_teacher_config(
                {"approach_teacher": {
                    "enabled": True,
                    "coefficient": 1.0,
                    "release_near_m": 0.4,
                    "release_far_m": 0.25,
                    "huber_delta": 0.25,
                }}
            )


if __name__ == "__main__":
    unittest.main()
