from __future__ import annotations

import unittest

import torch

from simulation.isaac_lab.g1_stairs.approach_teacher import (
    ACTION_GROUP_INDICES,
    action_group_mean_abs,
    gated_teacher_loss,
    parse_approach_teacher_config,
    smooth_l1_per_environment,
    smooth_release_gate,
    teacher_diagnostic_terms,
    teacher_gate,
    teacher_observation,
)


class ApproachTeacherTest(unittest.TestCase):
    def test_action_groups_are_disjoint_and_exhaustive(self) -> None:
        flattened = [
            index
            for indices
            in ACTION_GROUP_INDICES.values()
            for index in indices
        ]

        self.assertEqual(
            len(flattened),
            29,
        )

        self.assertEqual(
            len(set(flattened)),
            29,
        )

        self.assertEqual(
            sorted(flattened),
            list(range(29)),
        )

        self.assertEqual(
            len(
                ACTION_GROUP_INDICES[
                    "legs"
                ]
            ),
            12,
        )

        self.assertEqual(
            len(
                ACTION_GROUP_INDICES[
                    "arms"
                ]
            ),
            14,
        )

        self.assertEqual(
            len(
                ACTION_GROUP_INDICES[
                    "waist"
                ]
            ),
            3,
        )

    def test_action_group_mean_abs_uses_canonical_action_order(self) -> None:
        actions = torch.zeros(
            2,
            29,
        )

        actions[
            :,
            ACTION_GROUP_INDICES[
                "legs"
            ],
        ] = 1.0

        actions[
            :,
            ACTION_GROUP_INDICES[
                "arms"
            ],
        ] = 2.0

        actions[
            :,
            ACTION_GROUP_INDICES[
                "waist"
            ],
        ] = 3.0

        values = action_group_mean_abs(
            actions
        )

        self.assertAlmostEqual(
            float(values["legs"]),
            1.0,
        )

        self.assertAlmostEqual(
            float(values["arms"]),
            2.0,
        )

        self.assertAlmostEqual(
            float(values["waist"]),
            3.0,
        )

    def test_teacher_diagnostics_separate_phase_and_body_groups(self) -> None:
        student = torch.zeros(
            3,
            29,
        )

        teacher = torch.zeros_like(
            student
        )

        # Leg mismatch only in the far sample.
        teacher[
            0,
            ACTION_GROUP_INDICES[
                "legs"
            ],
        ] = 1.0

        # Arm mismatch only in the release sample.
        teacher[
            1,
            ACTION_GROUP_INDICES[
                "arms"
            ],
        ] = 1.0

        # Waist mismatch only in the near sample.
        teacher[
            2,
            ACTION_GROUP_INDICES[
                "waist"
            ],
        ] = 1.0

        gate = torch.tensor(
            [
                [1.0],
                [0.5],
                [0.0],
            ]
        )

        diagnostics = (
            teacher_diagnostic_terms(
                student,
                teacher,
                gate,
                0.25,
            )
        )

        torch.testing.assert_close(
            diagnostics["far_mask"],
            torch.tensor(
                [True, False, False]
            ),
        )

        torch.testing.assert_close(
            diagnostics["release_mask"],
            torch.tensor(
                [False, True, False]
            ),
        )

        torch.testing.assert_close(
            diagnostics["near_mask"],
            torch.tensor(
                [False, False, True]
            ),
        )

        self.assertGreater(
            float(
                diagnostics[
                    "legs_error"
                ][0]
            ),
            0.0,
        )

        self.assertEqual(
            float(
                diagnostics[
                    "legs_error"
                ][1]
            ),
            0.0,
        )

        self.assertGreater(
            float(
                diagnostics[
                    "arms_error"
                ][1]
            ),
            0.0,
        )

        self.assertEqual(
            float(
                diagnostics[
                    "arms_error"
                ][2]
            ),
            0.0,
        )

        self.assertGreater(
            float(
                diagnostics[
                    "waist_error"
                ][2]
            ),
            0.0,
        )

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

    def test_stored_teacher_rows_and_gates_remain_aligned(self) -> None:
        observations = {
            "approach_teacher": torch.arange(3.0).reshape(3, 1).repeat(1, 480),
            "approach_teacher_gate": torch.tensor([[0.0], [0.5], [1.0]]),
        }
        permutation = torch.tensor([2, 0, 1])
        shuffled = {
            name: value[permutation]
            for name, value in observations.items()
        }
        obs = teacher_observation(shuffled)
        gate = teacher_gate(shuffled)
        torch.testing.assert_close(obs[:, 0], torch.tensor([2.0, 0.0, 1.0]))
        torch.testing.assert_close(gate[:, 0], torch.tensor([1.0, 0.0, 0.5]))

    def test_auxiliary_loss_is_actor_only_and_teacher_is_frozen(self) -> None:
        actor = torch.nn.Linear(4, 2)
        critic = torch.nn.Linear(4, 1)
        teacher = torch.nn.Linear(4, 2)
        teacher.requires_grad_(False)
        inputs = torch.ones(3, 4)
        student_mean = actor(inputs)
        with torch.no_grad():
            target = teacher(inputs)
        loss = gated_teacher_loss(
            student_mean,
            target,
            torch.ones(3, 1),
            0.25,
        )
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in actor.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_zero_gate_produces_zero_actor_gradient(self) -> None:
        student = torch.tensor([[1.0, -1.0]], requires_grad=True)
        target = torch.zeros_like(student)
        loss = gated_teacher_loss(student, target, torch.zeros(1, 1), 0.25)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

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
