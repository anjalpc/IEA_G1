import unittest

import torch

from simulation.isaac_lab.g1_stairs.approach_teacher import (
    blend_teacher_student_actions,
    control_aware_actor_mean,
    student_control_authority,
)


class TeacherActionBlendTests(unittest.TestCase):

    def setUp(self):
        self.student = torch.full((3, 29), 2.0)
        self.teacher = torch.full((3, 29), -1.0)

    def test_far_executes_teacher_exactly(self):
        result = blend_teacher_student_actions(
            self.student,
            self.teacher,
            torch.ones((3, 1)),
        )

        self.assertTrue(
            torch.equal(
                result,
                self.teacher,
            )
        )

    def test_near_executes_student_exactly(self):
        result = blend_teacher_student_actions(
            self.student,
            self.teacher,
            torch.zeros((3, 1)),
        )

        self.assertTrue(
            torch.equal(
                result,
                self.student,
            )
        )

    def test_half_gate_is_exact_convex_blend(self):
        gate = torch.full((3, 1), 0.5)

        result = blend_teacher_student_actions(
            self.student,
            self.teacher,
            gate,
        )

        expected = (
            0.5 * self.teacher
            + 0.5 * self.student
        )

        self.assertTrue(
            torch.equal(
                result,
                expected,
            )
        )

    def test_nonfinite_fails_closed(self):
        bad = self.student.clone()
        bad[0, 0] = float("nan")

        with self.assertRaisesRegex(
            RuntimeError,
            "non-finite",
        ):
            blend_teacher_student_actions(
                bad,
                self.teacher,
                torch.ones((3, 1)),
            )

    def test_invalid_gate_fails_closed(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"\[0, 1\]",
        ):
            blend_teacher_student_actions(
                self.student,
                self.teacher,
                torch.full((3, 1), 1.01),
            )

    def test_student_control_authority_is_complement_of_teacher_gate(self):
        gate = torch.tensor([[1.0], [0.5], [0.0]])
        authority = student_control_authority(gate)
        self.assertTrue(
            torch.equal(
                authority,
                torch.tensor([0.0, 0.5, 1.0]),
            )
        )

    def test_student_control_authority_rejects_invalid_gate(self):
        with self.assertRaisesRegex(RuntimeError, r"\[0, 1\]"):
            student_control_authority(
                torch.tensor([[1.1]])
            )

    def test_control_aware_actor_loss_is_zero_under_full_teacher_authority(self):
        values = torch.tensor([2.0, -1.0, 4.0])
        result = control_aware_actor_mean(
            values,
            torch.ones((3, 1)),
            enabled=True,
        )
        self.assertEqual(float(result), 0.0)

    def test_control_aware_actor_loss_is_full_under_student_authority(self):
        values = torch.tensor([2.0, -1.0, 4.0])
        result = control_aware_actor_mean(
            values,
            torch.zeros((3, 1)),
            enabled=True,
        )
        self.assertEqual(float(result), float(values.mean()))

    def test_historical_actor_loss_ignores_gate_when_disabled(self):
        values = torch.tensor([2.0, -1.0, 4.0])
        result = control_aware_actor_mean(
            values,
            torch.ones((3, 1)),
            enabled=False,
        )
        self.assertEqual(float(result), float(values.mean()))


if __name__ == "__main__":
    unittest.main()
