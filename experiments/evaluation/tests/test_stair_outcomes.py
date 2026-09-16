"""Tests for reward-independent stair outcome classification."""

import math
import unittest

from experiments.evaluation.stair_outcomes import (
    Outcome,
    StairOutcomeEvaluator,
    StepSample,
)


def sample(
    time_s,
    x,
    step,
    *,
    height=0.80,
    roll=0.0,
    pitch=0.0,
    contacts=(),
    joint_violation=False,
    velocity_violation=False,
    torque_violation=False,
    slip_violation=False,
    timed_out=False,
):
    return StepSample.create(
        time_s=time_s,
        base_x_m=x,
        base_height_m=height,
        roll_rad=roll,
        pitch_rad=pitch,
        step_index=step,
        contacts=contacts,
        joint_violation=joint_violation,
        velocity_violation=velocity_violation,
        torque_violation=torque_violation,
        slip_violation=slip_violation,
        timed_out=timed_out,
    )


class StairOutcomeEvaluatorTest(unittest.TestCase):
    def test_success_requires_continuation_and_dwell(self):
        evaluator = StairOutcomeEvaluator()

        self.assertIsNone(
            evaluator.update(sample(0.0, 1.0, 0))
        )
        self.assertIsNone(
            evaluator.update(sample(1.0, 2.1, 1))
        )
        self.assertIsNone(
            evaluator.update(sample(2.0, 2.4, 2))
        )
        self.assertIsNone(
            evaluator.update(sample(3.0, 2.8, 3))
        )

        # First entry starts dwell at zero.
        self.assertIsNone(
            evaluator.update(sample(4.0, 4.75, 3))
        )

        result = evaluator.update(
            sample(5.1, 4.80, 3)
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.outcome,
            Outcome.SUCCESS,
        )
        self.assertTrue(result.success)
        self.assertGreaterEqual(
            result.metrics["platform_continuation_m"],
            2.0,
        )
        self.assertGreaterEqual(
            result.metrics["upright_dwell_s"],
            1.0,
        )

    def test_reaching_height_without_continuation_stalls(self):
        evaluator = StairOutcomeEvaluator()

        evaluator.update(sample(0.0, 1.0, 0))
        evaluator.update(
            sample(
                1.0,
                2.8,
                3,
                height=1.40,
            )
        )

        result = evaluator.update(
            sample(
                18.0,
                2.8,
                3,
                height=1.40,
                timed_out=True,
            )
        )

        self.assertEqual(
            result.outcome,
            Outcome.STALL,
        )
        self.assertFalse(result.success)

    def test_forbidden_contact_is_primary_failure(self):
        evaluator = StairOutcomeEvaluator()

        evaluator.update(sample(0.0, 1.0, 0))

        result = evaluator.update(
            sample(
                0.5,
                1.5,
                0,
                contacts=("left_knee",),
            )
        )

        self.assertEqual(
            result.outcome,
            Outcome.FORBIDDEN_CONTACT,
        )
        self.assertIn(
            Outcome.FORBIDDEN_CONTACT.value,
            result.secondary_flags,
        )

    def test_fall_has_priority_over_forbidden_contact(self):
        evaluator = StairOutcomeEvaluator()

        result = evaluator.update(
            sample(
                0.0,
                1.0,
                0,
                height=0.30,
                contacts=("pelvis", "left_knee"),
            )
        )

        self.assertEqual(
            result.outcome,
            Outcome.FALL,
        )
        self.assertIn(
            Outcome.FORBIDDEN_CONTACT.value,
            result.secondary_flags,
        )

    def test_constraint_priority_is_deterministic(self):
        evaluator = StairOutcomeEvaluator()

        result = evaluator.update(
            sample(
                0.0,
                1.0,
                0,
                torque_violation=True,
                velocity_violation=True,
                joint_violation=True,
            )
        )

        self.assertEqual(
            result.outcome,
            Outcome.TORQUE_VIOLATION,
        )
        self.assertIn(
            Outcome.VELOCITY_VIOLATION.value,
            result.secondary_flags,
        )
        self.assertIn(
            Outcome.JOINT_VIOLATION.value,
            result.secondary_flags,
        )

    def test_nonfinite_state_fails_closed(self):
        evaluator = StairOutcomeEvaluator()

        result = evaluator.update(
            sample(
                0.0,
                math.nan,
                0,
                contacts=("pelvis",),
            )
        )

        self.assertEqual(
            result.outcome,
            Outcome.INVALID_SIMULATION_STATE,
        )
        self.assertFalse(result.success)

    def test_wrong_direction_is_separate_failure(self):
        evaluator = StairOutcomeEvaluator()

        result = evaluator.update(
            sample(0.5, 0.70, 0)
        )

        self.assertEqual(
            result.outcome,
            Outcome.WRONG_DIRECTION,
        )

    def test_stall_is_separate_failure(self):
        evaluator = StairOutcomeEvaluator()

        evaluator.update(sample(0.0, 1.0, 0))

        result = evaluator.update(
            sample(2.1, 1.0, 0)
        )

        self.assertEqual(
            result.outcome,
            Outcome.STALL,
        )

    def test_small_continuous_progress_does_not_stall(self):
        evaluator = StairOutcomeEvaluator()

        self.assertIsNone(
            evaluator.update(sample(0.0, 1.0, 0))
        )

        for index in range(1, 31):
            result = evaluator.update(
                sample(
                    index * 0.1,
                    1.0 + index * 0.01,
                    0,
                )
            )
            self.assertIsNone(result)

    def test_time_must_be_monotonic(self):
        evaluator = StairOutcomeEvaluator()

        evaluator.update(sample(1.0, 1.1, 0))

        with self.assertRaises(ValueError):
            evaluator.update(
                sample(0.5, 1.2, 0)
            )

    def test_terminal_evaluator_requires_reset(self):
        evaluator = StairOutcomeEvaluator()

        evaluator.update(
            sample(
                0.0,
                1.0,
                0,
                height=0.30,
            )
        )

        with self.assertRaises(RuntimeError):
            evaluator.update(
                sample(0.1, 1.0, 0)
            )

        evaluator.reset()

        self.assertIsNone(
            evaluator.update(
                sample(0.0, 1.0, 0)
            )
        )


if __name__ == "__main__":
    unittest.main()