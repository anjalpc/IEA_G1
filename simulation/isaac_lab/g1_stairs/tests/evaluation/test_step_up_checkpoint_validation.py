"""Tests for checkpoint step-up calibration lifecycle."""

import unittest

from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
    calibration_ready,
)


class StepUpCheckpointCalibrationTest(unittest.TestCase):
    def test_calibration_finishes_as_soon_as_both_populations_are_ready(self):
        forces = [100.0] * 10
        heights = [0.78] * 10

        self.assertTrue(calibration_ready(forces, heights))

    def test_calibration_requires_both_measurement_populations(self):
        self.assertFalse(
            calibration_ready([100.0] * 10, [0.78] * 9)
        )
        self.assertFalse(
            calibration_ready([100.0] * 9, [0.78] * 10)
        )

    def test_calibration_rejects_invalid_minimum(self):
        with self.assertRaises(ValueError):
            calibration_ready([], [], minimum_samples=0)


if __name__ == "__main__":
    unittest.main()
