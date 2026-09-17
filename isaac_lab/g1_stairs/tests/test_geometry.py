import unittest
from pathlib import Path

import yaml

from isaac_lab.g1_stairs.contracts import (
    BODY_JOINTS,
    MODE_MACHINE,
    NOMINAL_RISER_HEIGHT_M,
)
from isaac_lab.g1_stairs.geometry import StaircaseSpec, nominal_staircase


class StaircaseGeometryTest(unittest.TestCase):
    def test_nominal_dimensions(self):
        stairs = nominal_staircase()
        self.assertAlmostEqual(stairs.total_rise, 0.60)
        self.assertAlmostEqual(stairs.stair_run, 0.75)
        self.assertAlmostEqual(stairs.course_length, 5.75)
        self.assertEqual(stairs.fixed_start_xy_yaw, (1.0, 0.0, 0.0))

    def test_initial_training_uses_exact_nominal_riser(self):
        root = Path(__file__).resolve().parents[3]
        task = yaml.safe_load(
            (root / "isaac_lab/g1_stairs/task.yaml").read_text()
        )
        terrain = task["terrain"]

        self.assertAlmostEqual(
            terrain["initial_training_riser_height_m"],
            NOMINAL_RISER_HEIGHT_M,
        )
        self.assertFalse(terrain["curriculum_enabled_initially"])
        self.assertEqual(
            terrain["planned_curriculum_riser_height_m"],
            [0.05, 0.20],
        )

    def test_three_treads_then_continuous_platform(self):
        stairs = nominal_staircase()
        boxes = stairs.collision_boxes()
        self.assertEqual(len(boxes), 5)  # approach + 3 steps + platform
        step_boxes = boxes[1:4]
        for actual, expected in zip(
            (box.size[2] for box in step_boxes), (0.2, 0.4, 0.6), strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(step_boxes[-1].center[0] + 0.125, 2.75)
        platform = boxes[-1]
        self.assertAlmostEqual(platform.center[0] - platform.size[0] / 2.0, 2.75)

    def test_invalid_dimensions_fail_early(self):
        with self.assertRaises(ValueError):
            StaircaseSpec(riser_height=0.0).collision_boxes()
        with self.assertRaises(ValueError):
            StaircaseSpec(start_clearance=2.0).collision_boxes()

    def test_training_joint_order_matches_deployment_contract(self):
        root = Path(__file__).resolve().parents[3]
        mapping = yaml.safe_load(
            (root / "simulation/config/joint_mapping.yaml").read_text()
        )
        robot = yaml.safe_load((root / "simulation/config/robot.yaml").read_text())
        self.assertEqual(tuple(mapping["policy_order"]), BODY_JOINTS)
        self.assertEqual(robot["control"]["mode_machine"], MODE_MACHINE)


if __name__ == "__main__":
    unittest.main()
