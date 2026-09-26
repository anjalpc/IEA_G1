import unittest
from pathlib import Path

import yaml

from simulation.isaac_lab.g1_stairs.contracts import (
    BODY_JOINTS,
    MODE_MACHINE,
    NOMINAL_RISER_HEIGHT_M,
)
from simulation.isaac_lab.g1_stairs.geometry import (
    StaircaseSpec,
    effective_platform_length,
    nominal_staircase,
)
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
)


class StaircaseGeometryTest(unittest.TestCase):
    def test_nominal_dimensions(self):
        stairs = nominal_staircase()
        self.assertAlmostEqual(stairs.total_rise, 0.60)
        self.assertAlmostEqual(stairs.stair_run, 0.75)
        self.assertAlmostEqual(stairs.course_length, 5.75)
        self.assertAlmostEqual(stairs.width, 1.20)
        self.assertEqual(stairs.fixed_start_xy_yaw, (1.0, 0.0, 0.0))

    def test_task_keeps_nominal_three_step_target(self):
        root = Path(__file__).resolve().parents[4]
        task = yaml.safe_load(
            (root / "simulation/isaac_lab/g1_stairs/task.yaml").read_text()
        )
        terrain = task["terrain"]
        self.assertAlmostEqual(
            terrain["initial_training_riser_height_m"],
            NOMINAL_RISER_HEIGHT_M,
        )
        self.assertEqual(terrain["planned_curriculum_riser_height_m"], [0.05, 0.20])

    def test_specialist_smoke_resolves_single_step_geometry(self):
        root = Path(__file__).resolve().parents[4]
        resolved, sources = resolve_config(
            root / "experiments/g1_stairs/configs/specialist_smoke.yaml"
        )
        self.assertEqual(
            sources,
            [
                "experiments/g1_stairs/configs/nominal_stairs_base.yaml",
                "experiments/g1_stairs/configs/specialist_base.yaml",
                "experiments/g1_stairs/configs/specialist_smoke.yaml",
            ],
        )
        self.assertTrue(resolved["specialist_stair"]["enabled"])
        self.assertEqual(resolved["terrain"]["step_count"], 1)
        self.assertAlmostEqual(resolved["terrain"]["riser_height_m"], 0.06)
        self.assertAlmostEqual(resolved["terrain"]["tread_depth_m"], 0.25)
        self.assertEqual(resolved["policy_initialization"]["type"], "random")

    def test_three_treads_then_continuous_platform(self):
        stairs = nominal_staircase()
        boxes = stairs.collision_boxes()
        self.assertEqual(len(boxes), 5)
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

    def test_training_joint_contract_is_29dof_mode5(self):
        self.assertEqual(len(BODY_JOINTS), 29)
        self.assertEqual(MODE_MACHINE, 5)
        self.assertEqual(BODY_JOINTS[0], "left_hip_pitch_joint")
        self.assertEqual(BODY_JOINTS[1], "right_hip_pitch_joint")
        self.assertEqual(BODY_JOINTS[2], "waist_yaw_joint")
        self.assertEqual(BODY_JOINTS[-1], "right_wrist_yaw_joint")


class TestVariableStepTerrainGeometry(unittest.TestCase):
    def test_platform_fills_fixed_tile_for_one_to_three_steps(self):
        expected = {1: 3.50, 2: 3.25, 3: 3.00}
        for step_count, expected_platform in expected.items():
            platform = effective_platform_length(
                terrain_size_x=5.75,
                approach_length=2.0,
                step_count=step_count,
                tread_depth=0.25,
                minimum_platform_length=3.0,
            )
            self.assertAlmostEqual(platform, expected_platform, places=8)
            self.assertAlmostEqual(2.0 + step_count * 0.25 + platform, 5.75, places=8)

    def test_invalid_step_count_rejected(self):
        with self.assertRaises(ValueError):
            effective_platform_length(
                terrain_size_x=5.75,
                approach_length=2.0,
                step_count=0,
                tread_depth=0.25,
                minimum_platform_length=3.0,
            )


if __name__ == "__main__":
    unittest.main()
