import unittest
import ast
from pathlib import Path

import yaml

from simulation.isaac_lab.g1_stairs.contracts import (
    BODY_JOINTS,
    MODE_MACHINE,
    NOMINAL_RISER_HEIGHT_M,
)
from simulation.isaac_lab.g1_stairs.geometry import StaircaseSpec, nominal_staircase
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import resolve_config


class StaircaseGeometryTest(unittest.TestCase):
    def test_nominal_dimensions(self):
        stairs = nominal_staircase()
        self.assertAlmostEqual(stairs.total_rise, 0.60)
        self.assertAlmostEqual(stairs.stair_run, 0.75)
        self.assertAlmostEqual(stairs.course_length, 5.75)
        self.assertAlmostEqual(stairs.width, 1.20)
        self.assertEqual(stairs.fixed_start_xy_yaw, (1.0, 0.0, 0.0))

    def test_initial_training_uses_exact_nominal_riser(self):
        root = Path(__file__).resolve().parents[4]
        task = yaml.safe_load(
            (root / "simulation/isaac_lab/g1_stairs/task.yaml").read_text()
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

        training = task["training"]
        self.assertEqual(
            training["observation_history"]["proprioception_steps"],
            5,
        )
        self.assertEqual(
            training["observation_history"]["terrain_height_scan_steps"],
            1,
        )
        self.assertEqual(
            training["initial_randomization"],
            {
                "policy_observation_corruption": False,
                "external_force_reset": False,
                "terrain_curriculum": False,
                "velocity_curriculum": False,
            },
        )

        smoke = yaml.safe_load(
            (root / "experiments/g1_stairs/configs/002_stairs_smoke.yaml").read_text()
        )
        self.assertEqual(smoke["terrain"]["course_size_m"], [5.75, 1.2])
        self.assertAlmostEqual(
            smoke["terrain"]["riser_height_m"]["initial_smoke_level"],
            NOMINAL_RISER_HEIGHT_M,
        )
        self.assertFalse(smoke["observations"]["corruption_enabled"])
        self.assertFalse(smoke["curriculum"]["terrain_riser_height"]["enabled"])
        self.assertFalse(
            smoke["curriculum"]["command_curriculum"][
                "inherited_from_upstream"
            ]
        )
        self.assertEqual(
            set(smoke["rewards"]["active_terms"]),
            {"forward_progress", "stair_progress"},
        )
        self.assertFalse(smoke["rewards"]["inherited_flat_ground_terms_active"])

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
        root = Path(__file__).resolve().parents[4]
        mapping = yaml.safe_load(
            (root / "simulation/mujoco/config/joint_mapping.yaml").read_text()
        )
        robot = yaml.safe_load((root / "simulation/mujoco/config/robot.yaml").read_text())
        self.assertEqual(tuple(mapping["policy_order"]), BODY_JOINTS)
        self.assertEqual(robot["control"]["mode_machine"], MODE_MACHINE)

    def test_reward_stage_plan_preserves_a_and_adds_only_upright_in_b(self):
        root = Path(__file__).resolve().parents[4]
        experiment = yaml.safe_load(
            (root / "experiments/g1_stairs/configs/003_nominal_reward_debug.yaml").read_text()
        )
        stage_plan = experiment["rewards"]["stage_plan"]
        experiment_a = stage_plan["A_progress_only"][
            "enabled_terms"
        ]
        self.assertEqual(
            experiment_a,
            {"forward_progress_raw": 1.0, "stair_progress_raw": 1.0},
        )
        self.assertEqual(
            stage_plan["B_add_upright"]["enabled_terms"],
            {"upright_raw": -1.0},
        )
        self.assertEqual(
            experiment["reward_experiment"]["active_stage"],
            "B_add_upright",
        )
        self.assertEqual(
            experiment["rewards"]["always_logged"]["upright_raw"],
            {
                "formula": "squared_xy_projected_gravity",
                "interpretation": "zero_when_upright_increases_with_tilt",
                "active_weight": -1.0,
            },
        )

    def test_experiment_b_runtime_rewards_are_exact(self):
        root = Path(__file__).resolve().parents[4]

        tree = ast.parse(
            (root / "simulation/isaac_lab/g1_stairs/env_cfg.py").read_text(encoding="utf-8")
        )
        reward_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "StairRewardsCfg"
        )
        configured = {
            target.id
            for node in reward_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(
            configured,
            {"forward_progress", "stair_progress", "upright"},
        )
        assignments = {
            node.targets[0].id: node.value
            for node in reward_class.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        expected = {
            "forward_progress": ("stair_rewards.forward_progress", 1.0),
            "stair_progress": ("stair_rewards.stair_progress", 1.0),
            "upright": ("mdp.flat_orientation_l2", -1.0),
        }
        for name, (function, weight) in expected.items():
            call = assignments[name]
            keywords = {item.arg: item.value for item in call.keywords}
            self.assertEqual(ast.unparse(keywords["func"]), function)
            self.assertEqual(ast.literal_eval(keywords["weight"]), weight)

    def test_experiment_b_smoke_changes_only_iteration_count(self):
        root = Path(__file__).resolve().parents[4]
        smoke_path = root / "experiments/g1_stairs/configs/004_experiment_b_smoke.yaml"
        leaf = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(leaf),
            {"schema_version", "experiment", "inheritance", "execution"},
        )
        self.assertEqual(leaf["execution"], {"maximum_iterations": 1})

        resolved, sources = resolve_config(smoke_path)
        self.assertEqual(
            sources,
            [
                "experiments/g1_stairs/configs/001_upstream_flat_ground.yaml",
                "experiments/g1_stairs/configs/002_stairs_smoke.yaml",
                "experiments/g1_stairs/configs/003_nominal_reward_debug.yaml",
                "experiments/g1_stairs/configs/004_experiment_b_smoke.yaml",
            ],
        )
        self.assertEqual(resolved["execution"]["maximum_iterations"], 1)
        self.assertEqual(resolved["execution"]["environments"], 128)
        self.assertEqual(resolved["execution"]["training_seed"], 42)
        self.assertEqual(
            resolved["reward_experiment"]["active_stage"],
            "B_add_upright",
        )
        self.assertEqual(
            resolved["rewards"]["stage_plan"]["B_add_upright"][
                "enabled_terms"
            ],
            {"upright_raw": -1.0},
        )

    def test_metrics_gate_is_open_after_live_smoke(self):
        root = Path(__file__).resolve().parents[4]
        experiment = yaml.safe_load(
            (root / "experiments/g1_stairs/configs/003_nominal_reward_debug.yaml").read_text()
        )
        gates = experiment["launch_gates"]
        self.assertTrue(gates["independent_evaluator_implemented"])
        self.assertTrue(gates["essential_metrics_implemented"])
        self.assertTrue(gates["allow_launch"])
        self.assertEqual(
            set(experiment["metric_deferrals"]),
            {
                "approximate_kl",
                "clip_fraction",
                "inactive_reward_components",
                "automatic_periodic_evaluation",
            },
        )


if __name__ == "__main__":
    unittest.main()
