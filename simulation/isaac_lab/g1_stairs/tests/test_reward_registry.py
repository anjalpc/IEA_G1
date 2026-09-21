import unittest

from simulation.isaac_lab.g1_stairs.reward_registry import (
    experiment_stage_weights,
    materialize_legacy_reward_config,
    raw_name,
    runtime_name,
)


class RewardRegistryTest(unittest.TestCase):

    def test_raw_runtime_roundtrip(self):
        self.assertEqual(
            runtime_name("pre_riser_foot_lift_raw"),
            "pre_riser_foot_lift",
        )
        self.assertEqual(
            raw_name("pre_riser_foot_lift"),
            "pre_riser_foot_lift_raw",
        )

    def test_stage_resolution(self):
        cfg = {
            "reward_experiment": {
                "active_stage": "test",
                "rewards": {
                    "stage_plan": {
                        "test": {
                            "upright": -5.0,
                            "pre_riser_foot_lift": 1.0,
                        }
                    }
                },
            }
        }

        stage, weights = experiment_stage_weights(cfg)

        self.assertEqual(stage, "test")
        self.assertEqual(weights["upright"], -5.0)
        self.assertEqual(
            weights["pre_riser_foot_lift"],
            1.0,
        )

    def test_materializes_legacy_schema(self):
        cfg = {
            "reward_experiment": {
                "active_stage": "test",
                "rewards": {
                    "stage_plan": {
                        "test": {
                            "upright": -5.0,
                            "pre_riser_foot_lift": 1.0,
                        }
                    }
                },
            }
        }

        materialize_legacy_reward_config(cfg)

        enabled = (
            cfg["rewards"]
            ["stage_plan"]
            ["test"]
            ["enabled_terms"]
        )

        self.assertEqual(
            enabled["upright_raw"],
            -5.0,
        )
        self.assertEqual(
            enabled["pre_riser_foot_lift_raw"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
