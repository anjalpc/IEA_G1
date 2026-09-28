import types
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TerrainCurriculumTests(unittest.TestCase):
    def test_platform_success_requires_progress_rise_and_upright(self):
        from simulation.isaac_lab.g1_stairs import terrain_curriculum

        terrain = types.SimpleNamespace(
            terrain_levels=torch.tensor([1, 1, 1], dtype=torch.long),
            max_terrain_level=8,
        )

        def update_env_origins(env_ids, move_up, move_down):
            terrain.terrain_levels[env_ids] += (
                move_up.to(torch.long) - move_down.to(torch.long)
            )

        terrain.update_env_origins = update_env_origins
        terrain.cfg = types.SimpleNamespace(
            terrain_generator=types.SimpleNamespace(
                sub_terrains={
                    "three_step_ascent": types.SimpleNamespace(
                        riser_heights=(
                            0.0,
                            0.04,
                            0.06,
                            0.08,
                            0.10,
                            0.12,
                            0.14,
                            0.16,
                        ),
                        start_clearance=1.0,
                        step_count=3,
                        tread_depth=0.25,
                    )
                }
            )
        )

        robot = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=torch.tensor(
                    [
                        [1.90, 0.0, 0.80],
                        [1.90, 0.0, 0.60],
                        [1.90, 0.0, 0.80],
                    ]
                ),
                projected_gravity_b=torch.tensor(
                    [
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -0.50],
                    ]
                ),
            )
        )

        scene = types.SimpleNamespace(
            terrain=terrain,
            env_origins=torch.zeros((3, 3)),
        )
        scene.__getitem__ = lambda self, name: robot

        # SimpleNamespace special methods are resolved on the type, so use a
        # tiny wrapper for scene["robot"].
        class Scene(types.SimpleNamespace):
            def __getitem__(self, name):
                self.assert_name = name
                return robot

        env = types.SimpleNamespace(
            scene=Scene(
                terrain=terrain,
                env_origins=torch.zeros((3, 3)),
            ),
            num_envs=3,
            episode_length_buf=torch.tensor(
                [100, 100, 100],
                dtype=torch.long,
            ),
        )

        result = terrain_curriculum.stair_platform_levels(
            env,
            torch.tensor([0, 1, 2]),
            platform_margin_m=0.10,
            minimum_upright_gravity_z=0.75,
            minimum_base_clearance_m=0.55,
        )

        self.assertEqual(terrain.terrain_levels.tolist(), [2, 0, 0])
        self.assertAlmostEqual(result["reset_success_rate"], 1.0 / 3.0)
        self.assertEqual(
            env._g1_stair_curriculum_attempts.tolist(),
            [0, 3, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(
            env._g1_stair_curriculum_successes.tolist(),
            [0, 1, 0, 0, 0, 0, 0, 0],
        )


    def test_initial_reset_does_not_demote_curriculum(self):
        from simulation.isaac_lab.g1_stairs import terrain_curriculum

        terrain = types.SimpleNamespace(
            terrain_levels=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            max_terrain_level=8,
        )

        def update_env_origins(env_ids, move_up, move_down):
            terrain.terrain_levels[env_ids] += (
                move_up.to(torch.long)
                - move_down.to(torch.long)
            )

        terrain.update_env_origins = update_env_origins
        terrain.cfg = types.SimpleNamespace(
            terrain_generator=types.SimpleNamespace(
                sub_terrains={
                    "three_step_ascent": types.SimpleNamespace(
                        riser_heights=(
                            0.0,
                            0.04,
                            0.06,
                            0.08,
                            0.10,
                            0.12,
                            0.14,
                            0.16,
                        ),
                        start_clearance=1.0,
                        step_count=3,
                        tread_depth=0.25,
                    )
                }
            )
        )

        robot = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=torch.tensor(
                    [
                        [0.0, 0.0, 0.80],
                        [0.0, 0.0, 0.80],
                    ]
                ),
                projected_gravity_b=torch.tensor(
                    [
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                    ]
                ),
            )
        )

        class Scene(types.SimpleNamespace):
            def __getitem__(self, name):
                return robot

        env = types.SimpleNamespace(
            scene=Scene(
                terrain=terrain,
                env_origins=torch.zeros((2, 3)),
            ),
            num_envs=2,
            episode_length_buf=torch.zeros(
                2,
                dtype=torch.long,
            ),
        )

        terrain_curriculum.stair_platform_levels(
            env,
            torch.tensor([0, 1]),
        )

        self.assertEqual(
            terrain.terrain_levels.tolist(),
            [0, 1],
        )

        self.assertEqual(
            env._g1_stair_curriculum_attempts.tolist(),
            [0, 0, 0, 0, 0, 0, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
