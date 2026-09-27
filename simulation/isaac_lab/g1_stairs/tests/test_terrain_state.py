import types
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from simulation.isaac_lab.g1_stairs.terrain_state import riser_height_per_env


@unittest.skipIf(torch is None, "torch is unavailable")
class TerrainStateTests(unittest.TestCase):
    def _env(self, levels, heights):
        cfg = types.SimpleNamespace(
            riser_heights=tuple(heights),
            riser_height_range=(0.12, 0.12),
            difficulty_bins=len(heights),
        )

        generator = types.SimpleNamespace(
            sub_terrains={
                "three_step_ascent": cfg,
            }
        )

        terrain = types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                terrain_generator=generator,
            ),
            terrain_levels=torch.tensor(
                levels,
                dtype=torch.long,
            ),
        )

        scene = types.SimpleNamespace(
            terrain=terrain,
            env_origins=torch.zeros(
                (len(levels), 3),
                dtype=torch.float32,
            ),
        )

        return types.SimpleNamespace(
            scene=scene,
            num_envs=len(levels),
        )

    def test_exact_discrete_heights_follow_terrain_levels(self):
        env = self._env(
            [0, 1, 2, 3],
            [0.0, 0.12, 0.14, 0.16],
        )

        actual = riser_height_per_env(env)

        torch.testing.assert_close(
            actual,
            torch.tensor(
                [0.0, 0.12, 0.14, 0.16]
            ),
        )

    def test_invalid_level_is_rejected(self):
        env = self._env(
            [4],
            [0.0, 0.12, 0.14, 0.16],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "outside riser_heights",
        ):
            riser_height_per_env(env)


if __name__ == "__main__":
    unittest.main()
