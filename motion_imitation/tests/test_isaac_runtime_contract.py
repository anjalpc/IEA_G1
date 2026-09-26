from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


MOTION_ROOT = Path(__file__).resolve().parents[1]

if str(MOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_ROOT))

from g1_stair_imitation.isaac.runtime_contract import (
    EXPECTED_ACTION_DIM,
    FIRST_RISER_COURSE_X_M,
    RISER_HEIGHT_M,
    STEP_COUNT,
    TERRAIN_KEY,
    TREAD_DEPTH_M,
    configure_imitation_preflight_env,
    validate_action_cfg,
)


class IsaacRuntimeContractTests(unittest.TestCase):
    def _fake_env_cfg(self):
        stair = SimpleNamespace(
            riser_height_range=(0.06, 0.06),
            tread_depth=0.25,
            step_count=1,
            approach_length=2.0,
            platform_length=3.0,
            start_clearance=1.0,
        )
        generator = SimpleNamespace(
            num_rows=8,
            num_cols=16,
            sub_terrains={TERRAIN_KEY: stair},
        )
        action = SimpleNamespace(
            scale=0.25,
            use_default_offset=True,
            joint_names=[f"joint_{i}" for i in range(EXPECTED_ACTION_DIM)],
        )
        return SimpleNamespace(
            scene=SimpleNamespace(
                terrain=SimpleNamespace(
                    terrain_generator=generator,
                )
            ),
            actions=SimpleNamespace(
                JointPositionAction=action,
            ),
        )

    def test_local_geometry_override(self):
        cfg = self._fake_env_cfg()
        stair = configure_imitation_preflight_env(cfg)

        self.assertEqual(stair.riser_height_range, (RISER_HEIGHT_M, RISER_HEIGHT_M))
        self.assertEqual(stair.tread_depth, TREAD_DEPTH_M)
        self.assertEqual(stair.step_count, STEP_COUNT)
        self.assertEqual(stair.start_clearance, FIRST_RISER_COURSE_X_M)
        self.assertEqual(cfg.scene.terrain.terrain_generator.num_rows, 1)
        self.assertEqual(cfg.scene.terrain.terrain_generator.num_cols, 1)

    def test_action_contract(self):
        cfg = self._fake_env_cfg()
        scale, default_offset, dim = validate_action_cfg(cfg)

        self.assertEqual(scale, 0.25)
        self.assertTrue(default_offset)
        self.assertEqual(dim, EXPECTED_ACTION_DIM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
