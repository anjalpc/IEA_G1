import types
import unittest

from simulation.isaac_lab.g1_stairs.command_runtime import (
    apply_stair_heading_command,
)


class CommandRuntimeTests(unittest.TestCase):
    def _env_cfg(self):
        ranges = types.SimpleNamespace(
            heading=(-1.0, 1.0),
            ang_vel_z=(0.0, 0.0),
        )
        limit_ranges = types.SimpleNamespace(
            heading=(-1.0, 1.0),
            ang_vel_z=(0.0, 0.0),
        )
        base_velocity = types.SimpleNamespace(
            heading_command=False,
            rel_heading_envs=0.0,
            rel_standing_envs=0.0,
            heading_control_stiffness=0.5,
            ranges=ranges,
            limit_ranges=limit_ranges,
        )
        return types.SimpleNamespace(
            commands=types.SimpleNamespace(
                base_velocity=base_velocity,
            )
        )

    def test_enabled_heading_preserves_three_channel_command_contract(self):
        env_cfg = self._env_cfg()
        resolved = {
            "command": {
                "stair_heading": {
                    "enabled": True,
                    "target_heading_rad": 0.0,
                    "control_stiffness": 1.0,
                    "max_yaw_rate_radps": 0.6,
                }
            }
        }

        summary = apply_stair_heading_command(env_cfg, resolved)
        base = env_cfg.commands.base_velocity

        self.assertEqual(
            summary,
            {
                "target_heading_rad": 0.0,
                "control_stiffness": 1.0,
                "max_yaw_rate_radps": 0.6,
            },
        )
        self.assertTrue(base.heading_command)
        self.assertEqual(base.rel_heading_envs, 1.0)
        self.assertEqual(base.ranges.heading, (0.0, 0.0))
        self.assertEqual(base.ranges.ang_vel_z, (-0.6, 0.6))
        self.assertEqual(base.limit_ranges.ang_vel_z, (-0.6, 0.6))

    def test_disabled_heading_does_not_modify_command(self):
        env_cfg = self._env_cfg()
        original = env_cfg.commands.base_velocity.ranges.ang_vel_z
        summary = apply_stair_heading_command(
            env_cfg,
            {"command": {"stair_heading": {"enabled": False}}},
        )
        self.assertIsNone(summary)
        self.assertFalse(env_cfg.commands.base_velocity.heading_command)
        self.assertEqual(
            env_cfg.commands.base_velocity.ranges.ang_vel_z,
            original,
        )


if __name__ == "__main__":
    unittest.main()
