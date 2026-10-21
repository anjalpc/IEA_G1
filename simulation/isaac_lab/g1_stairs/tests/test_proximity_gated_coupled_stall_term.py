"""Runtime semantics for first-riser windowed anti-stall."""

import unittest
from types import SimpleNamespace

import torch

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from simulation.isaac_lab.g1_stairs.specialist_rewards import (
    proximity_gated_coupled_stall,
)


class Scene(dict):
    pass


class TestProximityGatedCoupledStallTerm(unittest.TestCase):

    def make_env(self, riser_height=0.04):
        data = SimpleNamespace(
            root_pos_w=torch.zeros((1, 3)),
            body_pos_w=torch.zeros((1, 2, 3)),
            body_quat_w=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]]
            ),
        )
        robot = SimpleNamespace(data=data)

        stairs = SimpleNamespace(
            start_clearance=1.0,
            tread_depth=0.25,
            riser_heights=None,
            riser_height_range=(riser_height, riser_height),
        )
        terrain = SimpleNamespace(
            cfg=SimpleNamespace(
                terrain_generator=SimpleNamespace(
                    sub_terrains={
                        "three_step_ascent": stairs,
                    }
                )
            )
        )

        scene = Scene()
        scene["robot"] = robot
        scene.env_origins = torch.zeros((1, 3))
        scene.terrain = terrain

        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            step_dt=0.02,
            scene=scene,
            command_manager=SimpleNamespace(
                get_command=lambda name: torch.tensor(
                    [[0.50, 0.0, 0.0]]
                )
            ),
        )

        return env

    @staticmethod
    def set_state(env, root_x, left_front_x, right_front_x):
        env.scene["robot"].data.root_pos_w[0, 0] = root_x

        # Identity-oriented collision envelope has front edge
        # body_x + 0.125 m.
        env.scene["robot"].data.body_pos_w[0, 0, 0] = (
            left_front_x - 0.125
        )
        env.scene["robot"].data.body_pos_w[0, 1, 0] = (
            right_front_x - 0.125
        )

    def make_term(self, env):
        cfg = SimpleNamespace(
            params={
                "window_s": 0.50,
            }
        )
        return proximity_gated_coupled_stall(
            cfg=cfg,
            env=env,
        )

    def call(self, term, env):
        return term(
            env,
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
            command_name="base_velocity",
            asset_cfg=SimpleNamespace(name="robot"),
            foot_cfg=SimpleNamespace(body_ids=[0, 1]),
        )

    def test_no_penalty_until_full_window_exists(self):
        env = self.make_env()
        term = self.make_term(env)

        self.set_state(env, 0.50, 0.75, 0.75)

        for _ in range(25):
            value = self.call(term, env)
            self.assertEqual(value.item(), 0.0)

        value = self.call(term, env)

        self.assertGreater(value.item(), 0.0)

    def test_healthy_windowed_progress_has_no_penalty(self):
        env = self.make_env()
        term = self.make_term(env)

        for step in range(26):
            advance = 0.004 * step
            self.set_state(
                env,
                root_x=0.30 + advance,
                left_front_x=0.60 + advance,
                right_front_x=0.60 + advance,
            )
            value = self.call(term, env)

        # 25 * 0.004 = 0.10 m in 0.5 s > required 0.05 m.
        self.assertEqual(value.item(), 0.0)

    def test_first_riser_crossing_latches_term_off(self):
        env = self.make_env()
        term = self.make_term(env)

        self.set_state(env, 0.50, 0.75, 0.75)

        for _ in range(26):
            value = self.call(term, env)

        self.assertGreater(value.item(), 0.0)

        # One sole front crosses the first-riser plane.
        self.set_state(env, 0.55, 1.01, 0.75)
        self.assertEqual(
            self.call(term, env).item(),
            0.0,
        )

        # Returning below the plane must not reactivate this
        # first-riser approach term during the same episode.
        self.set_state(env, 0.50, 0.75, 0.75)
        self.assertEqual(
            self.call(term, env).item(),
            0.0,
        )

    def test_reset_clears_crossing_latch_and_history(self):
        env = self.make_env()
        term = self.make_term(env)

        self.set_state(env, 0.50, 1.01, 0.75)
        self.call(term, env)

        term.reset(torch.tensor([0]))

        self.set_state(env, 0.50, 0.75, 0.75)

        # Reset must require a fresh 0.5 s history.
        for _ in range(25):
            self.assertEqual(
                self.call(term, env).item(),
                0.0,
            )

        self.assertGreater(
            self.call(term, env).item(),
            0.0,
        )

    def test_flat_terrain_never_activates(self):
        env = self.make_env(riser_height=0.0)
        term = self.make_term(env)

        self.set_state(env, 0.50, 0.75, 0.75)

        for _ in range(26):
            value = self.call(term, env)

        self.assertEqual(value.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
