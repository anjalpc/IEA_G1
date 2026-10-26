"""Behavior tests for command-gated, course-wide anti-stall."""
import unittest
from types import SimpleNamespace

import torch
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from simulation.isaac_lab.g1_stairs.specialist_rewards import (
    course_windowed_forward_stall,
)


class Scene(dict):
    pass


class TestCourseWindowedForwardStall(unittest.TestCase):

    def make_env(self):
        data = SimpleNamespace(
            root_pos_w=torch.zeros((1, 3)),
            body_pos_w=torch.zeros((1, 2, 3)),
            body_quat_w=torch.tensor([[
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]]),
        )
        scene = Scene(robot=SimpleNamespace(data=data))
        scene.env_origins = torch.zeros((1, 3))
        command = torch.tensor([[0.5, 0.0, 0.0]])
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            step_dt=0.02,
            scene=scene,
            command_manager=SimpleNamespace(
                get_command=lambda name: command,
            ),
        )
        term = course_windowed_forward_stall(
            cfg=SimpleNamespace(params={"window_s": 0.50}),
            env=env,
        )
        return env, command, term

    @staticmethod
    def step(env, term, x):
        data = env.scene["robot"].data
        data.root_pos_w[0, 0] = x
        data.body_pos_w[0, :, 0] = x
        return term(
            env,
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
            command_name="base_velocity",
            asset_cfg=SimpleNamespace(name="robot"),
            foot_cfg=SimpleNamespace(body_ids=[0, 1]),
        ).item()

    def test_no_penalty_before_full_window(self):
        env, command, term = self.make_env()
        for _ in range(25):
            self.assertEqual(self.step(env, term, 0.5), 0.0)
        self.assertAlmostEqual(self.step(env, term, 0.5), 1.0)

    def test_zero_command_never_penalized(self):
        env, command, term = self.make_env()
        command[0, 0] = 0.0
        for _ in range(40):
            self.assertEqual(self.step(env, term, 0.5), 0.0)

    def test_healthy_forward_progress(self):
        env, command, term = self.make_env()
        for i in range(40):
            value = self.step(env, term, 0.5 + 0.004 * i)
        self.assertEqual(value, 0.0)

    def test_sustained_backward_retreat(self):
        env, command, term = self.make_env()
        for i in range(30):
            value = self.step(env, term, 1.0 - 0.004 * i)
        self.assertAlmostEqual(value, 1.0)

    def test_independent_of_stair_number(self):
        values = []
        for x in (0.5, 1.5, 2.5):
            env, command, term = self.make_env()
            for _ in range(30):
                value = self.step(env, term, x)
            values.append(value)
        self.assertEqual(values, [1.0, 1.0, 1.0])

    def test_reset_discards_history(self):
        env, command, term = self.make_env()
        for _ in range(30):
            self.step(env, term, 0.5)
        term.reset(torch.tensor([0]))
        for _ in range(25):
            self.assertEqual(self.step(env, term, 0.5), 0.0)
        self.assertAlmostEqual(self.step(env, term, 0.5), 1.0)


if __name__ == "__main__":
    unittest.main()
