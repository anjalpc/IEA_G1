"""State-history tests for windowed coupled locomotor progress."""

import unittest

import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    WindowedCoupledProgress,
)


class TestWindowedCoupledProgress(unittest.TestCase):

    def test_emits_exact_window_displacement_only_when_ready(self):
        history = WindowedCoupledProgress(
            num_envs=1,
            window_steps=3,
            device="cpu",
        )

        samples = [
            (0.00, 0.00),
            (0.02, 0.03),
            (0.05, 0.06),
            (0.08, 0.10),
        ]

        for index, (root_x, foot_x) in enumerate(samples):
            root_progress, foot_progress, ready = history.update(
                torch.tensor([root_x]),
                torch.tensor([foot_x]),
            )

            if index < 3:
                self.assertFalse(bool(ready[0]))
                self.assertEqual(root_progress.item(), 0.0)
                self.assertEqual(foot_progress.item(), 0.0)
            else:
                self.assertTrue(bool(ready[0]))
                self.assertAlmostEqual(
                    root_progress.item(),
                    0.08,
                    places=6,
                )
                self.assertAlmostEqual(
                    foot_progress.item(),
                    0.10,
                    places=6,
                )

    def test_sliding_window_advances_one_step_at_a_time(self):
        history = WindowedCoupledProgress(
            num_envs=1,
            window_steps=2,
            device="cpu",
        )

        history.update(
            torch.tensor([0.00]),
            torch.tensor([0.00]),
        )
        history.update(
            torch.tensor([0.02]),
            torch.tensor([0.03]),
        )

        root_progress, foot_progress, ready = history.update(
            torch.tensor([0.05]),
            torch.tensor([0.07]),
        )

        self.assertTrue(bool(ready[0]))
        self.assertAlmostEqual(root_progress.item(), 0.05, places=6)
        self.assertAlmostEqual(foot_progress.item(), 0.07, places=6)

        root_progress, foot_progress, ready = history.update(
            torch.tensor([0.06]),
            torch.tensor([0.08]),
        )

        self.assertTrue(bool(ready[0]))
        self.assertAlmostEqual(root_progress.item(), 0.04, places=6)
        self.assertAlmostEqual(foot_progress.item(), 0.05, places=6)

    def test_reset_environment_cannot_read_previous_episode_history(self):
        history = WindowedCoupledProgress(
            num_envs=2,
            window_steps=2,
            device="cpu",
        )

        history.update(
            torch.tensor([0.00, 0.00]),
            torch.tensor([0.00, 0.00]),
        )
        history.update(
            torch.tensor([0.05, 0.05]),
            torch.tensor([0.05, 0.05]),
        )
        history.update(
            torch.tensor([0.10, 0.10]),
            torch.tensor([0.10, 0.10]),
        )

        history.reset(torch.tensor([0]))

        root_progress, foot_progress, ready = history.update(
            torch.tensor([0.01, 0.15]),
            torch.tensor([0.01, 0.15]),
        )

        self.assertFalse(bool(ready[0]))
        self.assertEqual(root_progress[0].item(), 0.0)
        self.assertEqual(foot_progress[0].item(), 0.0)

        self.assertTrue(bool(ready[1]))
        self.assertAlmostEqual(root_progress[1].item(), 0.10, places=6)
        self.assertAlmostEqual(foot_progress[1].item(), 0.10, places=6)

    def test_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            WindowedCoupledProgress(
                num_envs=1,
                window_steps=0,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
