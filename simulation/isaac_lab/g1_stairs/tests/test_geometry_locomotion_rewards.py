import unittest
from types import SimpleNamespace

import torch

from simulation.isaac_lab.g1_stairs.geometry_locomotion_rewards import (
    physical_step_event_mask,
    unresolved_riser_strike_fall_event,
)


class UnresolvedRiserStrikeFallEventTests(unittest.TestCase):

    def make_env(self, terminated, invalid, dt=0.02):
        return SimpleNamespace(
            reset_terminated=torch.tensor(terminated, dtype=torch.bool),
            step_dt=dt,
            _g1_v04_acquisition=SimpleNamespace(
                crossing_attempt_invalid=torch.tensor(
                    invalid, dtype=torch.bool
                )
            ),
        )

    def test_pays_once_for_native_fall_with_unresolved_strike(self):
        env = self.make_env(
            [True, True, False],
            [
                [True, False],
                [False, False],
                [True, True],
            ],
        )

        value = unresolved_riser_strike_fall_event(env)

        torch.testing.assert_close(
            value,
            torch.tensor([50.0, 0.0, 0.0]),
        )

    def test_safe_recovery_cleared_latch_is_not_penalized(self):
        env = self.make_env(
            [True],
            [[False, False]],
        )

        torch.testing.assert_close(
            unresolved_riser_strike_fall_event(env),
            torch.tensor([0.0]),
        )

    def test_requires_acquisition_runtime(self):
        env = SimpleNamespace(
            reset_terminated=torch.tensor([True]),
            step_dt=0.02,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "acquisition runtime",
        ):
            unresolved_riser_strike_fall_event(env)

    def test_requires_positive_dt(self):
        env = self.make_env(
            [True],
            [[True, False]],
            dt=0.0,
        )

        with self.assertRaisesRegex(ValueError, "step dt"):
            unresolved_riser_strike_fall_event(env)


class PhysicalStepEventMaskTests(unittest.TestCase):
    def test_requires_complete_alternating_physical_step(self):
        qualified = physical_step_event_mask(
            torch.tensor([0.20, 0.10, 0.20, 0.20]),
            torch.tensor([0.020, 0.020, 0.005, 0.020]),
            torch.tensor([0.20, 0.20, 0.20, 0.50]),
            torch.tensor([True, True, True, False]),
            minimum_forward_travel_m=0.15,
            minimum_vertical_excursion_m=0.015,
            maximum_touchdown_speed_mps=0.35,
        )
        torch.testing.assert_close(
            qualified,
            torch.tensor([True, False, False, False]),
        )

if __name__ == "__main__":
    unittest.main()
