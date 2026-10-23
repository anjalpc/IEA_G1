import unittest
from types import SimpleNamespace

import torch

from simulation.isaac_lab.g1_stairs.geometry_locomotion_rewards import (
    planar_velocity_change_cost,
    approach_overspeed_cost,
    physical_step_event_mask,
    physical_step_phase_scale,
    unresolved_riser_strike_fall_event,
)



class ApproachOverspeedTests(unittest.TestCase):
    def test_only_excess_speed_before_riser(self):
        cost = approach_overspeed_cost(
            torch.tensor([0.50, 0.60, 0.80, 1.10, 1.10]),
            torch.full((5,), 0.50),
            torch.tensor([0.2, 0.4, 0.7, 0.99, 1.0]),
            1.0,
        )
        torch.testing.assert_close(
            cost, torch.tensor([0.0, 0.0, 0.16, 1.0, 0.0])
        )

    def test_command_change_and_no_underspeed_penalty(self):
        cost = approach_overspeed_cost(
            torch.tensor([0.4, 0.7]),
            torch.tensor([0.5, 0.8]),
            torch.tensor([0.0, 0.0]),
            1.0,
        )
        torch.testing.assert_close(cost, torch.zeros(2))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            approach_overspeed_cost(
                torch.zeros(1), torch.zeros(1),
                torch.zeros(1), 1.0, scale_mps=0.0
            )


class PlanarVelocityChangeCostTests(unittest.TestCase):
    def test_ignores_reset_and_vertical_velocity(self):
        value = planar_velocity_change_cost(
            torch.tensor([[1.0, 0.0, 5.0], [0.0, 0.0, 5.0]]),
            torch.zeros(2, 3),
            torch.tensor([False, True]),
            dt=0.02,
            full_cost_acceleration_mps2=3.0,
        )
        torch.testing.assert_close(value, torch.tensor([0.0, 0.0]))

    def test_is_bounded_and_sensitive_to_planar_change(self):
        value = planar_velocity_change_cost(
            torch.tensor([[0.03, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.zeros(2, 3),
            torch.tensor([True, True]),
            dt=0.02,
            full_cost_acceleration_mps2=3.0,
        )
        torch.testing.assert_close(value, torch.tensor([0.25, 1.0]))


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

    def test_phase_scale_favors_stair_ascent(self):
        scale = physical_step_phase_scale(
            torch.tensor([0.9, 1.2, 1.8]),
            first_riser_x=1.0,
            platform_start_x=1.75,
            approach_scale=0.10,
            ascent_scale=1.0,
            platform_scale=0.10,
        )
        torch.testing.assert_close(scale, torch.tensor([0.10, 1.0, 0.10]))


if __name__ == "__main__":
    unittest.main()
