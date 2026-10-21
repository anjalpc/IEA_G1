"""Pure tests for proximity-gated, windowed coupled anti-stall severity."""

import unittest

import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    proximity_gated_coupled_stall_severity,
)


class TestProximityGatedCoupledStall(unittest.TestCase):

    def test_healthy_window_progress_has_zero_severity(self):
        severity = proximity_gated_coupled_stall_severity(
            foot_riser_distance_norm=torch.tensor([[0.4, 0.6]]),
            root_progress_m=torch.tensor([0.05]),
            mean_foot_progress_m=torch.tensor([0.06]),
            stair_active=torch.tensor([True]),
            forward_command_mps=torch.tensor([0.50]),
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
        )
        torch.testing.assert_close(severity, torch.tensor([0.0]))

    def test_partial_stall_is_continuous_and_bounded(self):
        severity = proximity_gated_coupled_stall_severity(
            foot_riser_distance_norm=torch.tensor([[0.4, 0.6]]),
            root_progress_m=torch.tensor([0.02]),
            mean_foot_progress_m=torch.tensor([0.03]),
            stair_active=torch.tensor([True]),
            forward_command_mps=torch.tensor([0.50]),
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
        )

        # Required progress = 0.05 m.
        # Coupled progress = min(0.02, 0.03) = 0.02 m.
        # Deficit = 0.60.
        # Proximity = 1 - 0.40 = 0.60.
        # Severity = 0.36.
        torch.testing.assert_close(
            severity,
            torch.tensor([0.36]),
        )

    def test_retreat_reaches_full_progress_deficit(self):
        severity = proximity_gated_coupled_stall_severity(
            foot_riser_distance_norm=torch.tensor([[0.2, 0.5]]),
            root_progress_m=torch.tensor([-0.01]),
            mean_foot_progress_m=torch.tensor([-0.02]),
            stair_active=torch.tensor([True]),
            forward_command_mps=torch.tensor([0.50]),
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
        )

        # Full progress deficit, but still proximity-scaled.
        torch.testing.assert_close(
            severity,
            torch.tensor([0.80]),
        )

    def test_far_from_riser_has_zero_severity(self):
        severity = proximity_gated_coupled_stall_severity(
            foot_riser_distance_norm=torch.tensor([[1.0, 1.0]]),
            root_progress_m=torch.tensor([0.0]),
            mean_foot_progress_m=torch.tensor([0.0]),
            stair_active=torch.tensor([True]),
            forward_command_mps=torch.tensor([0.50]),
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
        )
        torch.testing.assert_close(severity, torch.tensor([0.0]))

    def test_pressure_switches_off_after_first_foot_crosses_riser(self):
        severity = proximity_gated_coupled_stall_severity(
            foot_riser_distance_norm=torch.tensor([[0.0, 0.4]]),
            root_progress_m=torch.tensor([0.0]),
            mean_foot_progress_m=torch.tensor([0.0]),
            stair_active=torch.tensor([True]),
            forward_command_mps=torch.tensor([0.50]),
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
        )
        torch.testing.assert_close(severity, torch.tensor([0.0]))

    def test_inactive_stair_has_zero_severity(self):
        severity = proximity_gated_coupled_stall_severity(
            foot_riser_distance_norm=torch.tensor([[0.1, 0.2]]),
            root_progress_m=torch.tensor([0.0]),
            mean_foot_progress_m=torch.tensor([0.0]),
            stair_active=torch.tensor([False]),
            forward_command_mps=torch.tensor([0.50]),
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
        )
        torch.testing.assert_close(severity, torch.tensor([0.0]))

    def test_no_forward_command_has_zero_severity(self):
        severity = proximity_gated_coupled_stall_severity(
            foot_riser_distance_norm=torch.tensor([[0.1, 0.2]]),
            root_progress_m=torch.tensor([0.0]),
            mean_foot_progress_m=torch.tensor([0.0]),
            stair_active=torch.tensor([True]),
            forward_command_mps=torch.tensor([0.0]),
            minimum_forward_speed_mps=0.10,
            window_s=0.50,
        )
        torch.testing.assert_close(severity, torch.tensor([0.0]))


if __name__ == "__main__":
    unittest.main()
