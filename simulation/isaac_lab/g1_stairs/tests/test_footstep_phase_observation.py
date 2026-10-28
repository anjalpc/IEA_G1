"""Unit-level observation binding tests (not live Isaac validation)."""
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch
import torch
from simulation.isaac_lab.g1_stairs import footstep_phase_observation as phase


class PhaseBindingTests(TestCase):
    def build(self):
        coord = NS(runtime=NS())
        env = NS(num_envs=2, cfg=NS(flat_ground_footsteps=False))
        return env, coord

    def test_width_targets_then_zero_timing(self):
        env, coord = self.build()
        targets = torch.arange(28, dtype=torch.float32).reshape(2, 14)
        with (patch.object(phase, 'ensure_coordinator', return_value=coord),
              patch.object(phase, 'target_observation', return_value=targets) as obs):
            a = phase.planned_footsteps_with_phase(env)
        self.assertEqual(tuple(a.shape), (2, 16))
        self.assertTrue(torch.equal(a[:, :14], targets))
        # Timing is policy-chosen: former clock channels carry no signal.
        self.assertEqual(a[:, 14:].abs().sum().item(), 0.0)
        self.assertEqual(obs.call_args.kwargs['foot_separation_m'], 0.235)

    def test_reset_calls_coordinator_reset(self):
        env, coord = self.build()
        selected = torch.tensor([1])
        calls = []
        coord.reset = lambda ids: calls.append(ids)
        with patch.object(phase, 'ensure_coordinator', return_value=coord):
            phase.reset_footsteps_with_phase(env, selected)
        self.assertIs(calls[0], selected)
