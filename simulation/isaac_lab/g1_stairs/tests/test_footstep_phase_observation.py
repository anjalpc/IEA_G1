"""Unit-level observation binding tests (not live Isaac validation)."""
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch
import torch
from simulation.isaac_lab.g1_stairs import footstep_phase_observation as phase


class PhaseBindingTests(TestCase):
    def build(self):
        coord = NS(last_step=3, runtime=NS())
        coord.clock = NS(features=lambda **kwargs: (
            torch.tensor([[0., 1.], [1., 0.]]), None, None))
        coord.runtime.targets = lambda **kwargs: (
            None, torch.tensor([[0, 1], [1, 0]]),
            torch.ones((2, 2), dtype=torch.bool))
        return NS(num_envs=2, common_step_counter=3), coord

    def test_width_and_order_without_duplicate_update(self):
        env, coord = self.build()
        with (patch.object(phase, 'ensure_coordinator', return_value=coord),
              patch.object(phase, 'audit_after_step') as audit,
              patch.object(phase, 'target_observation', return_value=torch.ones((2,14))),
              patch.object(phase, 'course_tensors', return_value={})):
            a = phase.planned_footsteps_with_phase(env)
            self.assertEqual(a.shape, (2,16))
            self.assertEqual(a[0,-2:].tolist(), [0.,1.])
            audit.assert_not_called()

    def test_update_before_observation_on_new_policy_step(self):
        env, coord = self.build()
        coord.last_step = 2
        order = []
        with (patch.object(phase, 'ensure_coordinator', return_value=coord),
              patch.object(phase, 'audit_after_step', side_effect=lambda e: order.append('contact')),
              patch.object(phase, 'target_observation', side_effect=lambda *a, **k: (
                  order.append('read') or torch.ones((2,14)))),
              patch.object(phase, 'course_tensors', return_value={})):
            phase.planned_footsteps_with_phase(env)
        self.assertEqual(order, ['contact','read'])

    def test_reset_calls_coordinator_reset(self):
        env, coord = self.build()
        selected = torch.tensor([1])
        calls=[]
        coord.reset=lambda ids: calls.append(ids)
        with patch.object(phase, 'ensure_coordinator', return_value=coord):
            phase.reset_footsteps_with_phase(env, selected)
        self.assertIs(calls[0], selected)
