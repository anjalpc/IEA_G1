"""Run with: python -m unittest simulation.isaac_lab.g1_stairs.tests.test_d2_recovery."""
import unittest
from types import SimpleNamespace
import torch
from simulation.isaac_lab.g1_stairs.geometry_episode_runtime import GeometryEpisodeRuntime
from simulation.isaac_lab.g1_stairs.strict_stair_gait import StrictStairGait
from simulation.isaac_lab.g1_stairs.tests.test_geometry_episode import tracker, sample
from simulation.isaac_lab.g1_stairs.geometry_episode import Result


class RecoveryTests(unittest.TestCase):
    def runtime(self, duration=.04):
        r = object.__new__(GeometryEpisodeRuntime)
        r.contract = SimpleNamespace(training_recovery_s=duration)
        r.strict_failure_latched = torch.zeros(2, dtype=torch.bool)
        r.recovery_elapsed_s = torch.zeros(2)
        return r

    def test_only_bad_landing_recovers_and_expires(self):
        for immediate in (StrictStairGait.RISER_STRIKE, StrictStairGait.EDGE_BRIDGE, StrictStairGait.LOST_SUPPORT):
            r = self.runtime()
            reason = torch.tensor([StrictStairGait.BAD_LANDING, immediate])
            violation = torch.ones(2, dtype=torch.bool)
            self.assertEqual(r._recovery_violation(violation, reason, torch.zeros(2, dtype=torch.bool), .02).tolist(), [False, True])
            self.assertEqual(r._recovery_violation(violation, reason, torch.zeros(2, dtype=torch.bool), .02).tolist(), [False, True])
            self.assertEqual(r._recovery_violation(violation, reason, torch.zeros(2, dtype=torch.bool), .02).tolist(), [True, True])

    def test_new_strike_during_recovery_is_immediate(self):
        r = self.runtime()
        v = torch.ones(2, dtype=torch.bool)
        reason = torch.full((2,), StrictStairGait.BAD_LANDING)
        r._recovery_violation(v, reason, ~v, .02)
        self.assertTrue(r._recovery_violation(v, reason, v, .02).all())

    def test_disabled_is_strict(self):
        r = self.runtime(0)
        v = torch.tensor([True, False])
        self.assertEqual(r._recovery_violation(v, torch.tensor([3, 0]), torch.zeros(2, dtype=torch.bool), .02).tolist(), v.tolist())

    def test_safety_still_terminates_with_contract_masked(self):
        for field, expected in [('forbidden_contact', Result.FORBIDDEN_CONTACT), ('invalid_state', Result.INVALID_STATE), ('velocity_limit', Result.VELOCITY_LIMIT), ('torque_limit', Result.TORQUE_LIMIT), ('timed_out', Result.TIMEOUT)]:
            t = tracker()
            s = sample(x=.9)
            s[field][:] = True
            s['step_over_contract_violation'] = torch.zeros(1, dtype=torch.bool)
            self.assertEqual(t.update(**s).item(), expected)

    def test_partial_reset_clears_only_completed_environment(self):
        from simulation.isaac_lab.g1_stairs.tests.test_geometry_episode import EpisodeAccountingTests
        from simulation.isaac_lab.g1_stairs.geometry_episode_runtime import finish_geometry_episodes
        env = EpisodeAccountingTests().fake_env([Result.STEP_OVER_CONTRACT, Result.RUNNING], [1, 1])
        r = env._g1_geometry_episode
        r.strict_failure_latched[:] = True
        r.recovery_elapsed_s[:] = .02
        finish_geometry_episodes(env, [0])
        self.assertEqual(r.strict_failure_latched.tolist(), [False, True])
        self.assertEqual(r.recovery_elapsed_s[0].item(), 0.)
        self.assertGreater(r.recovery_elapsed_s[1].item(), 0.)

    def test_fall_terminates_during_recovery(self):
        t, s = tracker(), sample(x=.9)
        s['roll_pitch'][:] = 1.0
        s['step_over_contract_violation'] = torch.zeros(1, dtype=torch.bool)
        self.assertEqual(t.update(**s).item(), Result.FALL)


if __name__ == '__main__':
    unittest.main()
