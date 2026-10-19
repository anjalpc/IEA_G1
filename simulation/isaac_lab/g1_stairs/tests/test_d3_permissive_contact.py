import unittest
import torch

from simulation.isaac_lab.g1_stairs.geometry_episode import EpisodeContract, Result
from simulation.isaac_lab.g1_stairs.tests.test_geometry_episode import tracker, sample


class D3Tests(unittest.TestCase):
    def test_config_validation(self):
        self.assertTrue(EpisodeContract(
            strict_stair_gait=True,
            training_permissive_contact=True,
        ).training_permissive_contact)
        with self.assertRaises(ValueError):
            EpisodeContract(training_permissive_contact=True)
        with self.assertRaises(ValueError):
            EpisodeContract(
                strict_stair_gait=True,
                training_permissive_contact=True,
                training_recovery_s=0.3,
            )

    def test_contact_mask_allows_continuation(self):
        t, s = tracker(), sample(x=0.9)
        s["step_over_contract_violation"] = torch.zeros(1, dtype=torch.bool)
        self.assertEqual(t.update(**s).item(), Result.RUNNING)

    def test_strict_contact_still_terminates(self):
        t, s = tracker(), sample(x=0.9)
        s["step_over_contract_violation"] = torch.ones(1, dtype=torch.bool)
        self.assertEqual(t.update(**s).item(), Result.STEP_OVER_CONTRACT)

    def test_safety_remains_terminal(self):
        for field, reason in (
            ("invalid_state", Result.INVALID_STATE),
            ("forbidden_contact", Result.FORBIDDEN_CONTACT),
            ("velocity_limit", Result.VELOCITY_LIMIT),
            ("torque_limit", Result.TORQUE_LIMIT),
            ("timed_out", Result.TIMEOUT),
        ):
            with self.subTest(field=field):
                t, s = tracker(), sample(x=0.9)
                s["step_over_contract_violation"] = torch.zeros(1, dtype=torch.bool)
                s[field][:] = True
                self.assertEqual(t.update(**s).item(), reason)


if __name__ == "__main__":
    unittest.main()
