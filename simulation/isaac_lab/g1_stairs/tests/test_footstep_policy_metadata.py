import unittest
from simulation.isaac_lab.g1_stairs.policy_initialization import known_observation_layout

class PolicyMetadataTests(unittest.TestCase):
    def test_688(self):
        d = known_observation_layout(688)
        self.assertEqual((d["actor_height_scan_observations"],
                          d["explicit_geometry_observations"],
                          d["planned_footstep_observations"],
                          d["gait_clock_observations"],
                          d["unclassified_additional_observations"]), (187,5,14,2,0))
    def test_672(self):
        d=known_observation_layout(672)
        self.assertEqual(d["actor_height_scan_observations"],187)
        self.assertEqual(d["unclassified_additional_observations"],0)
    def test_unknown_is_unclassified(self):
        d=known_observation_layout(492)
        self.assertEqual(d["actor_height_scan_observations"],0)
        self.assertEqual(d["unclassified_additional_observations"],12)
    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            known_observation_layout(479)
