import unittest

from simulation.isaac_lab.g1_stairs.mechanics_qualification import (
    classify_candidate,
    linear_ramp,
    overall_verdict,
    scripted_candidates,
)


class TestMechanicsQualification(unittest.TestCase):
    def test_candidate_bank_is_deterministic_and_bounded(self):
        candidates = scripted_candidates()
        self.assertEqual(len(candidates), 48)
        self.assertEqual(len({candidate.name for candidate in candidates}), 48)
        for candidate in candidates:
            self.assertIn(candidate.sign, (-1, 1))
            self.assertLessEqual(max(abs(v) for v in candidate.values.values()), 1.84)

    def test_linear_ramp(self):
        self.assertEqual(linear_ramp(2, 3, 4), 0.0)
        self.assertAlmostEqual(linear_ramp(3, 3, 4), 0.25)
        self.assertAlmostEqual(linear_ramp(6, 3, 4), 1.0)
        self.assertAlmostEqual(linear_ramp(100, 3, 4), 1.0)

    def test_candidate_requires_point_three_seconds(self):
        summary = {
            "single_support_flags": [False] * 2 + [True] * 14,
        }
        result = classify_candidate(summary, step_dt=0.02)
        self.assertFalse(result["single_support_pass"])
        summary["single_support_flags"] = [True] * 15
        result = classify_candidate(summary, step_dt=0.02)
        self.assertTrue(result["single_support_pass"])

    def test_neutral_failure_wins(self):
        verdict = overall_verdict(
            {
                "right_anchor_drift_max_m": 0.03,
                "left_anchor_drift_max_m": 0.0,
                "upright_min": 1.0,
            },
            [],
        )
        self.assertEqual(verdict["verdict"], "NEUTRAL_STANCE_FAIL")

    def test_single_support_pass_is_plant_capable(self):
        candidate = {
            "candidate": "good",
            "single_support_pass": True,
            "right_anchor_drift_max_m": 0.01,
            "right_planar_speed_mean_mps": 0.01,
            "critical_tracking_rms_rad": 0.01,
            "torque_utilization_max": 0.5,
            "left_force_share_min": 0.0,
            "left_foot_height_gain_max_m": 0.05,
        }
        verdict = overall_verdict(
            {
                "right_anchor_drift_max_m": 0.001,
                "left_anchor_drift_max_m": 0.001,
                "upright_min": 0.99,
            },
            [candidate],
        )
        self.assertEqual(verdict["verdict"], "PLANT_CAPABLE_SINGLE_SUPPORT")

    def test_unload_with_large_support_motion_is_slip_limited(self):
        candidate = {
            "candidate": "slip",
            "single_support_pass": False,
            "right_anchor_drift_max_m": 0.10,
            "right_planar_speed_mean_mps": 0.18,
            "critical_tracking_rms_rad": 0.02,
            "torque_utilization_max": 0.7,
            "left_force_share_min": 0.10,
            "left_foot_height_gain_max_m": 0.04,
        }
        verdict = overall_verdict(
            {
                "right_anchor_drift_max_m": 0.001,
                "left_anchor_drift_max_m": 0.001,
                "upright_min": 0.99,
            },
            [candidate],
        )
        self.assertEqual(verdict["verdict"], "SUPPORT_SLIP_LIMITING")


if __name__ == "__main__":
    unittest.main()
