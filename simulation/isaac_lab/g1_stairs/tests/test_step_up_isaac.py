import unittest

from simulation.isaac_lab.g1_stairs.step_up_isaac import (
    FOOT_BODY_NAMES,
    FORBIDDEN_CONTACT_BODY_NAMES,
    StepUpIsaacConfig,
    _resolve_indices,
)


class TestStepUpIsaacAdapter(unittest.TestCase):

    def test_expected_g1_foot_names(self):
        self.assertEqual(
            FOOT_BODY_NAMES,
            (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
        )

    def test_forbidden_contacts_cover_non_feet(self):
        self.assertIn(
            "pelvis",
            FORBIDDEN_CONTACT_BODY_NAMES,
        )
        self.assertIn(
            "torso_link",
            FORBIDDEN_CONTACT_BODY_NAMES,
        )
        self.assertIn(
            "left_knee_link",
            FORBIDDEN_CONTACT_BODY_NAMES,
        )
        self.assertIn(
            "right_knee_link",
            FORBIDDEN_CONTACT_BODY_NAMES,
        )

        for foot in FOOT_BODY_NAMES:
            self.assertNotIn(
                foot,
                FORBIDDEN_CONTACT_BODY_NAMES,
            )

    def test_resolve_indices(self):
        names = (
            "pelvis",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        )

        indices = _resolve_indices(
            names,
            FOOT_BODY_NAMES,
        )

        self.assertEqual(
            indices,
            {
                "left_ankle_roll_link": 1,
                "right_ankle_roll_link": 2,
            },
        )

    def test_missing_body_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "right_ankle_roll_link",
        ):
            _resolve_indices(
                ("left_ankle_roll_link",),
                FOOT_BODY_NAMES,
            )

    def test_optional_forbidden_contacts_follow_sensor_inventory(self):
        from simulation.isaac_lab.g1_stairs.contact_sensor_contract import (
            resolve_available_body_indices,
        )

        indices = resolve_available_body_indices(
            (
                "pelvis",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
            FORBIDDEN_CONTACT_BODY_NAMES,
        )

        self.assertEqual(indices, {"pelvis": 0})

    def test_single_step_geometry_defaults(self):
        cfg = StepUpIsaacConfig()

        self.assertAlmostEqual(
            cfg.first_riser_x_m,
            1.0,
        )
        self.assertAlmostEqual(
            cfg.tread_depth_m,
            0.25,
        )
        self.assertAlmostEqual(
            cfg.riser_height_m,
            0.05,
        )
        self.assertAlmostEqual(
            cfg.nominal_ankle_height_m,
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
