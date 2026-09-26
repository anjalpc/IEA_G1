import math
import unittest

from simulation.isaac_lab.g1_stairs.evaluation.isaac_lab_adapter import (
    IsaacLabStepState,
    quaternion_wxyz_to_roll_pitch,
    resolve_body_indices,
    step_index_from_course_x,
    to_step_sample,
)


class IsaacLabAdapterTest(unittest.TestCase):
    def state(self, *, x=0.0, z=0.8, quaternion=(1.0, 0.0, 0.0, 0.0)):
        return IsaacLabStepState(
            time_s=0.5,
            root_position_world_m=(x, 0.0, z),
            environment_origin_world_m=(0.0, 0.0, 0.0),
            root_quaternion_wxyz=quaternion,
        )

    def test_robot_start_maps_to_course_x_one(self):
        sample = to_step_sample(self.state())
        self.assertAlmostEqual(sample.base_x_m, 1.0)
        self.assertEqual(sample.step_index, 0)

    def test_stair_edges_map_to_three_levels(self):
        self.assertEqual(step_index_from_course_x(1.99), 0)
        self.assertEqual(step_index_from_course_x(2.00), 1)
        self.assertEqual(step_index_from_course_x(2.25), 2)
        self.assertEqual(step_index_from_course_x(2.50), 3)

    def test_environment_origin_is_removed(self):
        state = IsaacLabStepState(
            time_s=0.0,
            root_position_world_m=(-3.75, -15.0, 0.8),
            environment_origin_world_m=(-4.75, -15.0, 0.0),
            root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        sample = to_step_sample(state)
        self.assertAlmostEqual(sample.base_x_m, 2.0)
        self.assertEqual(sample.step_index, 1)

    def test_roll_and_pitch_use_wxyz_contract(self):
        half = math.sqrt(0.5)
        roll, pitch = quaternion_wxyz_to_roll_pitch((half, half, 0.0, 0.0))
        self.assertAlmostEqual(roll, math.pi / 2.0)
        self.assertAlmostEqual(pitch, 0.0)

    def test_invalid_quaternion_fails_closed_in_evaluator(self):
        sample = to_step_sample(self.state(quaternion=(0.0, 0.0, 0.0, 0.0)))
        self.assertTrue(math.isnan(sample.roll_rad))
        self.assertTrue(math.isnan(sample.pitch_rad))

    def test_flags_and_contacts_are_preserved(self):
        state = IsaacLabStepState(
            time_s=1.0,
            root_position_world_m=(0.0, 0.0, 0.8),
            environment_origin_world_m=(0.0, 0.0, 0.0),
            root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            contacts=("left_knee",),
            joint_violation=True,
            velocity_violation=True,
            torque_violation=True,
            slip_violation=True,
            timed_out=True,
        )
        sample = to_step_sample(state)
        self.assertEqual(sample.contacts, frozenset({"left_knee"}))
        self.assertTrue(sample.joint_violation)
        self.assertTrue(sample.velocity_violation)
        self.assertTrue(sample.torque_violation)
        self.assertTrue(sample.slip_violation)
        self.assertTrue(sample.timed_out)

    def test_body_indices_are_resolved_by_name(self):
        indices = resolve_body_indices(
            ("torso_link", "pelvis", "left_ankle_roll_link"),
            ("pelvis", "left_ankle_roll_link"),
        )
        self.assertEqual(indices, {"pelvis": 1, "left_ankle_roll_link": 2})

    def test_optional_contact_bodies_may_be_absent_from_sensor(self):
        from simulation.isaac_lab.g1_stairs.contact_sensor_contract import (
            resolve_available_body_indices,
        )

        indices = resolve_available_body_indices(
            (
                "pelvis",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
            (
                "pelvis",
                "left_rubber_hand",
                "right_rubber_hand",
            ),
        )
        self.assertEqual(indices, {"pelvis": 0})

    def test_missing_required_body_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "right_ankle_roll_link"):
            resolve_body_indices(
                ("pelvis", "left_ankle_roll_link"),
                ("left_ankle_roll_link", "right_ankle_roll_link"),
            )


if __name__ == "__main__":
    unittest.main()
