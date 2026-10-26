from __future__ import annotations

import math
import unittest

import numpy as np

from g1_stair_imitation.isaac.runtime_diagnostics import (
    body_angular_velocity_to_world,
    quaternion_angle_error_rad,
    reference_position_world,
    reference_sole_world,
    sole_points_world_from_link_pose,
    xyzw_to_wxyz,
)


class RuntimeDiagnosticsTest(unittest.TestCase):
    def test_reference_frame_translation(self):
        origin = np.asarray((-1.875, 0.0, 0.0))
        local = np.asarray((-0.225, 0.01, 0.80))
        world = reference_position_world(local, origin)
        np.testing.assert_allclose(world, (-1.100, 0.01, 0.80), atol=1.0e-12)

        sole = np.zeros((4, 3), dtype=np.float64)
        sole_world = reference_sole_world(sole, origin)
        np.testing.assert_allclose(
            sole_world,
            np.tile(np.asarray((-0.875, 0.0, 0.0)), (4, 1)),
            atol=1.0e-12,
        )

    def test_quaternion_and_body_angular_velocity(self):
        identity_wxyz = xyzw_to_wxyz((0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(identity_wxyz, (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(
            quaternion_angle_error_rad(identity_wxyz, identity_wxyz),
            0.0,
            places=12,
        )

        # +90 deg yaw: body +X points along world +Y.
        half = math.sqrt(0.5)
        omega_world = body_angular_velocity_to_world(
            (0.0, 0.0, half, half),
            (1.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(omega_world, (0.0, 1.0, 0.0), atol=1.0e-12)

    def test_audited_sole_geometry_identity_pose(self):
        sole = sole_points_world_from_link_pose(
            (1.0, 2.0, 3.0),
            (1.0, 0.0, 0.0, 0.0),
        )
        expected = np.asarray(
            [
                (0.95, 2.025, 2.97),
                (0.95, 1.975, 2.97),
                (1.12, 2.030, 2.97),
                (1.12, 1.970, 2.97),
            ]
        )
        np.testing.assert_allclose(sole, expected, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
