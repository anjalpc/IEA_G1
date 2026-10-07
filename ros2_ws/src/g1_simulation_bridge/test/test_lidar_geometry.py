"""Pure geometry contract for the simulated MID-360 mount."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest


SIMULATION_ROOT = (
    Path(__file__).parents[4]
    / "simulation"
    / "mujoco"
)
sys.path.insert(0, str(SIMULATION_ROOT))

from navigation.lidar_geometry import (  # noqa: E402
    MID360_RPY_RADIANS,
    MID360_TRANSLATION_XYZ,
    compose_mid360_pose,
)


def test_preserves_official_unitree_mount_constants():
    assert MID360_TRANSLATION_XYZ == (
        0.0002835,
        0.00003,
        0.40618,
    )
    assert MID360_RPY_RADIANS == (
        0.0,
        0.04014257279586953,
        0.0,
    )


def test_composes_mount_from_identity_torso_pose():
    pose = compose_mid360_pose(
        torso_position_xyz=(0.0, 0.0, 0.0),
        torso_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(
        pose.position_xyz,
        MID360_TRANSLATION_XYZ,
        rtol=0.0,
        atol=1.0e-12,
    )

    half_pitch = MID360_RPY_RADIANS[1] / 2.0
    np.testing.assert_allclose(
        pose.orientation_wxyz,
        (
            math.cos(half_pitch),
            0.0,
            math.sin(half_pitch),
            0.0,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_rotates_mount_translation_with_live_torso_pose():
    half_yaw = math.pi / 4.0
    torso_position = (1.0, 2.0, 3.0)
    pose = compose_mid360_pose(
        torso_position_xyz=torso_position,
        torso_orientation_wxyz=(
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        ),
    )

    mount_x, mount_y, mount_z = (
        MID360_TRANSLATION_XYZ
    )
    np.testing.assert_allclose(
        pose.position_xyz,
        (
            torso_position[0] - mount_y,
            torso_position[1] + mount_x,
            torso_position[2] + mount_z,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )

    half_pitch = MID360_RPY_RADIANS[1] / 2.0
    yaw_component = math.cos(half_yaw)
    pitch_cos = math.cos(half_pitch)
    pitch_sin = math.sin(half_pitch)
    np.testing.assert_allclose(
        pose.orientation_wxyz,
        (
            yaw_component * pitch_cos,
            -yaw_component * pitch_sin,
            yaw_component * pitch_sin,
            yaw_component * pitch_cos,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("position", "orientation", "message"),
    [
        (
            (0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            "position",
        ),
        (
            (0.0, math.nan, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            "finite",
        ),
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            "orientation",
        ),
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            "normalized",
        ),
        (
            (0.0, 0.0, 0.0),
            (math.inf, 0.0, 0.0, 0.0),
            "finite",
        ),
    ],
)
def test_rejects_invalid_torso_pose(
    position,
    orientation,
    message,
):
    with pytest.raises(
        (TypeError, ValueError),
        match=message,
    ):
        compose_mid360_pose(
            torso_position_xyz=position,
            torso_orientation_wxyz=orientation,
        )
