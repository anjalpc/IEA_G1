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
    INTEGRATION_AZIMUTH_INCREMENT_RADIANS,
    INTEGRATION_ELEVATIONS_RADIANS,
    MID360_RPY_RADIANS,
    MID360_TRANSLATION_XYZ,
    SensorPose,
    compose_mid360_pose,
    integration_ray_directions,
    points_from_ray_distances,
    ray_directions,
    sensor_directions_to_world,
    world_points_to_sensor,
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

def test_generates_elevation_major_normalized_rays():
    rays = ray_directions(
        azimuths_radians=(
            0.0,
            math.pi / 2.0,
        ),
        elevations_radians=(
            0.0,
            math.pi / 2.0,
        ),
    )

    np.testing.assert_allclose(
        rays,
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.linalg.norm(rays, axis=1),
        np.ones(4),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_integration_grid_is_complete_and_deterministic():
    assert math.isclose(
        INTEGRATION_AZIMUTH_INCREMENT_RADIANS,
        math.radians(0.5),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    np.testing.assert_allclose(
        INTEGRATION_ELEVATIONS_RADIANS,
        tuple(
            math.radians(value)
            for value in (-20.0, 0.0, 20.0)
        ),
        rtol=0.0,
        atol=1.0e-15,
    )

    first = integration_ray_directions()
    second = integration_ray_directions()

    assert first == second
    assert len(first) == 2160

    np.testing.assert_allclose(
        first[0],
        (
            -math.cos(math.radians(20.0)),
            0.0,
            -math.sin(math.radians(20.0)),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("azimuths", "elevations", "message"),
    [
        ((), (0.0,), "azimuth"),
        ((0.0,), (), "elevation"),
        ((math.nan,), (0.0,), "finite"),
        ((0.0,), (math.inf,), "finite"),
        ((0.0,), (math.pi,), "elevation"),
    ],
)
def test_rejects_invalid_ray_angles(
    azimuths,
    elevations,
    message,
):
    with pytest.raises(
        (TypeError, ValueError),
        match=message,
    ):
        ray_directions(
            azimuths_radians=azimuths,
            elevations_radians=elevations,
        )


def test_filters_misses_and_out_of_range_distances():
    points = points_from_ray_distances(
        directions_sensor=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
        distances=(
            -1.0,
            0.24,
            1.5,
            math.inf,
            40.1,
        ),
        range_min=0.25,
        range_max=40.0,
    )

    assert points == (
        (0.0, 0.0, 1.5),
    )


def test_converts_world_hits_back_to_sensor_coordinates():
    half_yaw = math.pi / 4.0
    pose = SensorPose(
        position_xyz=(1.0, 2.0, 3.0),
        orientation_wxyz=(
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        ),
    )

    points = world_points_to_sensor(
        pose,
        (
            (1.0, 3.0, 3.0),
            (0.0, 2.0, 3.0),
        ),
    )

    np.testing.assert_allclose(
        points,
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("directions", "distances", "range_min", "range_max"),
    [
        (((1.0, 0.0, 0.0),), (), 0.25, 40.0),
        (((2.0, 0.0, 0.0),), (1.0,), 0.25, 40.0),
        (((1.0, 0.0, 0.0),), (1.0,), 0.0, 40.0),
        (((1.0, 0.0, 0.0),), (1.0,), 1.0, 1.0),
    ],
)
def test_rejects_invalid_range_projection(
    directions,
    distances,
    range_min,
    range_max,
):
    with pytest.raises(
        (TypeError, ValueError),
    ):
        points_from_ray_distances(
            directions_sensor=directions,
            distances=distances,
            range_min=range_min,
            range_max=range_max,
        )


def test_rotates_sensor_ray_directions_into_world_frame():
    half_yaw = math.pi / 4.0
    pose = SensorPose(
        position_xyz=(10.0, 20.0, 30.0),
        orientation_wxyz=(
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        ),
    )

    directions_world = sensor_directions_to_world(
        pose,
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )

    np.testing.assert_allclose(
        directions_world,
        (
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
