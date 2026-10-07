"""Pure geometry for the simulated MID-360 source."""

from dataclasses import dataclass
import math

import numpy as np


MID360_TRANSLATION_XYZ = (
    0.0002835,
    0.00003,
    0.40618,
)
MID360_RPY_RADIANS = (
    0.0,
    0.04014257279586953,
    0.0,
)

INTEGRATION_AZIMUTH_INCREMENT_RADIANS = (
    math.radians(0.5)
)
INTEGRATION_ELEVATIONS_RADIANS = tuple(
    math.radians(value)
    for value in (-20.0, 0.0, 20.0)
)

_NORMALIZATION_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class SensorPose:
    """World pose expressed with a MuJoCo-order quaternion."""

    position_xyz: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


def _finite_vector(name, values, size):
    try:
        vector = np.asarray(
            tuple(values),
            dtype=float,
        )
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be a numeric sequence"
        ) from None

    if vector.shape != (size,):
        raise ValueError(
            f"{name} must contain {size} values"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} values must be finite"
        )
    return vector


def _quaternion_multiply(left, right):
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right

    return np.asarray(
        (
            (
                left_w * right_w
                - left_x * right_x
                - left_y * right_y
                - left_z * right_z
            ),
            (
                left_w * right_x
                + left_x * right_w
                + left_y * right_z
                - left_z * right_y
            ),
            (
                left_w * right_y
                - left_x * right_z
                + left_y * right_w
                + left_z * right_x
            ),
            (
                left_w * right_z
                + left_x * right_y
                - left_y * right_x
                + left_z * right_w
            ),
        ),
        dtype=float,
    )


def _rotate_vector(orientation_wxyz, vector_xyz):
    vector_quaternion = np.asarray(
        (
            0.0,
            vector_xyz[0],
            vector_xyz[1],
            vector_xyz[2],
        ),
        dtype=float,
    )
    conjugate = np.asarray(
        (
            orientation_wxyz[0],
            -orientation_wxyz[1],
            -orientation_wxyz[2],
            -orientation_wxyz[3],
        ),
        dtype=float,
    )
    rotated = _quaternion_multiply(
        _quaternion_multiply(
            orientation_wxyz,
            vector_quaternion,
        ),
        conjugate,
    )
    return rotated[1:]


def _rpy_to_wxyz(roll, pitch, yaw):
    roll_sin = math.sin(roll / 2.0)
    roll_cos = math.cos(roll / 2.0)
    pitch_sin = math.sin(pitch / 2.0)
    pitch_cos = math.cos(pitch / 2.0)
    yaw_sin = math.sin(yaw / 2.0)
    yaw_cos = math.cos(yaw / 2.0)

    return np.asarray(
        (
            (
                roll_cos * pitch_cos * yaw_cos
                + roll_sin * pitch_sin * yaw_sin
            ),
            (
                roll_sin * pitch_cos * yaw_cos
                - roll_cos * pitch_sin * yaw_sin
            ),
            (
                roll_cos * pitch_sin * yaw_cos
                + roll_sin * pitch_cos * yaw_sin
            ),
            (
                roll_cos * pitch_cos * yaw_sin
                - roll_sin * pitch_sin * yaw_cos
            ),
        ),
        dtype=float,
    )


def _numeric_sequence(name, values):
    try:
        sequence = np.asarray(
            tuple(values),
            dtype=float,
        )
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be a numeric sequence"
        ) from None

    if sequence.ndim != 1 or sequence.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )
    return sequence


def _finite_sequence(name, values):
    sequence = _numeric_sequence(
        name,
        values,
    )
    if not np.all(np.isfinite(sequence)):
        raise ValueError(
            f"{name} values must be finite"
        )
    return sequence


def ray_directions(
    *,
    azimuths_radians,
    elevations_radians,
):
    """Return elevation-major normalized sensor-frame rays."""
    azimuths = _finite_sequence(
        "azimuth angles",
        azimuths_radians,
    )
    elevations = _finite_sequence(
        "elevation angles",
        elevations_radians,
    )

    if np.any(
        np.abs(elevations)
        > math.pi / 2.0
    ):
        raise ValueError(
            "elevation angles must be within "
            "[-pi/2, pi/2]"
        )

    rays = []
    for elevation in elevations:
        horizontal = math.cos(elevation)
        vertical = math.sin(elevation)

        for azimuth in azimuths:
            rays.append((
                horizontal * math.cos(azimuth),
                horizontal * math.sin(azimuth),
                vertical,
            ))

    return tuple(rays)


def integration_ray_directions():
    """Return the deterministic navigation integration grid."""
    azimuth_count = round(
        2.0
        * math.pi
        / INTEGRATION_AZIMUTH_INCREMENT_RADIANS
    )
    azimuths = tuple(
        -math.pi
        + index
        * INTEGRATION_AZIMUTH_INCREMENT_RADIANS
        for index in range(azimuth_count)
    )

    return ray_directions(
        azimuths_radians=azimuths,
        elevations_radians=(
            INTEGRATION_ELEVATIONS_RADIANS
        ),
    )


def _validated_directions(directions_sensor):
    try:
        directions = np.asarray(
            tuple(directions_sensor),
            dtype=float,
        )
    except (TypeError, ValueError):
        raise TypeError(
            "ray directions must be numeric"
        ) from None

    if (
        directions.ndim != 2
        or directions.shape[0] == 0
        or directions.shape[1] != 3
    ):
        raise ValueError(
            "ray directions must have shape (N, 3)"
        )
    if not np.all(np.isfinite(directions)):
        raise ValueError(
            "ray directions must be finite"
        )

    norms = np.linalg.norm(
        directions,
        axis=1,
    )
    if not np.allclose(
        norms,
        np.ones(len(norms)),
        rtol=0.0,
        atol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(
            "ray directions must be normalized"
        )
    return directions


def points_from_ray_distances(
    *,
    directions_sensor,
    distances,
    range_min,
    range_max,
):
    """Return sensor-frame points for accepted ray distances."""
    directions = _validated_directions(
        directions_sensor
    )
    distance_values = _numeric_sequence(
        "ray distances",
        distances,
    )

    if len(distance_values) != len(directions):
        raise ValueError(
            "ray directions and distances must align"
        )

    range_min = float(range_min)
    range_max = float(range_max)
    if (
        not math.isfinite(range_min)
        or not math.isfinite(range_max)
        or range_min <= 0.0
        or range_max <= range_min
    ):
        raise ValueError(
            "range limits must be finite, positive, "
            "and increasing"
        )

    points = []
    for direction, distance in zip(
        directions,
        distance_values,
    ):
        if (
            not math.isfinite(distance)
            or distance < range_min
            or distance > range_max
        ):
            continue

        point = direction * distance
        points.append(tuple(
            float(value)
            for value in point
        ))

    return tuple(points)


def sensor_directions_to_world(
    pose,
    directions_sensor,
):
    """Rotate normalized sensor-frame directions into the world frame."""
    _finite_vector(
        "sensor position",
        pose.position_xyz,
        3,
    )
    orientation = _finite_vector(
        "sensor orientation",
        pose.orientation_wxyz,
        4,
    )
    if not math.isclose(
        float(np.linalg.norm(orientation)),
        1.0,
        rel_tol=0.0,
        abs_tol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(
            "sensor orientation must be normalized"
        )

    directions = _validated_directions(
        directions_sensor
    )

    return tuple(
        tuple(
            float(value)
            for value in _rotate_vector(
                orientation,
                direction,
            )
        )
        for direction in directions
    )


def world_points_to_sensor(pose, points_world):
    """Transform finite world points into the sensor frame."""
    position = _finite_vector(
        "sensor position",
        pose.position_xyz,
        3,
    )
    orientation = _finite_vector(
        "sensor orientation",
        pose.orientation_wxyz,
        4,
    )
    if not math.isclose(
        float(np.linalg.norm(orientation)),
        1.0,
        rel_tol=0.0,
        abs_tol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(
            "sensor orientation must be normalized"
        )

    try:
        points = np.asarray(
            tuple(points_world),
            dtype=float,
        )
    except (TypeError, ValueError):
        raise TypeError(
            "world points must be numeric"
        ) from None

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "world points must have shape (N, 3)"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(
            "world points must be finite"
        )

    inverse_orientation = np.asarray(
        (
            orientation[0],
            -orientation[1],
            -orientation[2],
            -orientation[3],
        ),
        dtype=float,
    )

    return tuple(
        tuple(
            float(value)
            for value in _rotate_vector(
                inverse_orientation,
                point - position,
            )
        )
        for point in points
    )


def compose_mid360_pose(
    *,
    torso_position_xyz,
    torso_orientation_wxyz,
):
    """Compose the live torso pose with the official fixed mount."""
    torso_position = _finite_vector(
        "torso position",
        torso_position_xyz,
        3,
    )
    torso_orientation = _finite_vector(
        "torso orientation",
        torso_orientation_wxyz,
        4,
    )

    orientation_norm = float(
        np.linalg.norm(torso_orientation)
    )
    if not math.isclose(
        orientation_norm,
        1.0,
        rel_tol=0.0,
        abs_tol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(
            "torso orientation must be normalized"
        )

    mount_translation = np.asarray(
        MID360_TRANSLATION_XYZ,
        dtype=float,
    )
    mount_orientation = _rpy_to_wxyz(
        *MID360_RPY_RADIANS
    )

    position = (
        torso_position
        + _rotate_vector(
            torso_orientation,
            mount_translation,
        )
    )
    orientation = _quaternion_multiply(
        torso_orientation,
        mount_orientation,
    )

    return SensorPose(
        position_xyz=tuple(
            float(value)
            for value in position
        ),
        orientation_wxyz=tuple(
            float(value)
            for value in orientation
        ),
    )
