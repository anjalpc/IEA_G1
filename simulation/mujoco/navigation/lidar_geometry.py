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
