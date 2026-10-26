"""Pure math helpers for the short Isaac imitation runtime probe.

This module intentionally has no Isaac imports so its coordinate transforms and
metric helpers can be unit-tested on the host before launching the simulator.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


FIRST_RISER_COURSE_X_M = 1.0
SOLE_RADIUS_M = 0.005
SOLE_POINT_NAMES = (
    "rear_left",
    "rear_right",
    "front_left",
    "front_right",
)
SOLE_POINTS_LOCAL_M = np.asarray(
    [
        (-0.05, +0.025, -0.03),
        (-0.05, -0.025, -0.03),
        (+0.12, +0.030, -0.03),
        (+0.12, -0.030, -0.03),
    ],
    dtype=np.float64,
)

WEAK_SAGITTAL_JOINTS = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)

LATERAL_DIAGNOSTIC_JOINTS = (
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_ankle_roll_joint",
)


def xyzw_to_wxyz(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    """Convert a quaternion from canonical-reference XYZW to Isaac WXYZ."""

    q = np.asarray(tuple(quaternion_xyzw), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    q = q / norm
    return np.asarray((q[3], q[0], q[1], q[2]), dtype=np.float64)


def quat_wxyz_to_matrix(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix for a normalized WXYZ quaternion."""

    q = np.asarray(tuple(quaternion_wxyz), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def reference_position_world(
    position_stair_frame_m: Iterable[float],
    env_origin_world_m: Iterable[float],
) -> np.ndarray:
    """Translate the canonical first-riser-local stair frame into Isaac world."""

    position = np.asarray(tuple(position_stair_frame_m), dtype=np.float64)
    origin = np.asarray(tuple(env_origin_world_m), dtype=np.float64)
    if position.shape != (3,) or origin.shape != (3,):
        raise ValueError("position and environment origin must each have shape (3,)")
    offset = np.asarray((FIRST_RISER_COURSE_X_M, 0.0, 0.0), dtype=np.float64)
    return origin + offset + position


def reference_sole_world(
    sole_points_stair_frame_m: np.ndarray,
    env_origin_world_m: Iterable[float],
) -> np.ndarray:
    """Translate canonical four-sphere sole centres into Isaac world."""

    sole = np.asarray(sole_points_stair_frame_m, dtype=np.float64)
    if sole.shape != (4, 3):
        raise ValueError(f"expected sole shape (4, 3), got {sole.shape}")
    origin = np.asarray(tuple(env_origin_world_m), dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError("environment origin must have shape (3,)")
    offset = np.asarray((FIRST_RISER_COURSE_X_M, 0.0, 0.0), dtype=np.float64)
    return sole + origin[None, :] + offset[None, :]


def body_angular_velocity_to_world(
    root_quat_xyzw: Iterable[float],
    angular_velocity_body_radps: Iterable[float],
) -> np.ndarray:
    """Rotate canonical body-frame root angular velocity into Isaac world."""

    q_wxyz = xyzw_to_wxyz(root_quat_xyzw)
    omega_body = np.asarray(tuple(angular_velocity_body_radps), dtype=np.float64)
    if omega_body.shape != (3,):
        raise ValueError(f"expected angular velocity shape (3,), got {omega_body.shape}")
    return quat_wxyz_to_matrix(q_wxyz) @ omega_body


def quaternion_angle_error_rad(
    actual_wxyz: Iterable[float],
    expected_wxyz: Iterable[float],
) -> float:
    """Shortest orientation error between two WXYZ quaternions."""

    actual = np.asarray(tuple(actual_wxyz), dtype=np.float64)
    expected = np.asarray(tuple(expected_wxyz), dtype=np.float64)
    if actual.shape != (4,) or expected.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    actual /= np.linalg.norm(actual)
    expected /= np.linalg.norm(expected)
    dot = float(np.clip(abs(np.dot(actual, expected)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def sole_points_world_from_link_pose(
    link_position_world_m: Iterable[float],
    link_quaternion_wxyz: Iterable[float],
) -> np.ndarray:
    """Transform the audited four local sole-sphere centres into world space."""

    position = np.asarray(tuple(link_position_world_m), dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"expected link position shape (3,), got {position.shape}")
    rotation = quat_wxyz_to_matrix(link_quaternion_wxyz)
    return position[None, :] + (rotation @ SOLE_POINTS_LOCAL_M.T).T


def named_joint_indices(
    joint_names: Iterable[str],
    selected_names: Iterable[str],
) -> tuple[int, ...]:
    """Resolve an exact ordered subset of joint names."""

    index = {name: i for i, name in enumerate(tuple(joint_names))}
    selected = tuple(selected_names)
    missing = [name for name in selected if name not in index]
    if missing:
        raise ValueError(f"missing joints: {missing}")
    return tuple(index[name] for name in selected)


def rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    """Finite scalar root-mean-square error."""

    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {e.shape}")
    if not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError("RMSE inputs must be finite")
    return float(np.sqrt(np.mean((a - e) ** 2)))
