"""Offline depth geometry; no ROS or hardware access."""

import numpy as np


def depth_to_points(depth, k, *, depth_scale):
    """Return valid pixels as XYZ metres in camera optical coordinates."""
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    z = np.asarray(depth, dtype=float) * depth_scale
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0)
    k = np.asarray(k, dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("intrinsics must be finite with positive focal lengths")
    v, u = np.indices(z.shape)
    xyz = np.stack((
        (u - k[0, 2]) * z / k[0, 0],
        (v - k[1, 2]) * z / k[1, 1],
        z,
    ), axis=-1)
    valid = (z > 0) & np.isfinite(xyz).all(axis=-1)
    return xyz[valid]


def heights_above_plane(points, plane):
    """Return signed metres; the plane normal must point above the floor."""
    points = np.asarray(points, dtype=float)
    plane = np.asarray(plane, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError("plane must contain four finite coefficients")
    length = np.linalg.norm(plane[:3])
    if not np.isfinite(length) or length <= 0:
        raise ValueError("plane normal must have finite positive length")
    return points @ (plane[:3] / length) + plane[3] / length


def transform_points(points, translation, quaternion_xyzw):
    """Apply a source-to-target rigid transform to XYZ metres."""
    points = np.asarray(points, dtype=float)
    translation = np.asarray(translation, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation must contain three finite values")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite values")
    length = np.linalg.norm(quaternion)
    if not np.isfinite(length) or abs(length - 1.0) > 0.001:
        raise ValueError("quaternion must have unit length")
    quaternion = quaternion / length
    vector, scalar = quaternion[:3], quaternion[3]
    cross = np.cross(vector, points)
    return points + 2 * scalar * cross + 2 * np.cross(vector, cross) + translation
