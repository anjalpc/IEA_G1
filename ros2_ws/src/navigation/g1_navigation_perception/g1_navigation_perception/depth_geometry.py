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
