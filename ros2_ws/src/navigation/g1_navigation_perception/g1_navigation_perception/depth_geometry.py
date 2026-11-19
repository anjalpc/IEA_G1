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


def fit_floor_candidate(points, up_direction):
    """Fit selected floor candidates; reject degenerate or steep planes."""
    points = np.asarray(points, dtype=float)
    up = np.asarray(up_direction, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("at least three XYZ points are required")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("up_direction must contain three finite values")
    length = np.linalg.norm(up)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("up_direction must be nonzero")
    up = up / length
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
    if singular[1] <= 1e-6:
        raise ValueError("plane points must span two directions")
    normal = vectors[-1]
    if normal @ up < 0:
        normal = -normal
    if normal @ up < np.cos(np.deg2rad(10.0)):
        raise ValueError("candidate plane exceeds diagnostic tilt limit")
    return np.r_[normal, -normal @ center]


def select_floor_plane(points, up_direction, *, distance_bounds):
    """Return a diagnostic plane candidate, or None when unsupported."""
    points = np.asarray(points, dtype=float)
    up = np.asarray(up_direction, dtype=float)
    bounds = np.asarray(distance_bounds, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("up_direction must contain three finite values")
    length = np.linalg.norm(up)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("up_direction must be nonzero")
    up = up / length
    if bounds.shape != (2,) or not np.isfinite(bounds).all():
        raise ValueError("distance_bounds must contain two finite values")
    if not 0 < bounds[0] < bounds[1]:
        raise ValueError("distance_bounds must be positive and increasing")
    if len(points) < 30:
        return None
    stride = max(1, int(np.ceil(len(points) / 2048)))
    sample = points[::stride]
    minimum = max(30, int(np.ceil(0.35 * len(sample))))
    rng = np.random.default_rng(42)
    best, best_support = None, 0
    for _ in range(192):
        try:
            plane = fit_floor_candidate(sample[rng.choice(len(sample), 3,
                                                         replace=False)], up)
            if not bounds[0] <= plane[3] <= bounds[1]:
                continue
            mask = np.abs(heights_above_plane(sample, plane)) <= 0.02
            # Refine before enforcing the final support fraction.
            if np.count_nonzero(mask) < 30:
                continue
            for refinement in range(4):
                plane = fit_floor_candidate(sample[mask], up)
                updated = np.abs(heights_above_plane(sample, plane)) <= 0.02
                if np.count_nonzero(updated) < 30 or np.array_equal(updated, mask):
                    break
                mask = updated
        except ValueError:
            continue
        support = np.count_nonzero(
            np.abs(heights_above_plane(sample, plane)) <= 0.02)
        if support < minimum or not bounds[0] <= plane[3] <= bounds[1]:
            continue
        lower = best is None or plane[3] > best[3] + 0.02
        stronger = (best is not None and abs(plane[3] - best[3]) <= 0.02
                    and support > best_support)
        if lower or stronger:
            best, best_support = plane, support
    if best is None:
        # Retry only after the original full-region selection has failed.
        margin = np.linalg.norm(points, axis=1) * np.sin(np.deg2rad(10.0)) + 0.02
        height = points @ up
        plausible = points[(height >= -bounds[1] - margin)
                           & (height <= -bounds[0] + margin)]
        if 30 <= len(plausible) < len(points):
            return select_floor_plane(plausible, up, distance_bounds=bounds)
    return best
