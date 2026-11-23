"""Diagnostic hand containment using image-time wrist transforms."""

import numpy as np

from .depth_geometry import transform_points


def hand_keep_mask(points, wrist_poses, bounds, *, padding_m):
    """Retain points outside both wrist-local boxes, padded on every face.

    Each pose is reference-from-wrist XYZ plus XYZW. Bounds contain
    xmin, ymin, zmin, xmax, ymax, zmax in wrist coordinates. Finger motion
    is not measured; callers must restrict use to the qualified hand pose.
    Invalid geometry raises rather than silently passing unfiltered points.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite XYZ rows")
    if not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError("hand padding must be finite and nonnegative")
    keep = np.ones(len(points), dtype=bool)
    for side in ("left", "right"):
        box = np.asarray(bounds[side], dtype=float)
        pose = np.asarray(wrist_poses[side], dtype=float)
        if (box.shape != (6,) or not np.isfinite(box).all()
                or np.any(box[:3] >= box[3:])):
            raise ValueError("hand bounds must have positive finite extents")
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("wrist pose must contain seven finite values")
        inverse_q = pose[3:] * [-1., -1., -1., 1.]
        local = transform_points(points - pose[:3], [0., 0., 0.], inverse_q)
        inside = np.all((local >= box[:3] - padding_m)
                        & (local <= box[3:] + padding_m), axis=1)
        keep &= ~inside
    return keep
