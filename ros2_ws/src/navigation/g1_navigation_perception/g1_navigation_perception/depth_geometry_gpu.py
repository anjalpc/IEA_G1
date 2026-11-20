"""Optional CUDA floor fitting; imported only when explicitly requested."""

import numpy as np
import cupy as cp


def select_floor_plane_gpu(points, up_direction, *, distance_bounds):
    """Return an experimental diagnostic floor plane using batched CUDA fitting."""
    points = cp.asarray(points, dtype=cp.float64)
    up = cp.asarray(up_direction, dtype=cp.float64)
    bounds = np.asarray(distance_bounds, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not bool(cp.isfinite(points).all().item()):
        raise ValueError("points must be finite")
    if up.shape != (3,) or not bool(cp.isfinite(up).all().item()):
        raise ValueError("up_direction must contain three finite values")
    length = float(cp.linalg.norm(up).item())
    if not np.isfinite(length) or length <= 0:
        raise ValueError("up_direction must be nonzero")
    up = up / length
    if bounds.shape != (2,) or not np.isfinite(bounds).all():
        raise ValueError("distance_bounds must contain two finite values")
    if not 0 < bounds[0] < bounds[1]:
        raise ValueError("distance_bounds must be positive and increasing")
    cos_limit = np.cos(np.deg2rad(10.0))
    for attempt in range(2):
        if len(points) < 30:
            return None
        sample = points[::max(1, int(np.ceil(len(points) / 2048)))]
        sample_count = len(sample)
        minimum = max(30, int(np.ceil(.35 * sample_count)))
        # Warm-up and live fitting share one shape. Padding is never an inlier.
        sample = cp.pad(sample, ((0, 2048 - sample_count), (0, 0)))
        real_columns = cp.arange(2048) < sample_count
        rng = np.random.default_rng(42)
        indices = cp.asarray(np.array([
            rng.choice(sample_count, 3, replace=False) for _ in range(192)]))
        triples = sample[indices]
        normals = cp.cross(triples[:, 1] - triples[:, 0],
                           triples[:, 2] - triples[:, 0])
        lengths = cp.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals /= cp.maximum(lengths[:, None], 1e-30)
        normals *= cp.where(normals @ up < 0, -1., 1.)[:, None]
        offsets = -cp.sum(normals * triples.mean(axis=1), axis=1)
        valid &= (normals @ up >= cos_limit)
        valid &= (offsets >= bounds[0]) & (offsets <= bounds[1])
        mask = ((cp.abs(normals @ sample.T + offsets[:, None]) <= .02)
                & real_columns[None, :])
        valid &= mask.sum(axis=1) >= 30
        active = valid.copy()

        for _ in range(4):
            weights = mask.astype(cp.float64)
            counts = weights.sum(axis=1)
            centers = weights @ sample / cp.maximum(counts[:, None], 1.)
            second = cp.einsum("hn,ni,nj->hij", weights, sample, sample)
            covariance = second - counts[:, None, None] * (
                centers[:, :, None] * centers[:, None, :])
            eigenvalues, eigenvectors = cp.linalg.eigh(covariance)
            fitted = eigenvectors[:, :, 0]
            fitted *= cp.where(fitted @ up < 0, -1., 1.)[:, None]
            fitted_offsets = -cp.sum(fitted * centers, axis=1)
            fit_ok = (eigenvalues[:, 1] > 1e-12) & (fitted @ up >= cos_limit)
            valid &= (~active) | fit_ok
            normals = cp.where(active[:, None], fitted, normals)
            offsets = cp.where(active, fitted_offsets, offsets)
            updated = ((cp.abs(normals @ sample.T + offsets[:, None]) <= .02)
                       & real_columns[None, :])
            changing = cp.any(updated != mask, axis=1)
            enough = updated.sum(axis=1) >= 30
            continue_refining = active & fit_ok & changing & enough
            mask = cp.where(continue_refining[:, None], updated, mask)
            active = continue_refining

        support = ((cp.abs(normals @ sample.T + offsets[:, None]) <= .02)
                   & real_columns[None, :]).sum(axis=1)
        valid &= (support >= minimum) & (offsets >= bounds[0]) & (offsets <= bounds[1])
        candidates = cp.asnumpy(cp.column_stack((
            normals, offsets, support, valid.astype(cp.float64))))
        best, best_support = None, 0
        for row in candidates:
            if not row[5]:
                continue
            plane, count = row[:4], row[4]
            lower = best is None or plane[3] > best[3] + .02
            stronger = (best is not None and abs(plane[3]-best[3]) <= .02
                        and count > best_support)
            if lower or stronger:
                best, best_support = plane.copy(), count
        if best is not None:
            return best
        if attempt == 0:
            margin = cp.linalg.norm(points, axis=1) * np.sin(np.deg2rad(10.0)) + .02
            height = points @ up
            plausible = points[(height >= -bounds[1]-margin) & (height <= -bounds[0]+margin)]
            if len(plausible) == len(points):
                return None
            points = plausible
    return None
