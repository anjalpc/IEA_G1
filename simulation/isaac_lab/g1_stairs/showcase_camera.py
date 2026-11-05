"""Moving camera for multi-robot showcase recordings (visual only).

Robots walk +x in parallel lanes 1.2 m apart (one stair tile each). The
camera follows the robots' mean x and moves through a few keyframes:
a high wide shot of the whole group from behind, a side sweep, a low
close-up of the nearest lane on the stairs, a front three-quarter view
and a high front view of the arrivals on the platform.

Keyframe = (progress, eye, lookat); eye and lookat are
(dx, ky, oy, z): x = mean_x + dx, y = centre_y + ky * half_width + oy,
z = ground_z + z. ky -1 is the lane edge nearest to the camera.
"""
from __future__ import annotations

LANE_WIDTH_M = 1.2   # env_cfg.COURSE_SIZE[1]

KEYFRAMES = (
    (0.00, (-5.5, -1.0, -3.0, 4.2), (1.2, 0.0, 0.0, 0.3)),   # wide, from behind-left
    (0.30, (-1.0, -1.0, -4.0, 2.0), (1.0, 0.0, 0.0, 0.4)),   # side sweep
    (0.55, (0.3, -1.0, -1.6, 0.9), (0.4, -1.0, 0.6, 0.55)),  # low close-up, nearest lane
    (0.80, (3.5, -1.0, -2.5, 1.6), (0.0, 0.0, 0.0, 0.5)),    # front three-quarter
    (1.00, (5.0, -0.5, -1.0, 2.6), (0.0, 0.0, 0.0, 0.4)),    # high front
)


def _smooth(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def _point(spec, mean_x, centre_y, half_width, ground_z):
    dx, ky, oy, z = spec
    return (mean_x + dx, centre_y + ky * half_width + oy, ground_z + z)


def camera_pose(progress: float, mean_x: float, centre_y: float,
                half_width: float, ground_z: float = 0.0):
    """(eye, lookat) world tuples for progress in [0, 1]."""
    p = min(max(float(progress), 0.0), 1.0)
    for (t0, e0, l0), (t1, e1, l1) in zip(KEYFRAMES, KEYFRAMES[1:]):
        if p <= t1:
            u = _smooth((p - t0) / (t1 - t0))
            break
    a = (_point(e0, mean_x, centre_y, half_width, ground_z), _point(l0, mean_x, centre_y, half_width, ground_z))
    b = (_point(e1, mean_x, centre_y, half_width, ground_z), _point(l1, mean_x, centre_y, half_width, ground_z))
    mix = lambda x, y: tuple(xi + u * (yi - xi) for xi, yi in zip(x, y))
    return mix(a[0], b[0]), mix(a[1], b[1])


def lane_geometry(origins_y):
    """(centre_y, half_width) of the lanes from the env origins' y values."""
    lo, hi = min(origins_y), max(origins_y)
    return 0.5 * (lo + hi), 0.5 * (hi - lo) + 0.5 * LANE_WIDTH_M


def stagger_hold_steps(num_envs: int, stagger_s: float, step_dt: float, seed: int):
    """Steps each robot stands before walking: a shuffled wave, stagger_s
    apart (deterministic for a seed). Visual only."""
    import random
    order = list(range(num_envs))
    random.Random(seed).shuffle(order)
    hold = [0] * num_envs
    for rank, env_id in enumerate(order):
        hold[env_id] = int(round(rank * stagger_s / step_dt))
    return hold
