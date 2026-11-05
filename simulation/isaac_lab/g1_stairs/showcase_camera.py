"""Moving camera and start schedule for multi-robot showcase recordings
(visual only, never used in training or evaluation).

Robots walk +x on their own stair courses, scattered over a field. The
camera moves through keyframes: a high wide shot of the field from behind,
a side sweep, a low close-up beside one "hero" robot climbing, a front
view of that robot, a front three-quarter view of the field and a high
front view of the arrivals.

Keyframe = (progress, frame, eye, lookat, wide). frame "group": a point
(dx, ky, oy, z) is x = mean_x + dx, y = centre_y + ky * half_width + oy,
z = ground_z + z (ky -1 = field edge nearest the camera). frame "hero":
x = hero_x + dx, y = hero_y + oy, z = ground_z + z (ky unused). Wide shots
move further out (dx, oy, z times ``spread``) for larger fields.
"""
from __future__ import annotations

import random

LANE_WIDTH_M = 1.2   # staircase width, env_cfg.COURSE_SIZE[1]

KEYFRAMES = (
    (0.00, "group", (-5.5, -1.0, -3.0, 4.2), (1.2, 0.0, 0.0, 0.3), True),    # wide, behind-left
    (0.28, "group", (-1.0, -1.0, -4.0, 2.0), (1.0, 0.0, 0.0, 0.4), True),    # side sweep
    (0.50, "hero", (0.0, 0.0, -2.0, 0.9), (0.5, 0.0, 0.0, 0.55), False),     # close-up beside hero
    (0.62, "hero", (1.8, 0.0, -1.6, 1.0), (0.0, 0.0, 0.0, 0.6), False),      # hero from the front
    (0.82, "group", (3.5, -1.0, -2.5, 1.6), (0.0, 0.0, 0.0, 0.5), True),     # front three-quarter
    (1.00, "group", (5.0, -0.5, -1.0, 2.6), (0.0, 0.0, 0.0, 0.4), True),     # high front
)


def _smooth(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def _point(spec, frame, group, hero, scale=1.0):
    dx, ky, oy, z = spec
    mean_x, centre_y, half_width, ground_z = group
    if frame == "hero":
        return (hero[0] + dx, hero[1] + oy, ground_z + z)
    return (mean_x + scale * dx, centre_y + ky * half_width + scale * oy, ground_z + scale * z)


def camera_pose(progress, group, hero, spread=1.0):
    """(eye, lookat) world tuples for progress in [0, 1].

    group = (mean_x, centre_y, half_width, ground_z); hero = (x, y)."""
    p = min(max(float(progress), 0.0), 1.0)
    for k0, k1 in zip(KEYFRAMES, KEYFRAMES[1:]):
        if p <= k1[0]:
            break
    u = _smooth(min(max((p - k0[0]) / (k1[0] - k0[0]), 0.0), 1.0))
    ends = []
    for _, frame, eye, look, wide in (k0, k1):
        ends.append((_point(eye, frame, group, hero, spread if wide else 1.0),
                     _point(look, frame, group, hero)))
    mix = lambda x, y: tuple(xi + u * (yi - xi) for xi, yi in zip(x, y))
    return mix(ends[0][0], ends[1][0]), mix(ends[0][1], ends[1][1])


def field_geometry(origins_xy):
    """(centre_y, half_width, spread) of the field from the env origins."""
    xs = [p[0] for p in origins_xy]
    ys = [p[1] for p in origins_xy]
    centre_y = 0.5 * (min(ys) + max(ys))
    half_width = 0.5 * (max(ys) - min(ys)) + 0.5 * LANE_WIDTH_M
    half_length = 0.5 * (max(xs) - min(xs))
    spread = max(1.0, half_width / 4.8, 1.0 + half_length / 5.0)
    return centre_y, half_width, spread


def start_delay_steps(num_envs: int, max_delay_s: float, step_dt: float, seed: int):
    """Steps each robot stands before walking: random in [0, max_delay_s],
    the first robot at 0 (deterministic for a seed)."""
    rng = random.Random(seed)
    delays = [rng.uniform(0.0, max_delay_s) for _ in range(num_envs)]
    first = min(delays) if delays else 0.0
    return [int(round((d - first) / step_dt)) for d in delays]
