from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Thesis staircase geometry in IEA world coordinates.
#
# Robot starts at x = 0.
# First riser is x = 1.00 m.
#
# Treat the staircase as one static terrain object whose local origin is the
# centre of the first riser at ground level.
# ---------------------------------------------------------------------------

STAIR_OBJECT_QUAT_W = (1.0, 0.0, 0.0, 0.0)

STAIR_HALF_WIDTH = 0.60

# Coordinates relative to the first riser.
STEP1_X0 = 0.00
STEP1_X1 = 0.25
STEP1_Z = 0.20

STEP2_X0 = 0.25
STEP2_X1 = 0.50
STEP2_Z = 0.40

STEP3_X0 = 0.50
STEP3_X1 = 0.75
STEP3_Z = 0.60

PLATFORM_X0 = 0.75
PLATFORM_X1 = 2.75
PLATFORM_Z = 0.60

HEIGHT_MAP_SIZE = 1.5
HEIGHT_MAP_RESOLUTION = 0.15
HEIGHT_MAP_MAX_DIST = 5.0

NUM_HEIGHT_RAYS = (
    int(HEIGHT_MAP_SIZE / HEIGHT_MAP_RESOLUTION) + 1
) ** 2

if NUM_HEIGHT_RAYS != 121:
    raise RuntimeError(
        f"Expected 121 height-map rays, got {NUM_HEIGHT_RAYS}"
    )


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] *= -1.0
    denom = torch.sum(q * q, dim=-1, keepdim=True)
    return out / denom.clamp_min(1.0e-12)


def quat_mul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(
    q: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    q_xyz = q[..., 1:]
    qw = q[..., :1]

    t = 2.0 * torch.cross(q_xyz, v, dim=-1)

    return (
        v
        + qw * t
        + torch.cross(q_xyz, t, dim=-1)
    )


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-12)

    w, x, y, z = q.unbind(-1)

    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),

            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),

            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rot6d(q: torch.Tensor) -> torch.Tensor:
    mat = quat_to_matrix(q)

    # Exact released convention:
    # first two matrix columns, row-major flatten.
    return mat[..., :2].reshape(
        q.shape[:-1] + (6,)
    )


def heading_quat_wxyz(
    q: torch.Tensor,
) -> torch.Tensor:
    """
    Return yaw-only quaternion from a WXYZ quaternion.
    """

    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-12)

    w, x, y, z = q.unbind(-1)

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = (
        1.0
        - 2.0 * (y * y + z * z)
    )

    yaw = torch.atan2(
        sin_yaw,
        cos_yaw,
    )

    half = 0.5 * yaw

    zeros = torch.zeros_like(half)

    return torch.stack(
        (
            torch.cos(half),
            zeros,
            zeros,
            torch.sin(half),
        ),
        dim=-1,
    )


def static_object_terms(
    robot_root_pos_w: torch.Tensor,
    robot_root_quat_w: torch.Tensor,
    stair_origin_w: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Exact static-terrain object observation semantics.

    Returns:
        object_pos_b:                  (B, 3)
        object_ori_b_6d:               (B, 6)
        object_pos_delta_future:       (B, 30)
        object_ori_delta_future_6d:    (B, 60)
    """

    batch = robot_root_pos_w.shape[0]
    device = robot_root_pos_w.device
    dtype = robot_root_pos_w.dtype

    if stair_origin_w.ndim == 1:
        stair_origin_w = stair_origin_w.reshape(1, 3)

    stair_origin_w = stair_origin_w.to(
        device=device,
        dtype=dtype,
    )

    if stair_origin_w.ndim == 1:
        stair_origin_w = stair_origin_w.reshape(1, 3)

    object_pos_w = stair_origin_w.to(
        device=device,
        dtype=dtype,
    ).expand(batch, -1)

    object_quat_w = torch.tensor(
        STAIR_OBJECT_QUAT_W,
        device=device,
        dtype=dtype,
    ).reshape(1, 4).expand(batch, -1)

    robot_quat_inv = quat_inv(
        robot_root_quat_w
    )

    object_pos_b = quat_apply(
        robot_quat_inv,
        object_pos_w - robot_root_pos_w,
    )

    object_ori_b = quat_mul(
        robot_quat_inv,
        object_quat_w,
    )

    object_ori_b_6d = rot6d(
        object_ori_b
    )

    # Static terrain:
    # reference future pose == simulated current terrain pose.
    object_pos_delta_future = torch.zeros(
        batch,
        10 * 3,
        device=device,
        dtype=dtype,
    )

    identity = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        device=device,
        dtype=dtype,
    )

    object_ori_delta_future_6d = (
        identity
        .reshape(1, 1, 6)
        .expand(batch, 10, 6)
        .reshape(batch, 60)
        .contiguous()
    )

    return (
        object_pos_b,
        object_ori_b_6d,
        object_pos_delta_future,
        object_ori_delta_future_6d,
    )


def _ray_ground_intersection(
    origin: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Intersection parameter t with z = 0 ground.
    """

    dz = direction[..., 2]

    valid = dz < -1.0e-8

    t = torch.where(
        valid,
        -origin[..., 2] / dz,
        torch.full_like(dz, float("inf")),
    )

    valid = (
        valid
        & (t >= 0.0)
        & (t <= HEIGHT_MAP_MAX_DIST)
    )

    return torch.where(
        valid,
        t,
        torch.full_like(t, float("inf")),
    )


def _ray_box_intersection(
    origin: torch.Tensor,
    direction: torch.Tensor,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> torch.Tensor:
    """
    First positive ray/AABB intersection distance.

    Shapes:
        origin:    (..., 3)
        direction: (..., 3)
        return:    (...)
    """

    device = origin.device
    dtype = origin.dtype

    bmin = torch.tensor(
        box_min,
        device=device,
        dtype=dtype,
    )

    bmax = torch.tensor(
        box_max,
        device=device,
        dtype=dtype,
    )

    eps = 1.0e-9

    parallel = torch.abs(direction) < eps

    outside_parallel = (
        parallel
        & (
            (origin < bmin)
            | (origin > bmax)
        )
    )

    impossible = outside_parallel.any(
        dim=-1
    )

    safe_dir = torch.where(
        parallel,
        torch.ones_like(direction),
        direction,
    )

    t1 = (
        bmin - origin
    ) / safe_dir

    t2 = (
        bmax - origin
    ) / safe_dir

    t_near_axis = torch.minimum(
        t1,
        t2,
    )

    t_far_axis = torch.maximum(
        t1,
        t2,
    )

    t_near_axis = torch.where(
        parallel,
        torch.full_like(
            t_near_axis,
            -float("inf"),
        ),
        t_near_axis,
    )

    t_far_axis = torch.where(
        parallel,
        torch.full_like(
            t_far_axis,
            float("inf"),
        ),
        t_far_axis,
    )

    t_near = torch.max(
        t_near_axis,
        dim=-1,
    ).values

    t_far = torch.min(
        t_far_axis,
        dim=-1,
    ).values

    hit = (
        (~impossible)
        & (t_far >= torch.maximum(
            t_near,
            torch.zeros_like(t_near),
        ))
    )

    t = torch.where(
        t_near >= 0.0,
        t_near,
        t_far,
    )

    hit = (
        hit
        & (t >= 0.0)
        & (t <= HEIGHT_MAP_MAX_DIST)
    )

    return torch.where(
        hit,
        t,
        torch.full_like(t, float("inf")),
    )


def _build_local_ray_directions(
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Reproduce the released 11 x 11 ray-direction construction.
    """

    n = (
        int(
            HEIGHT_MAP_SIZE
            / HEIGHT_MAP_RESOLUTION
        )
        + 1
    )

    lin = torch.linspace(
        -0.5 * HEIGHT_MAP_SIZE,
        0.5 * HEIGHT_MAP_SIZE,
        n,
        device=device,
        dtype=dtype,
    )

    grid_x, grid_y = torch.meshgrid(
        lin,
        lin,
        indexing="ij",
    )

    zeros = torch.zeros_like(
        grid_x
    )

    offsets = torch.stack(
        (
            grid_x,
            grid_y,
            zeros,
        ),
        dim=-1,
    ).reshape(-1, 3)

    offsets[:, 2] = -1.0

    return torch.nn.functional.normalize(
        offsets,
        dim=-1,
    )


def staircase_height_map_z(
    robot_root_pos_w: torch.Tensor,
    robot_root_quat_w: torch.Tensor,
    stair_origin_w: torch.Tensor,
) -> torch.Tensor:
    """
    Exact 121-ray geometric adapter for the fixed thesis staircase.

    The released controller:
      1. starts all rays from the robot root,
      2. rotates the local ray fan with robot heading,
      3. intersects terrain,
      4. expresses hit points relative to robot root in heading frame,
      5. returns only Z.

    This implementation also includes vertical riser intersections by treating
    each stair level as a solid AABB.
    """

    if robot_root_pos_w.ndim != 2:
        raise ValueError(
            "robot_root_pos_w must have shape (B,3)"
        )

    if robot_root_quat_w.ndim != 2:
        raise ValueError(
            "robot_root_quat_w must have shape (B,4)"
        )

    batch = robot_root_pos_w.shape[0]
    device = robot_root_pos_w.device
    dtype = robot_root_pos_w.dtype

    if stair_origin_w.ndim == 1:
        stair_origin_w = stair_origin_w.reshape(1, 3)

    stair_origin_w = stair_origin_w.to(
        device=device,
        dtype=dtype,
    )

    if stair_origin_w.shape[0] != 1:
        raise ValueError(
            "qualification adapter currently expects one environment"
        )

    sx = float(stair_origin_w[0, 0].item())
    sy = float(stair_origin_w[0, 1].item())
    sz = float(stair_origin_w[0, 2].item())

    ray_dirs_local = _build_local_ray_directions(
        device,
        dtype,
    )

    ray_dirs_local = (
        ray_dirs_local
        .unsqueeze(0)
        .expand(batch, -1, -1)
    )

    heading_q = heading_quat_wxyz(
        robot_root_quat_w
    )

    heading_q_expanded = (
        heading_q
        .unsqueeze(1)
        .expand(
            -1,
            NUM_HEIGHT_RAYS,
            -1,
        )
    )

    ray_dirs_w = quat_apply(
        heading_q_expanded.reshape(-1, 4),
        ray_dirs_local.reshape(-1, 3),
    ).reshape(
        batch,
        NUM_HEIGHT_RAYS,
        3,
    )

    origins = (
        robot_root_pos_w
        .unsqueeze(1)
        .expand(
            -1,
            NUM_HEIGHT_RAYS,
            -1,
        )
    )

    t_best = _ray_ground_intersection(
        origins,
        ray_dirs_w,
    )

    stair_boxes = (
        (
            (
                sx + STEP1_X0,
                sy - STAIR_HALF_WIDTH,
                sz,
            ),
            (
                sx + STEP1_X1,
                sy + STAIR_HALF_WIDTH,
                sz + STEP1_Z,
            ),
        ),
        (
            (
                sx + STEP2_X0,
                sy - STAIR_HALF_WIDTH,
                sz,
            ),
            (
                sx + STEP2_X1,
                sy + STAIR_HALF_WIDTH,
                sz + STEP2_Z,
            ),
        ),
        (
            (
                sx + STEP3_X0,
                sy - STAIR_HALF_WIDTH,
                sz,
            ),
            (
                sx + STEP3_X1,
                sy + STAIR_HALF_WIDTH,
                sz + STEP3_Z,
            ),
        ),
        (
            (
                sx + PLATFORM_X0,
                sy - STAIR_HALF_WIDTH,
                sz,
            ),
            (
                sx + PLATFORM_X1,
                sy + STAIR_HALF_WIDTH,
                sz + PLATFORM_Z,
            ),
        ),
    )

    for box_min, box_max in stair_boxes:
        t_box = _ray_box_intersection(
            origins,
            ray_dirs_w,
            box_min,
            box_max,
        )

        t_best = torch.minimum(
            t_best,
            t_box,
        )

    finite = torch.isfinite(
        t_best
    )

    safe_t = torch.where(
        finite,
        t_best,
        torch.zeros_like(t_best),
    )

    hit_w = (
        origins
        + safe_t.unsqueeze(-1)
        * ray_dirs_w
    )

    # Released implementation ultimately expresses hits relative to root
    # using the robot heading orientation.
    heading_inv = quat_inv(
        heading_q
    )

    heading_inv_expanded = (
        heading_inv
        .unsqueeze(1)
        .expand(
            -1,
            NUM_HEIGHT_RAYS,
            -1,
        )
    )

    hit_rel_w = (
        hit_w
        - robot_root_pos_w.unsqueeze(1)
    )

    hit_b = quat_apply(
        heading_inv_expanded.reshape(-1, 4),
        hit_rel_w.reshape(-1, 3),
    ).reshape(
        batch,
        NUM_HEIGHT_RAYS,
        3,
    )

    # No-hit rays follow the released nan_to_num behavior.
    hit_b = torch.where(
        finite.unsqueeze(-1),
        hit_b,
        torch.zeros_like(hit_b),
    )

    return hit_b[..., 2]


def build_static_terrain_contract(
    robot_root_pos_w: torch.Tensor,
    robot_root_quat_w: torch.Tensor,
    stair_origin_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    (
        object_pos_b,
        object_ori_b_6d,
        object_pos_delta,
        object_ori_delta,
    ) = static_object_terms(
        robot_root_pos_w,
        robot_root_quat_w,
        stair_origin_w,
    )

    height_map_z = staircase_height_map_z(
        robot_root_pos_w,
        robot_root_quat_w,
        stair_origin_w,
    )

    return {
        "object_pos_b": object_pos_b,
        "object_ori_b_6d": object_ori_b_6d,
        "object_pos_delta_multi_future": object_pos_delta,
        "object_ori_delta_multi_future_6d": object_ori_delta,
        "height_map_z_flat": height_map_z,
    }
