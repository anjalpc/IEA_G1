"""Geometric full-sole tread qualifier for the opt-in footstep diagnostic.

This is a geometric/contact-force approximation, NOT a contact-patch proof.
No dependency on legacy acquisition state machines.
"""
import torch

from .footstep_tensor import APPROACH_LAST_GAP_M, tread_target_x

# Training acceptance (full_sole=False). A centre-only rule paid landings
# with the heel 3-8 cm off the tread front, followed by forward replants and
# toe-riser contact (run 20261030_133621: 15 of 17 tread hits).
# 0.03 -> 0.01 (patch 38): with 3 cm allowed, placement drifted back to the
# limit and past it (run 20261101_202412 model_5000: heel median 1.5 cm
# behind the edge, 6/39 landings > 3 cm, 3/10 rollouts without a missed
# stair step vs 10/10 at model_4000/4700). Taller risers need the heel on.
TREAD_HEEL_OVERHANG_M = 0.01     # heel may overhang the tread front edge
SPOT_TOE_CLEARANCE_M = 0.02      # spot: toe this far before the first riser
# Patch 53: a tread hit pays by placement. A binary hit let the policy land at
# the acceptance limit (run 20261106_151951 model_14200, 4 cm: tread 1-2 heel
# median -0.1 cm vs the 1 cm overhang limit, misses at exactly -1.0 cm).
TREAD_QUALITY_FLOOR = 0.5        # pay at the acceptance limit, 1.0 at the planned centre
# The simulated foot touches a riser face when its sphere toe (sole_bounds
# x_max) is ~1.5 cm short of it: run 20261109_112507, all 435 riser-contact
# frames (F_y 0, F_x backwards) with the toe 1.4-2.0 cm behind the riser line;
# loaded feet stop 0.9-1.4 cm short. The URDF foot is four 5 mm spheres, so
# the extra reach comes from the simulated asset (collision or contact offset).
TOE_CONTACT_REACH_M = 0.015


def tread_placement_quality(*, sole_bounds, targets, feet, first_riser_x,
                            tread_depth, riser_height, step_count):
    """[N] acquisition pay factor of the assigned foot on a stair tread.

    1.0 with the sole centre on the planned target, falling linearly with
    the x error to TREAD_QUALITY_FLOOR at either physical limit: backwards
    where the heel reaches the TREAD_HEEL_OVERHANG_M acceptance limit,
    forwards where the toe reaches the riser ahead (TOE_CONTACT_REACH_M short
    of the tread end), and below. Rows whose current target is not a tread
    (flat tiles, approach, platform) get 1.0.
    """
    n = sole_bounds.shape[0]
    target_x = targets[:, 0, 0]
    index = torch.round((target_x - first_riser_x) / tread_depth - 0.5).long()
    tread = ((target_x >= first_riser_x) & (index >= 0) & (index < step_count)
             & (riser_height > 0))
    b = sole_bounds[torch.arange(n, device=sole_bounds.device), feet[:, 0].long()]
    centre_x = 0.5 * (b[:, 0] + b[:, 1])
    half = 0.5 * (b[:, 1] - b[:, 0])
    start = first_riser_x + index.to(target_x.dtype) * tread_depth
    back = target_x - (start - TREAD_HEEL_OVERHANG_M + half)
    ahead = (start + tread_depth - TOE_CONTACT_REACH_M - half) - target_x
    offset = centre_x - target_x
    limit = torch.where(offset < 0, back, ahead)
    error = (offset.abs() / limit.clamp_min(1e-3)).clamp(0.0, 1.0)
    quality = 1.0 - (1.0 - TREAD_QUALITY_FLOOR) * error
    return torch.where(tread, quality, torch.ones_like(quality))


def qualified_tread_mask(*, sole_bounds, vertical_force, targets, feet, valid,
                         first_riser_x, tread_depth, riser_height, step_count,
                         half_width, margin_m=0.005, height_tolerance_m=0.01,
                         support_force_n=30.0, approach_steps=0, full_sole=True):
    """Return [N,2] valid support on CURRENT target's level/tread.

    full_sole=True: the whole sole lies inside the cell (strict, evaluation).
    full_sole=False (training advancement): treads: heel at most
    TREAD_HEEL_OVERHANG_M over the front edge and, below a further riser, toe
    not past it; spot: centre on the cell and toe SPOT_TOE_CLEARANCE_M before
    the first riser; platform targets: centre at or past their cell start
    (an overshoot on the platform still completes the plan; patch 59 for the
    first platform target).
    Both require the sole flat on that level, loaded, and inside the course.

    Target XYZ denotes a planned sole centre. The active stair level comes from
    the current target's height; platform targets are checked against the
    platform surface. Only the assigned foot may pass. Approach targets are
    level-0 ground cells of one tread depth centred on the target, checked
    exactly like treads.
    """
    if approach_steps < 0:
        raise ValueError('approach_steps must be non-negative')
    n = sole_bounds.shape[0]
    if sole_bounds.shape != (n, 2, 6) or vertical_force.shape != (n, 2):
        raise ValueError('invalid sole/contact shapes')
    if targets.shape != (n, 2, 4) or feet.shape != (n, 2) or valid.shape != (n, 2):
        raise ValueError('invalid target shapes')
    for v in (first_riser_x, tread_depth, riser_height):
        if v.shape != (n,):
            raise ValueError('course parameters must be per-environment')
    if step_count < 1 or half_width <= 0 or margin_m < 0 or height_tolerance_m <= 0 or support_force_n <= 0:
        raise ValueError('invalid geometry or support parameters')
    if bool((tread_depth <= 2 * margin_m).any()) or bool((riser_height < 0).any()):
        raise ValueError('invalid stair dimensions')
    target_z = targets[:, 0, 2]
    target_x = targets[:, 0, 0]
    # Use target sequence's actual X to distinguish the two platform goals.
    # Both have the same upper-platform height but different tread intervals.
    # Approach targets sit on their own grid (footstep_tensor.approach_target_x).
    on_stairs = target_x >= first_riser_x
    index_float = torch.where(
        on_stairs, (target_x - first_riser_x) / tread_depth - 0.5,
        (target_x - first_riser_x + APPROACH_LAST_GAP_M) / tread_depth - 1.0)
    index = torch.round(index_float).long()
    # Tread targets sit TREAD_TARGET_BACK_M behind their centre.
    on_grid = torch.where(
        on_stairs,
        tread_target_x(first_riser_x, tread_depth, index, step_count) - target_x,
        (index_float - index.to(index_float.dtype)) * tread_depth)
    good_index = ((on_grid.abs() < 1e-3 * tread_depth)
                  & (on_stairs | (index < 0))
                  & (index >= -approach_steps) & (index < step_count + 2))
    level = torch.where(index < 0, torch.zeros_like(index), (index + 1).clamp(1, step_count))
    plausible_height = ((target_z - level.to(target_z.dtype) * riser_height).abs() <= height_tolerance_m)
    good_level = good_index & plausible_height
    # Stair and platform cells are the real treads; approach cells are one
    # tread depth centred on the target.
    start = torch.where(on_stairs, first_riser_x + index.to(target_x.dtype) * tread_depth,
                        target_x - 0.5 * tread_depth)
    end = start + tread_depth
    x_min, x_max, y_min, y_max, z_min, z_max = sole_bounds.unbind(-1)
    if full_sole:
        in_cell = ((x_min >= (start[:, None] + margin_m)) &
                   (x_max <= (end[:, None] - margin_m)))
    else:
        centre_x = 0.5 * (x_min + x_max)
        in_cell = (centre_x >= start[:, None]) & (centre_x <= end[:, None])
        k = index[:, None]
        tread = (k >= 0) & (k < step_count)
        heel_on = x_min >= start[:, None] - TREAD_HEEL_OVERHANG_M
        toe_clear = (k >= step_count - 1) | (x_max <= end[:, None])
        spot_toe = x_max <= first_riser_x[:, None] - SPOT_TOE_CLEARANCE_M
        in_cell = torch.where(
            tread, heel_on & toe_clear,
            torch.where(k < 0, in_cell & spot_toe,
                        torch.where(k >= step_count,
                                    centre_x >= start[:, None], in_cell)))
    good = (in_cell &
            (y_min >= (-half_width + margin_m)) &
            (y_max <= (half_width - margin_m)) &
            ((z_min - target_z[:, None]).abs() <= height_tolerance_m) &
            ((z_max - target_z[:, None]).abs() <= height_tolerance_m) &
            (vertical_force >= support_force_n))
    assigned = torch.arange(2, device=feet.device)[None, :] == feet[:, 0, None]
    # Zero-riser rows are flat tiles in mixed training: nothing to qualify.
    stairs = (riser_height > 0)[:, None]
    return good & assigned & good_level[:, None] & valid[:, :1] & stairs


def qualified_flat_mask(
    *, sole_centre, sole_bounds, vertical_force, targets, feet, valid,
    half_width, xy_tolerance_m=0.08, ground_tolerance_m=0.025,
    support_force_n=30.0,
):
    """Qualify assigned-foot placement on flat ground.

    Geometric/contact-force approximation, not contact-patch proof.
    New touchdown detection remains the FootstepRuntime's responsibility.
    """
    n = sole_centre.shape[0]
    if (
        sole_centre.shape != (n, 2, 3)
        or sole_bounds.shape != (n, 2, 6)
        or vertical_force.shape != (n, 2)
        or targets.shape != (n, 2, 4)
        or feet.shape != (n, 2)
        or valid.shape != (n, 2)
    ):
        raise ValueError("invalid flat landing shapes")
    if min(half_width, xy_tolerance_m, ground_tolerance_m, support_force_n) <= 0:
        raise ValueError("invalid flat landing parameters")

    target = targets[:, 0, :3]
    xy_error = torch.linalg.vector_norm(
        sole_centre[..., :2] - target[:, None, :2], dim=-1
    )
    ground_error = (
        sole_bounds[..., 4] - target[:, None, 2]
    ).abs()

    lateral_ok = (
        (sole_bounds[..., 2] >= -half_width)
        & (sole_bounds[..., 3] <= half_width)
    )
    assigned = (
        torch.arange(2, device=feet.device)[None, :]
        == feet[:, 0, None]
    )

    return (
        assigned
        & valid[:, :1]
        & (xy_error <= xy_tolerance_m)
        & (ground_error <= ground_tolerance_m)
        & lateral_ok
        & (vertical_force >= support_force_n)
    )
