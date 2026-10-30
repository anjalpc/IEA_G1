"""Geometric full-sole tread qualifier for the opt-in footstep diagnostic.

This is a geometric/contact-force approximation, NOT a contact-patch proof.
No dependency on legacy acquisition state machines.
"""
import torch

from .footstep_tensor import APPROACH_LAST_GAP_M


def qualified_tread_mask(*, sole_bounds, vertical_force, targets, feet, valid,
                         first_riser_x, tread_depth, riser_height, step_count,
                         half_width, margin_m=0.005, height_tolerance_m=0.01,
                         support_force_n=30.0, approach_steps=0, full_sole=True):
    """Return [N,2] valid support on CURRENT target's level/tread.

    full_sole=True: the whole sole lies inside the cell (strict, evaluation).
    full_sole=False: the sole centre lies on the cell (training advancement;
    a 25 cm window instead of ~6 cm, cf. the reference's 0.20 m radius).
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
    good_index = (((index_float - index.to(index_float.dtype)).abs() < 1e-3)
                  & (on_stairs | (index < 0))
                  & (index >= -approach_steps) & (index < step_count + 2))
    level = torch.where(index < 0, torch.zeros_like(index), (index + 1).clamp(1, step_count))
    plausible_height = ((target_z - level.to(target_z.dtype) * riser_height).abs() <= height_tolerance_m)
    good_level = good_index & plausible_height
    start = target_x - 0.5 * tread_depth      # one tread depth centred on the target
    end = start + tread_depth
    x_min, x_max, y_min, y_max, z_min, z_max = sole_bounds.unbind(-1)
    if full_sole:
        in_cell = ((x_min >= (start[:, None] + margin_m)) &
                   (x_max <= (end[:, None] - margin_m)))
    else:
        centre_x = 0.5 * (x_min + x_max)
        in_cell = (centre_x >= start[:, None]) & (centre_x <= end[:, None])
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
