"""Geometric full-sole tread qualifier for the opt-in footstep diagnostic.

This is a geometric/contact-force approximation, NOT a contact-patch proof.
No dependency on legacy acquisition state machines.
"""
import torch


def qualified_tread_mask(*, sole_bounds, vertical_force, targets, feet, valid,
                         first_riser_x, tread_depth, riser_height, step_count,
                         half_width, margin_m=0.005, height_tolerance_m=0.01,
                         support_force_n=30.0):
    """Return [N,2] valid full-sole support on CURRENT target's level/tread.

    Target XYZ denotes a planned sole centre. The active stair level comes from
    the current target's height; platform targets are checked against the
    platform surface. Only the assigned foot may pass.
    """
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
    if bool((tread_depth <= 2 * margin_m).any()) or bool((riser_height <= 0).any()):
        raise ValueError('invalid stair dimensions')
    target_z = targets[:, 0, 2]
    # Use target sequence's actual X to distinguish the two platform goals.
    # Both have the same upper-platform height but different tread intervals.
    index_float = ((targets[:, 0, 0] - first_riser_x) / tread_depth - 0.5)
    index = torch.round(index_float).long()
    good_index = ((index_float - index.to(index_float.dtype)).abs() < 1e-3) & (index >= 0) & (index < step_count + 2)
    level = (index + 1).clamp(1, step_count)
    plausible_height = ((target_z - level.to(target_z.dtype) * riser_height).abs() <= height_tolerance_m)
    good_level = good_index & plausible_height
    start = first_riser_x + index.to(target_z.dtype) * tread_depth
    end = start + tread_depth
    x_min, x_max, y_min, y_max, z_min, z_max = sole_bounds.unbind(-1)
    good = ((x_min >= (start[:, None] + margin_m)) &
            (x_max <= (end[:, None] - margin_m)) &
            (y_min >= (-half_width + margin_m)) &
            (y_max <= (half_width - margin_m)) &
            ((z_min - target_z[:, None]).abs() <= height_tolerance_m) &
            ((z_max - target_z[:, None]).abs() <= height_tolerance_m) &
            (vertical_force >= support_force_n))
    assigned = torch.arange(2, device=feet.device)[None, :] == feet[:, 0, None]
    return good & assigned & good_level[:, None] & valid[:, :1]
