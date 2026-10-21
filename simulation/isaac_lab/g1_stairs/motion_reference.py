"""C6 coordinated stair-motion reference math. No simulator imports."""
import math
import torch
from .contracts import G1_SOLE_SPHERE_CENTRES_M, G1_SOLE_SPHERE_RADIUS_M


def rotate(q, v):
    xyz = q[..., 1:].expand_as(v)
    cross = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * cross + torch.cross(xyz, cross, dim=-1)


def sole_state(position, quaternion, velocity, omega):
    """Collision-sphere envelope and surface velocity, not contact-patch area."""
    offsets = position.new_tensor(G1_SOLE_SPHERE_CENTRES_M)
    offsets = offsets.expand(*position.shape[:-1], -1, -1)
    rotated = rotate(quaternion.unsqueeze(-2), offsets)
    points = position.unsqueeze(-2) + rotated
    radius = G1_SOLE_SPHERE_RADIUS_M
    bottom = points.clone()
    bottom[..., 2] -= radius
    surface_offset = rotated.clone()
    surface_offset[..., 2] -= radius
    surface_velocity = velocity.unsqueeze(-2) + torch.cross(
        omega.unsqueeze(-2).expand_as(surface_offset), surface_offset, dim=-1
    )
    bounds = torch.stack((points[..., 0].amin(-1)-radius,
                          points[..., 0].amax(-1)+radius,
                          points[..., 1].amin(-1)-radius,
                          points[..., 1].amax(-1)+radius,
                          bottom[..., 2].amin(-1), bottom[..., 2].amax(-1)), -1)
    return bottom.mean(-2), bounds, surface_velocity


def foot_riser_distance(next_riser_x, sole_bounds, tread_depth, active):
    """Normalized signed distance from each sole front edge to the next riser."""
    if tread_depth <= 0.0:
        raise ValueError("tread_depth must be positive")
    foot_span = sole_bounds[..., 1] - sole_bounds[..., 0]
    scale = (2.0 * tread_depth + foot_span).clamp_min(1.0e-6)
    distance = (next_riser_x[:, None] - sole_bounds[..., 1]) / scale
    return active[:, None] * distance.clamp(-1.0, 1.0)



def strict_touchdown_height_tolerance(riser_height):
    """Shared strict stair touchdown tolerance: 20% of riser, capped at 1 cm."""
    return torch.minimum(
        riser_height * 0.20,
        torch.full_like(riser_height, 0.01),
    )


def strict_tread_landing_mask(
    bounds, vertical_force, speed, tilt, yaw, *, tread_start, tread_end,
    tread_top, lane_target, terrain_half_width, riser_height,
    tread_margin_m=0.025, center_tolerance_m=0.010, lane_tolerance_m=0.050,
    min_force_n=30.0, max_speed_mps=0.20, max_tilt_rad=0.15, max_yaw_rad=0.20,
):
    """Shared physical acceptance definition for one strict tread landing."""
    center_x = 0.5 * (bounds[..., 0] + bounds[..., 1])
    center_y = 0.5 * (bounds[..., 2] + bounds[..., 3])
    tread_center = 0.5 * (tread_start + tread_end)
    height_tolerance = strict_touchdown_height_tolerance(riser_height)
    return (
        (vertical_force >= min_force_n)
        & (bounds[..., 0] >= tread_start + tread_margin_m)
        & (bounds[..., 1] <= tread_end - tread_margin_m)
        & (bounds[..., 2] >= -terrain_half_width + tread_margin_m)
        & (bounds[..., 3] <= terrain_half_width - tread_margin_m)
        & ((center_x - tread_center).abs() <= center_tolerance_m)
        & ((center_y - lane_target).abs() <= lane_tolerance_m)
        & ((bounds[..., 4] - tread_top).abs() <= height_tolerance)
        & ((bounds[..., 5] - tread_top).abs() <= height_tolerance)
        & (speed <= max_speed_mps)
        & (tilt <= max_tilt_rad)
        & (yaw.abs() <= max_yaw_rad)
    )


def riser_contact_masks(
    bounds, force, *, first_riser, tread, steps, riser_height, active,
    support_force_n=30.0, face_epsilon_m=0.005,
):
    """Return per-environment ``(strike, bridge)`` from sole/riser overlap.

    Strike covers collisions anywhere on the vertical riser face, including
    below-edge impacts. Bridge is reserved for a loaded sole spanning both
    sides of the upper tread surface.
    """
    edge_x = first_riser + torch.arange(
        steps, device=bounds.device, dtype=bounds.dtype
    ) * tread
    idx = torch.arange(steps, device=bounds.device, dtype=bounds.dtype)
    face_bottom = riser_height[:, None] * idx
    face_top = riser_height[:, None] * (idx + 1)

    x_overlap = (bounds[..., 0, None] <= edge_x) & (bounds[..., 1, None] >= edge_x)
    # bounds[..., 4:6] are sphere-bottom heights. Strike detection must use
    # the physical collision-sphere vertical envelope so a low horizontal toe
    # can still overlap the riser face.
    collider_bottom = bounds[..., 4]
    collider_top = bounds[..., 5] + 2.0 * G1_SOLE_SPHERE_RADIUS_M
    z_face_overlap = (
        (collider_bottom[..., None] <= face_top[:, None] - face_epsilon_m)
        & (collider_top[..., None] >= face_bottom[:, None] + face_epsilon_m)
    )
    horizontal_force = force[..., :2].norm(dim=-1)
    strike = (
        x_overlap
        & z_face_overlap
        & (horizontal_force[..., None] >= support_force_n)
        & active[:, None, None]
    ).any(dim=(-1, -2))

    spans_upper_surface = (
        x_overlap
        & (bounds[..., 4, None] < face_top[:, None] - face_epsilon_m)
        & (bounds[..., 5, None] > face_top[:, None] + face_epsilon_m)
    )
    loaded = force[..., 2].clamp_min(0) >= support_force_n
    bridge = (
        spans_upper_surface
        & loaded[..., None]
        & active[:, None, None]
    ).any(dim=(-1, -2))
    return strike, bridge

def smooth(u):
    """Quintic smoothstep with zero velocity and acceleration at both ends."""
    u = u.clamp(0, 1)
    return u**3 * (10 + u * (-15 + 6*u))


def swing_reference(start, landing, progress, clearance):
    """Smooth overlapping forward/lift/lower trajectory.

    Horizontal motion spans the whole swing instead of waiting for a vertical
    lift plateau. Vertical endpoint interpolation uses the same quintic clock,
    with a smooth mid-swing clearance bump whose value and first derivative
    vanish at touchdown/liftoff.
    """
    u = progress.clamp(0, 1)
    s = smooth(u)
    result = start + s[:, None] * (landing-start)
    # 16*u^2*(1-u)^2 peaks at one at mid swing and is zero with zero slope
    # at both endpoints. The quintic endpoint interpolation keeps endpoint
    # acceleration small as well.
    bump = 64.0 * u.pow(3) * (1-u).pow(3)
    midpoint = .5 * (start[:, 2] + landing[:, 2])
    apex = torch.maximum(start[:, 2], landing[:, 2]) + clearance
    bump_amplitude = (apex-midpoint).clamp_min(0)
    result[:, 2] += bump_amplitude * bump
    return result


def bounded_square(error, scale):
    squared = (error / scale).square()
    return squared / (1 + squared)


def normalized_huber(error, delta):
    """Dimensionless Huber loss; remains linearly informative for large errors."""
    if not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError('delta must be finite and positive')
    x = error.abs() / delta
    return torch.where(x <= 1, .5*x.square(), x-.5)


def target_support_mask(bounds, vertical_force, *, tread_start, tread_end,
                        tread_top, terrain_half_width, min_force_n,
                        height_tolerance_m):
    """True only when the full sole envelope is loaded on the intended tread."""
    return (
        (vertical_force >= min_force_n)
        & (bounds[..., 0] >= tread_start)
        & (bounds[..., 1] <= tread_end)
        & (bounds[..., 2] >= -terrain_half_width)
        & (bounds[..., 3] <= terrain_half_width)
        & ((bounds[..., 4]-tread_top).abs() <= height_tolerance_m)
    )


def provisional_supported_acquisition_mask(
    bounds,
    vertical_force,
    *,
    intended_foot,
    support_transition,
    previous_acquired_foot,
    strike,
    bridge,
    tread_start,
    tread_end,
    tread_top,
    terrain_half_width,
    min_force_n,
    height_tolerance_m,
):
    """Relaxed curriculum acquisition, kept separate from strict success.

    Precision/style constraints such as tread-centre error, yaw, tilt and
    touchdown speed are intentionally absent.  The event still requires one
    intended-foot support transition onto the intended tread, full sole
    containment, meaningful load, and no riser strike/bridge.
    """
    if support_transition.shape != vertical_force.shape:
        raise ValueError("support_transition must match per-foot force shape")
    if bounds.shape[:-1] != vertical_force.shape:
        raise ValueError("bounds and vertical_force must describe the same feet")
    if bounds.shape[-1] != 6:
        raise ValueError("bounds must contain [xmin,xmax,ymin,ymax,zmin,zmax]")

    support = target_support_mask(
        bounds,
        vertical_force,
        tread_start=tread_start,
        tread_end=tread_end,
        tread_top=tread_top,
        terrain_half_width=terrain_half_width,
        min_force_n=min_force_n,
        height_tolerance_m=height_tolerance_m,
    )

    rows = torch.arange(vertical_force.shape[0], device=vertical_force.device)
    foot = intended_foot.clamp(0, vertical_force.shape[1] - 1)
    intended_valid = (intended_foot >= 0) & (intended_foot < vertical_force.shape[1])
    intended_event = support_transition[rows, foot] & support[rows, foot]
    unique_transition = support_transition.sum(dim=-1) == 1
    alternating = (previous_acquired_foot < 0) | (previous_acquired_foot != intended_foot)

    return (
        intended_valid
        & intended_event
        & unique_transition
        & alternating
        & ~strike
        & ~bridge
    )


def advance_reference_phase(phase, frequency, active, acquired_foot, dt,
                            previous_preferred, swing_started=None,
                            expired=None):
    """Single clock owner with liftoff hold and finite touchdown grace.

    `active` means a transition is selected. The clock remains at its aligned
    start until `swing_started`. A caller can mark `expired` after touchdown
    grace; expired transitions become inactive rather than freezing forever.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if swing_started is None:
        swing_started = active
    if expired is None:
        expired = torch.zeros_like(active)
    endpoint = torch.where(previous_preferred == 0, 1., .5)
    advanced = torch.minimum(phase + dt*frequency, endpoint)
    held_or_advanced = torch.where(swing_started, advanced, phase)
    aligned = torch.where(acquired_foot == 0, 0.,
                          torch.where(acquired_foot == 1, .5, held_or_advanced))
    return torch.where(active & ~expired, aligned, torch.zeros_like(phase))


def transition_quality_escrow_update(total, count, sample_quality,
                                     terminal_quality, active, event_code,
                                     discard=None):
    """Accumulate bounded quality; release once only on DIRECT acquisition."""
    if total.shape != count.shape or total.shape != sample_quality.shape or total.shape != terminal_quality.shape:
        raise ValueError('escrow tensors must have matching shapes')
    if active.shape != total.shape or event_code.shape != total.shape:
        raise ValueError('escrow masks/codes must match state shape')
    if discard is None:
        discard = torch.zeros_like(active)
    quality = sample_quality.clamp(0, 1)
    terminal_quality = terminal_quality.clamp(0, 1)
    add = (active & ~discard).to(total.dtype)
    updated_total = total + quality * add
    updated_count = count + add
    direct = active & ~discard & (event_code == 1)
    terminal = (event_code != 0) | discard
    mean_quality = updated_total / updated_count.clamp_min(1)
    released = torch.where(direct, mean_quality * terminal_quality, torch.zeros_like(mean_quality))
    updated_total = torch.where(terminal, torch.zeros_like(updated_total), updated_total)
    updated_count = torch.where(terminal, torch.zeros_like(updated_count), updated_count)
    return updated_total, updated_count, released
