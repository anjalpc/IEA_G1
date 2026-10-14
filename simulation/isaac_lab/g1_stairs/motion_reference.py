"""C4 geometry and smooth reference math. No simulator imports."""
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


def smooth(u):
    u = u.clamp(0, 1)
    return u**3 * (10 + u * (-15 + 6*u))


def swing_reference(start, landing, progress, clearance):
    """Raise, advance, lower; flat sole reference clears the whole riser.

    Terrain-relative XYZ extension of smooth polynomial swing shaping.
    Progress saturates: a missed touchdown never starts another reward cycle.
    """
    u = progress.clamp(0, 1)
    xy = smooth((u-.20)/.60)
    result = start + xy[:, None]*(landing-start)
    apex = torch.maximum(start[:, 2], landing[:, 2]) + clearance
    rise = start[:, 2] + smooth(u/.20)*(apex-start[:, 2])
    lower = apex + smooth((u-.80)/.20)*(landing[:, 2]-apex)
    result[:, 2] = torch.where(u < .20, rise, torch.where(u > .80, lower, apex))
    return result


def bounded_square(error, scale):
    # No zero-gradient hard saturation in the mathematical objective.
    squared = (error / scale).square()
    return squared / (1 + squared)


def advance_reference_phase(phase, frequency, active, acquired_foot, dt, previous_preferred):
    """C4 single clock owner: saturate old swing, then align new acquisition.

    Called only by GaitFrequencyAction.finish_step, after scoring old phase.
    Left swing occupies [.5, 1]; right swing occupies [0, .5]. At the
    endpoint hold (rather than wrap) until physical acquisition advances it.
    """
    import math
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    endpoint = torch.where(previous_preferred == 0, 1., .5)
    advanced = torch.minimum(phase + dt*frequency, endpoint)
    aligned = torch.where(acquired_foot == 0, 0.,
                          torch.where(acquired_foot == 1, .5, advanced))
    return torch.where(active, aligned, 0.)
