"""Batched, side-effect-free footstep targets for Isaac Lab policies.

The caller supplies course geometry and root poses in LOCAL coordinates.
Target indexes come from a separate contact-driven progress tracker; this
module never fabricates contact or advancement from root displacement.
"""
from __future__ import annotations

import torch

OBS_WIDTH = 22
METRIC_CONTRACT = "footstep_target_v1"


def planned_target_batch(*, first_riser_x: torch.Tensor,
                         tread_depth: torch.Tensor, riser_height: torch.Tensor,
                         step_count: int, first_foot: torch.Tensor,
                         target_index: torch.Tensor, centre_y: torch.Tensor,
                         foot_separation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [N,2,4] world-course target poses, [N,2] assigned feet,
    and [N,2] validity. First-foot and target index may differ by env.
    """
    if step_count <= 0:
        raise ValueError("step_count must be positive")
    n = target_index.shape[0]
    if any(x.shape != (n,) for x in (first_riser_x, tread_depth, riser_height,
                                    first_foot, centre_y, foot_separation)):
        raise ValueError("all inputs must be per-environment vectors")
    if torch.any(tread_depth <= 0) or torch.any(riser_height <= 0):
        raise ValueError("course dimensions must be positive")
    idx = target_index[:, None] + torch.arange(2, device=target_index.device)
    valid = (idx >= 0) & (idx < step_count + 2)
    foot = (first_foot[:, None] + idx) % 2
    level = (idx + 1).clamp(1, step_count)
    x = first_riser_x[:, None] + (idx + .5) * tread_depth[:, None]
    y = centre_y[:, None] + torch.where(foot == 0, .5, -.5) * foot_separation[:, None]
    z = level * riser_height[:, None]
    yaw = torch.zeros_like(x)
    targets = torch.stack((x, y, z, yaw), dim=-1)
    return torch.where(valid[..., None], targets, torch.zeros_like(targets)), foot, valid


def footstep_observation(*, targets: torch.Tensor, feet: torch.Tensor,
                         valid: torch.Tensor, root_xyz: torch.Tensor,
                         root_yaw: torch.Tensor, phase: torch.Tensor,
                         position_scale: float) -> torch.Tensor:
    """22-D, clock and two 10-D target slots; zero finished slots."""
    if position_scale <= 0:
        raise ValueError("position_scale must be positive")
    n = root_xyz.shape[0]
    if targets.shape != (n, 2, 4) or feet.shape != (n, 2) or valid.shape != (n, 2):
        raise ValueError("invalid target batch shapes")
    if root_xyz.shape != (n, 3) or root_yaw.shape != (n,) or phase.shape != (n,):
        raise ValueError("invalid root/phase shapes")
    delta = targets[..., :3] - root_xyz[:, None, :]
    c, s = torch.cos(root_yaw[:, None]), torch.sin(root_yaw[:, None])
    dx = c * delta[..., 0] + s * delta[..., 1]
    dy = -s * delta[..., 0] + c * delta[..., 1]
    dyaw = torch.atan2(torch.sin(targets[..., 3] - root_yaw[:, None]),
                       torch.cos(targets[..., 3] - root_yaw[:, None]))
    slots = torch.stack((valid.float(), (feet == 0).float(), (feet == 1).float(),
                         dx / position_scale, dy / position_scale,
                         delta[..., 2] / position_scale, dyaw / torch.pi,
                         targets[..., 2] / position_scale,
                         torch.zeros_like(dx),
                         torch.tensor((1., 0.), device=dx.device, dtype=dx.dtype).expand(n, -1)), dim=-1)
    slots = torch.where(valid[..., None], slots, torch.zeros_like(slots))
    return torch.cat((torch.sin(2 * torch.pi * phase)[:, None],
                      torch.cos(2 * torch.pi * phase)[:, None],
                      slots.reshape(n, -1)), dim=1)


def footstep_scores(*, foot_distance: torch.Tensor, midpoint_distance: torch.Tensor,
                     measured_velocity: torch.Tensor, commanded_velocity: torch.Tensor,
                     swing_force: torch.Tensor, support_force: torch.Tensor,
                     swing_speed: torch.Tensor, swing_phase: torch.Tensor,
                     distance_scale: float, velocity_scale: float,
                     force_scale: float, speed_scale: float) -> dict[str, torch.Tensor]:
    """Bounded positive continuous reward components; not terminal success."""
    if min(distance_scale, velocity_scale, force_scale, speed_scale) <= 0:
        raise ValueError("all reward scales must be positive")
    if not all(x.shape == foot_distance.shape for x in
               (midpoint_distance, measured_velocity, commanded_velocity,
                swing_force, support_force, swing_speed, swing_phase)):
        raise ValueError("reward inputs must have equal shapes")
    phase = swing_phase.clamp(0, 1)
    load = (swing_force / force_scale).clamp(0, 1)
    support = (support_force / force_scale).clamp(0, 1)
    speed = (swing_speed / speed_scale).clamp(0, 1)
    return {
        "footstep_target": torch.exp(-foot_distance.clamp_min(0) / distance_scale),
        "footstep_root_progress": torch.exp(-midpoint_distance.clamp_min(0) / distance_scale),
        "footstep_velocity_tracking": torch.exp(-((measured_velocity - commanded_velocity) / velocity_scale).square()),
        "footstep_force_phase": ((1-phase)*load + phase*(1-load) + support)/2,
        "footstep_velocity_phase": phase*speed + (1-phase)*(1-speed),
    }
