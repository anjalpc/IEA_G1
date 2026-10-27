"""Compact planned-footstep policy input; independent of legacy stair rewards.

This is an observation encoder only. Targets and progress must be supplied by a
separate planner/contact-aware adapter; no inference from root displacement.
"""
import math
import torch

# Each slot: [valid, left, right, local_x, local_y, relative_z, relative_yaw].
SLOT_WIDTH = 7
OBS_WIDTH = 2 * SLOT_WIDTH


def encode_next_two(*, targets: torch.Tensor, feet: torch.Tensor,
                    valid: torch.Tensor, root_xyz: torch.Tensor,
                    root_yaw: torch.Tensor, position_scale: float) -> torch.Tensor:
    """Return [N,14] current-frame targets; invalid slots are entirely zero.

    Input target coordinates are course-local [x,y,z,yaw], and root_xyz must be
    expressed in the same course-local frame. Foot convention: 0=L, 1=R.
    """
    if not math.isfinite(position_scale) or position_scale <= 0:
        raise ValueError("position_scale must be finite and positive")
    if root_xyz.ndim != 2 or root_xyz.shape[1] != 3:
        raise ValueError("root_xyz must have shape [N,3]")
    n = root_xyz.shape[0]
    if (targets.shape != (n, 2, 4) or feet.shape != (n, 2)
            or valid.shape != (n, 2) or root_yaw.shape != (n,)):
        raise ValueError("invalid footstep observation shapes")
    if (targets.device != root_xyz.device or feet.device != root_xyz.device
            or valid.device != root_xyz.device or root_yaw.device != root_xyz.device):
        raise ValueError("all observation tensors must be on the same device")
    if torch.any(valid & ((feet != 0) & (feet != 1))):
        raise ValueError("valid target foot must be 0 or 1")
    if not (torch.isfinite(targets[valid]).all() and
            torch.isfinite(root_xyz).all() and torch.isfinite(root_yaw).all()):
        raise ValueError("active targets and robot pose must be finite")

    delta = targets[..., :3] - root_xyz[:, None, :]
    c, s = torch.cos(root_yaw[:, None]), torch.sin(root_yaw[:, None])
    local_x = (c * delta[..., 0] + s * delta[..., 1]) / position_scale
    local_y = (-s * delta[..., 0] + c * delta[..., 1]) / position_scale
    local_z = delta[..., 2] / position_scale
    yaw_delta = targets[..., 3] - root_yaw[:, None]
    local_yaw = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)) / torch.pi
    out = torch.stack((valid.to(local_x.dtype), (feet == 0).to(local_x.dtype),
                       (feet == 1).to(local_x.dtype), local_x, local_y,
                       local_z, local_yaw), dim=-1)
    return torch.where(valid[..., None], out, torch.zeros_like(out)).reshape(n, OBS_WIDTH)
