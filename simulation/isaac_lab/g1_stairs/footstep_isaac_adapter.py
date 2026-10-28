"""Read-only Isaac Lab bridge for planned footholds; no PPO or event side effects.

The caller owns FootstepRuntime progression/reset. This adapter reads only the
resolved stair course and current robot root pose. Calling it cannot advance
footsteps or manufacture contact events.
"""
from __future__ import annotations

import torch

from .footstep_gait_clock import NOMINAL_STEP_LENGTH_M
from .terrain_state import riser_height_per_env, stair_cfg


def course_tensors(
    env, *, foot_separation_m: float, flat_ground: bool = False,
    flat_start_x_m: float = 0.0,
    flat_step_length_m: float = NOMINAL_STEP_LENGTH_M,
):
    """Geometry in the per-env course frame used by terrain observations."""
    if not 0.0 < foot_separation_m:
        raise ValueError("foot_separation_m must be positive")
    cfg = stair_cfg(env)
    if flat_ground:
        if flat_step_length_m <= 0:
            raise ValueError("flat step length must be positive")
        height = torch.zeros(env.num_envs, device=env.device)
        return dict(
            first_riser_x=torch.full_like(height, flat_start_x_m),
            tread_depth=torch.full_like(height, flat_step_length_m),
            riser_height=height,
            centre_y=torch.zeros_like(height),
            foot_separation=torch.full_like(height, foot_separation_m),
        )
    height = riser_height_per_env(env)
    if height.shape != (env.num_envs,):
        raise ValueError("riser height must have one value per environment")
    if int(cfg.step_count) < 1 or float(cfg.tread_depth) <= 0:
        raise ValueError("invalid resolved stair course")
    return dict(
        first_riser_x=torch.full_like(height, float(cfg.start_clearance)),
        tread_depth=torch.full_like(height, float(cfg.tread_depth)),
        riser_height=height,
        centre_y=torch.zeros_like(height),
        foot_separation=torch.full_like(height, foot_separation_m),
    )


def course_root_pose(env):
    """Return local XYZ and yaw using Isaac's scalar-first root quaternion."""
    robot = env.scene["robot"]
    xyz = robot.data.root_pos_w - env.scene.env_origins
    w, x, y, z = robot.data.root_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return xyz, yaw


def target_observation(env, runtime, *, position_scale_m: float,
                       foot_separation_m: float, flat_ground: bool = False,
                       flat_start_x_m: float = 0.0,
                       flat_step_length_m: float = NOMINAL_STEP_LENGTH_M):
    """Read-only [N,14] target observation; never updates progress."""
    cfg = stair_cfg(env)
    if runtime.num_envs != env.num_envs or runtime.step_count != int(cfg.step_count):
        raise ValueError("footstep runtime and resolved stair course disagree")
    args = course_tensors(
        env, foot_separation_m=foot_separation_m,
        flat_ground=flat_ground, flat_start_x_m=flat_start_x_m,
        flat_step_length_m=flat_step_length_m,
    )
    targets, feet, valid = runtime.targets(**args, flat_ground=flat_ground)
    xyz, yaw = course_root_pose(env)
    return runtime.observation(targets=targets, feet=feet, valid=valid,
                               root_xyz=xyz, root_yaw=yaw,
                               position_scale=position_scale_m)
