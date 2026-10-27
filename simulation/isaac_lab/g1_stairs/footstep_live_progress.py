"""Opt-in post-step footstep/gait diagnostic.

Called after super().step(), hence returned observations are NOT recomputed.
Never enable for training until a pre-observation lifecycle hook is validated.
"""
import torch

from .footstep_live import ensure_runtime
from .footstep_coordinator import FootstepCoordinator
from .footstep_gait_clock import FootstepGaitClock
from .footstep_isaac_adapter import course_tensors
from .footstep_tread import qualified_tread_mask
from .contact_sensor_contract import FOOT_BODY_NAMES
from .motion_reference import sole_state
from .terrain_state import stair_cfg


def ensure_coordinator(env):
    coord = getattr(env, '_g1_footstep_coordinator', None)
    if coord is None:
        coord = FootstepCoordinator(
            runtime=ensure_runtime(env),
            clock=FootstepGaitClock(num_envs=env.num_envs, device=env.device,
                                    single_support_s=0.8, double_support_s=0.2),
        )
        env._g1_footstep_coordinator = coord
    return coord


def reset_coordinator(env, env_ids):
    """Called after existing reset events; reset both runtime and clock."""
    if getattr(env, '_g1_footstep_coordinator', None) is not None:
        env._g1_footstep_coordinator.reset(env_ids)


def audit_after_step(env):
    """Read post-physics state; do not mutate the returned policy observation."""
    coordinator = ensure_coordinator(env)
    course = course_tensors(env, foot_separation_m=0.16)
    targets, feet, valid = coordinator.runtime.targets(**course)
    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    foot_ids = [list(robot.body_names).index(name) for name in FOOT_BODY_NAMES]
    sensor_ids = [list(sensor.body_names).index(name) for name in FOOT_BODY_NAMES]
    data = robot.data
    centre, bounds, _ = sole_state(
        data.body_pos_w[:, foot_ids], data.body_quat_w[:, foot_ids],
        data.body_lin_vel_w[:, foot_ids], data.body_ang_vel_w[:, foot_ids],
    )
    centre = centre - env.scene.env_origins[:, None, :]
    bounds = bounds.clone()
    bounds[..., 0:4] -= env.scene.env_origins[:, None, [0, 0, 1, 1]]
    bounds[..., 4:6] -= env.scene.env_origins[:, None, 2, None]
    force_z = sensor.data.net_forces_w[:, sensor_ids, 2]
    supported = force_z >= 30.0
    cfg = stair_cfg(env)
    qualified = qualified_tread_mask(
        sole_bounds=bounds, vertical_force=force_z, targets=targets,
        feet=feet, valid=valid, step_count=coordinator.runtime.step_count,
        first_riser_x=course['first_riser_x'], tread_depth=course['tread_depth'],
        riser_height=course['riser_height'], half_width=float(cfg.size[1]) / 2,
    )
    # super().step() has ALREADY auto-reset terminating envs. Suppress progress
    # in those rows; no reset frame may manufacture an acquisition.
    resets = env.reset_buf.bool()
    valid = valid.clone()
    valid[resets] = False
    advanced = coordinator.step(
        step_id=int(env.common_step_counter), dt=float(env.step_dt),
        targets=targets, feet=feet, valid=valid, sole_xyz=centre,
        supported=supported, qualified_tread=qualified,
        xy_tolerance=0.08, z_tolerance=0.01,
    )
    env._g1_footstep_last_advanced = advanced
    return advanced
