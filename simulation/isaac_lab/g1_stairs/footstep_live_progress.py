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
    # Called after termination calculation and before reward/reset.
    # Exclude terminating environments from target acquisition.
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    valid = valid.clone()
    valid[resets] = False
    # Diagnostic reward measurement uses the pre-transition assigned target.
    # No old stair reward functions; no planner update from a reward callback.
    sole_velocity = sole_state(
        data.body_pos_w[:, foot_ids], data.body_quat_w[:, foot_ids],
        data.body_lin_vel_w[:, foot_ids], data.body_ang_vel_w[:, foot_ids],
    )[2].mean(dim=-2)
    clock, swing, stance = coordinator.clock.features(
        target_foot=feet[:, 0], valid=valid[:, 0]
    )
    from .footstep_rewards import clock_contact_scores
    _stance_score, directed_swing = clock_contact_scores(
        force_z, centre, sole_velocity, targets, feet, valid,
        swing, stance,
    )
    directed_swing = torch.where(resets, torch.zeros_like(directed_swing), directed_swing)
    env._g1_footstep_directed_swing = directed_swing
    # Pre-transition assigned foothold; reward only new best distance.
    from .footstep_rewards import new_target_progress

    rows = torch.arange(env.num_envs, device=env.device)
    assigned = feet[:, 0].long()
    distance = torch.linalg.vector_norm(
        centre[rows, assigned] - targets[:, 0, :3], dim=-1
    )

    previous_best = getattr(env, "_g1_footstep_best_distance", None)
    previous_index = getattr(env, "_g1_footstep_progress_target_index", None)

    if previous_best is None or previous_index is None:
        previous_best = distance.clone()
    else:
        assert previous_best.shape == distance.shape
        changed = previous_index != coordinator.runtime.target_index
        previous_best = torch.where(changed | resets, distance, previous_best)

    progress, new_best = new_target_progress(
        distance, distance, previous_best, valid[:, 0] & ~resets
    )

    env._g1_footstep_net_progress = progress
    env._g1_footstep_best_distance = new_best.detach().clone()
    env._g1_footstep_progress_target_index = (
        coordinator.runtime.target_index.clone()
    )

    advanced = coordinator.step(
        step_id=int(env.common_step_counter), dt=float(env.step_dt),
        targets=targets, feet=feet, valid=valid, sole_xyz=centre,
        supported=supported, qualified_tread=qualified,
        xy_tolerance=0.08, z_tolerance=0.01,
    )
    env._g1_footstep_best_distance[advanced] = float("inf")
    env._g1_footstep_last_advanced = advanced
    return advanced
