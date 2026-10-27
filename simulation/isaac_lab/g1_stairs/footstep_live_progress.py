"""Opt-in post-step footstep/gait diagnostic.

Called after super().step(), hence returned observations are NOT recomputed.
Never enable for training until a pre-observation lifecycle hook is validated.
"""
import torch

from .footstep_live import ensure_runtime
from .footstep_coordinator import FootstepCoordinator
from .footstep_gait_clock import FootstepGaitClock
from .footstep_isaac_adapter import course_tensors
from .footstep_tread import qualified_tread_mask, qualified_flat_mask
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
    flat = bool(getattr(env.cfg, "flat_ground_footsteps", False))
    course = course_tensors(
        env, foot_separation_m=0.16, flat_ground=flat,
    )
    targets, feet, valid = coordinator.runtime.targets(
        **course, flat_ground=flat,
    )
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
    if flat:
        qualified = qualified_flat_mask(
            sole_centre=centre, sole_bounds=bounds,
            vertical_force=force_z, targets=targets,
            feet=feet, valid=valid,
            half_width=float(cfg.size[1]) / 2,
        )
    else:
        qualified = qualified_tread_mask(
            sole_bounds=bounds, vertical_force=force_z, targets=targets,
            feet=feet, valid=valid, step_count=coordinator.runtime.step_count,
            first_riser_x=course['first_riser_x'],
            tread_depth=course['tread_depth'],
            riser_height=course['riser_height'],
            half_width=float(cfg.size[1]) / 2,
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
    env._g1_footstep_stance_support = torch.where(
        resets, torch.zeros_like(_stance_score), _stance_score
    )
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


    # Assigned-foot peak clearance, only before crossing the riser.
    # Opposite-foot support is required; repeated lift/drop earns no
    # additional credit. This is measured before target acquisition.
    target_foot = feet[:, 0].long()
    other_foot = 1 - target_foot
    rows = torch.arange(env.num_envs, device=env.device)

    foot_bottom = bounds[rows, target_foot, 4]
    foot_front = bounds[rows, target_foot, 1]
    other_supported = supported[rows, other_foot]
    assigned_unloaded = ~supported[rows, target_foot]

    riser_x = course["first_riser_x"] + (
        coordinator.runtime.target_index.to(foot_front.dtype)
        * course["tread_depth"]
    )
    height_target = targets[:, 0, 2]
    approach_window = (
        (foot_front >= riser_x - 0.25)
        & (foot_front <= riser_x)
    )

    eligible = (
        valid[:, 0]
        & ~resets
        & other_supported
        & assigned_unloaded
        & approach_window
    )

    old_peak = getattr(env, "_g1_footstep_peak_clearance", None)
    old_target = getattr(env, "_g1_footstep_clearance_target_index", None)

    if old_peak is None or old_target is None:
        old_peak = torch.zeros_like(foot_bottom)
    else:
        changed = old_target != coordinator.runtime.target_index
        old_peak = torch.where(
            changed | resets,
            torch.zeros_like(old_peak),
            old_peak,
        )

    capped_height = foot_bottom.clamp(min=0.0)
    capped_height = torch.minimum(capped_height, height_target)
    improvement = (capped_height - old_peak).clamp_min(0.0)

    clearance = torch.where(
        eligible,
        improvement / height_target.clamp_min(0.01),
        torch.zeros_like(improvement),
    )

    if flat:
        clearance = torch.zeros_like(clearance)
        eligible = torch.zeros_like(eligible)

    env._g1_footstep_clearance_progress = clearance
    env._g1_footstep_peak_clearance = torch.where(
        eligible,
        torch.maximum(old_peak, capped_height),
        old_peak,
    ).detach().clone()
    env._g1_footstep_clearance_target_index = (
        coordinator.runtime.target_index.clone()
    )

    # A timeout is not a physical failure. Terminated environments
    # receive a separate event penalty.
    env._g1_footstep_failure_event = (
        env.termination_manager.terminated & ~env.termination_manager.time_outs
    ).float()
    # Preserve evidence before target-index progression.
    pre_target_index = coordinator.runtime.target_index.clone()
    pre_supported = coordinator.runtime.was_supported.clone()
    pre_primed = coordinator._primed.clone()

    advanced = coordinator.step(
        step_id=int(env.common_step_counter), dt=float(env.step_dt),
        targets=targets, feet=feet, valid=valid, sole_xyz=centre,
        supported=supported, qualified_tread=qualified,
        xy_tolerance=0.08,
        z_tolerance=0.025 if flat else 0.01,
    )
    env._g1_footstep_landing_audit = {
        "target_index_before": pre_target_index,
        "assigned_foot": feet[:, 0].clone(),
        "target_xyz_before": targets[:, 0, :3].clone(),
        "target_valid_before": valid[:, 0].clone(),
        "sole_centre_local": centre.clone(),
        "sole_bounds_local": bounds.clone(),
        "supported_lr": supported.clone(),
        "support_before_lr": pre_supported,
        "support_primed_before": pre_primed,
        "qualified_geometry_lr": qualified.clone(),
        "foot_planar_speed_lr_mps": torch.linalg.vector_norm(
            sole_velocity[..., :2], dim=-1
        ),
        "raw_acquisition": advanced.clone(),
    }
    env._g1_footstep_best_distance[advanced] = float("inf")
    env._g1_footstep_last_advanced = advanced
    return advanced
