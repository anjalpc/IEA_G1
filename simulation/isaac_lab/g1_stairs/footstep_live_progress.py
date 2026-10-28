"""Footstep progress hook for the footstep tasks.

Training path (FootstepManagerBasedRLEnv): audit_after_step runs inside
termination_manager.compute(), i.e. after physics and terminations and
BEFORE reward computation, resets and observation computation of the same
Isaac Lab 2.3 step. Rewards and the next observation therefore see the
updated plan. (The legacy footstep_progress_diagnostic flag, which called
this after super().step(), is not used by the footstep tasks.)
"""
import torch

from .footstep_live import ensure_runtime
from .footstep_coordinator import FootstepCoordinator
from .footstep_gait_clock import (
    FootstepGaitClock, NOMINAL_DOUBLE_SUPPORT_S, NOMINAL_SINGLE_SUPPORT_S,
)
from .footstep_isaac_adapter import course_root_pose, course_tensors
from .footstep_mixing import course_exit_timeout, stair_escape, target_stalled
from .footstep_tensor import NOMINAL_FOOT_SEPARATION_M
from .footstep_tread import qualified_tread_mask, qualified_flat_mask
from .contact_sensor_contract import FOOT_BODY_NAMES
from .motion_reference import sole_state
from .terrain_state import riser_height_per_env, stair_cfg


def ensure_coordinator(env):
    coord = getattr(env, '_g1_footstep_coordinator', None)
    if coord is None:
        coord = FootstepCoordinator(
            runtime=ensure_runtime(env),
            clock=FootstepGaitClock(num_envs=env.num_envs, device=env.device,
                                    single_support_s=NOMINAL_SINGLE_SUPPORT_S,
                                    double_support_s=NOMINAL_DOUBLE_SUPPORT_S),
        )
        env._g1_footstep_coordinator = coord
    return coord


def course_exit(env):
    """Time-out (not failure) when the root leaves the safe tile region."""
    xyz, yaw = course_root_pose(env)
    return course_exit_timeout(xyz[:, :2], yaw, riser_height_per_env(env) > 0)


def stair_target_stall(env):
    """Failure when a stair env makes no plan progress for too long."""
    since = getattr(env, "_g1_footstep_since_advance", None)
    if since is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    runtime = ensure_coordinator(env).runtime
    active = runtime.target_index < runtime.plan_length(False)
    return target_stalled(since, active, riser_height_per_env(env) > 0)


def stair_course_escape(env):
    """Failure when a stair env leaves its corridor or turns away."""
    xyz, yaw = course_root_pose(env)
    return stair_escape(xyz[:, :2], yaw, riser_height_per_env(env) > 0)


def reset_coordinator(env, env_ids):
    """Called after existing reset events; reset both runtime and clock."""
    if getattr(env, '_g1_footstep_coordinator', None) is not None:
        env._g1_footstep_coordinator.reset(env_ids)


def audit_after_step(env):
    """Read post-physics state; do not mutate the returned policy observation."""
    coordinator = ensure_coordinator(env)
    flat = bool(getattr(env.cfg, "flat_ground_footsteps", False))
    course = course_tensors(
        env, foot_separation_m=NOMINAL_FOOT_SEPARATION_M, flat_ground=flat,
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
            approach_steps=coordinator.runtime.approach_for(flat),
            full_sole=False,
        )
        # Strict full-sole placement, recorded for evaluation only.
        strict = qualified_tread_mask(
            sole_bounds=bounds, vertical_force=force_z, targets=targets,
            feet=feet, valid=valid, step_count=coordinator.runtime.step_count,
            first_riser_x=course['first_riser_x'],
            tread_depth=course['tread_depth'],
            riser_height=course['riser_height'],
            half_width=float(cfg.size[1]) / 2,
            approach_steps=coordinator.runtime.approach_for(flat),
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


    # Assigned-foot peak lift over the riser of its tread target, only in
    # the 0.25 m before that riser. Opposite-foot support is required;
    # repeated lift/drop earns no additional credit. Measured before target
    # acquisition. Approach and platform targets have no riser: no credit.
    from .footstep_rewards import tread_lift

    target_foot = feet[:, 0].long()
    other_foot = 1 - target_foot
    rows = torch.arange(env.num_envs, device=env.device)

    foot_bottom = bounds[rows, target_foot, 4]
    foot_front = bounds[rows, target_foot, 1]
    other_supported = supported[rows, other_foot]
    assigned_unloaded = ~supported[rows, target_foot]

    plan_index = (
        coordinator.runtime.target_index
        - coordinator.runtime.approach_for(flat)
    ).to(foot_front.dtype)
    riser_x = course["first_riser_x"] + plan_index * course["tread_depth"]
    lift, lift_goal = tread_lift(
        foot_bottom, targets[:, 0, 2], course["riser_height"], plan_index,
        coordinator.runtime.step_count,
    )
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
        & (plan_index >= 0)
        & (plan_index < coordinator.runtime.step_count)
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

    improvement = (lift - old_peak).clamp_min(0.0)

    clearance = torch.where(
        eligible,
        improvement / lift_goal,
        torch.zeros_like(improvement),
    )

    if flat:
        clearance = torch.zeros_like(clearance)
        eligible = torch.zeros_like(eligible)

    env._g1_footstep_clearance_progress = clearance
    env._g1_footstep_peak_clearance = torch.where(
        eligible,
        torch.maximum(old_peak, lift),
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
        # Stairs: anywhere on the tread along x (half tread depth).
        xy_tolerance=0.08 if flat else 0.125,
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
        "strict_full_sole_lr": (qualified if flat else strict).clone(),
        "foot_planar_speed_lr_mps": torch.linalg.vector_norm(
            sole_velocity[..., :2], dim=-1
        ),
        "raw_acquisition": advanced.clone(),
    }
    env._g1_footstep_best_distance[advanced] = float("inf")
    env._g1_footstep_last_advanced = advanced
    # Stall terminations this step (same condition the termination term saw:
    # the timer is only advanced below).
    env._g1_footstep_stall_event = (
        env.termination_manager.terminated & stair_target_stall(env)
    ).float()
    since = getattr(env, "_g1_footstep_since_advance", None)
    if since is None:
        since = torch.zeros(env.num_envs, device=env.device)
    since = since + float(env.step_dt)
    since[advanced] = 0.0
    env._g1_footstep_since_advance = since
    stop_command_after_plan(env, coordinator.runtime, flat, resets)
    return advanced


def stop_command_after_plan(env, runtime, flat_ground, resets):
    """Plan complete: zero the velocity command so the prior stands still.

    Replaces the removed clock stand mode. The command term may resample
    on its own timer; this runs every policy step, so at most one step can
    observe the resampled value before it is zeroed again.
    """
    done = (runtime.target_index >= runtime.plan_length(flat_ground)) & ~resets
    if bool(done.any()):
        command = env.command_manager.get_command("base_velocity")
        command[done] = 0.0
    return done
