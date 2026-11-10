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
from .footstep_clock import (
    RollingClockState, approach_steps_left, clock_schedule, follow_plan_foot,
    heading_targets, landing_impact, late_swing_cap, moving, own_window,
    rolling_origin, speed_period, stair_period, swing_progress, follow_heading)
from .footstep_mixing import (
    course_exit_timeout, stair_escape, stair_steering, target_stalled,
)
from .footstep_tensor import NOMINAL_FOOT_SEPARATION_M, approach_target_x
from .footstep_rewards import stair_ground_height
from .footstep_tread import (qualified_tread_mask, qualified_flat_mask,
                             tread_placement_quality)
from .contact_sensor_contract import FOOT_BODY_NAMES
from .motion_reference import sole_state
from .terrain_state import riser_height_per_env, stair_cfg

# Stair plan hit: sideways limit of the distance check (patch 60).
STAIR_HIT_LATERAL_M = 0.20


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
    clock = getattr(env, "_g1_rolling_clock", None)
    if clock is not None:
        active = active & clock.on_plan      # the approach has no plan yet
    return target_stalled(since, active, riser_height_per_env(env) > 0)


def stair_course_escape(env):
    """Failure when a stair env leaves its corridor or turns away."""
    xyz, yaw = course_root_pose(env)
    return stair_escape(xyz[:, :2], yaw, riser_height_per_env(env) > 0)


def ensure_rolling_clock(env):
    """Rolling clock state when ``env.cfg.footstep_clock`` is enabled."""
    if getattr(env.cfg, "footstep_clock", None) is None:
        return None
    state = getattr(env, "_g1_rolling_clock", None)
    if state is None:
        state = RollingClockState(env.num_envs, env.device)
        state.reset(None, env.cfg.footstep_clock["period_jitter"])
        env._g1_rolling_clock = state
    return state


def apply_rolling_clock(env, state, *, centre, targets, feet, valid, resets,
                        first_plan_x, terrain_z=None):
    """One clock for all tiles.

    Rolling rows (flat tiles and the approach of stair tiles): a new target
    is placed one step ahead of the stance foot (on flat tiles and after the
    stairs: ahead of that foot's previous target, see rolling_origin), along the robot's heading
    plus the commanded turn, whenever the clock hands the swing to the other
    foot (targets/feet/valid updated in place). Approach rows count their
    steps down to the plan's first target (``first_plan_x``) in equal steps
    and hand over to the plan when one step is left; the caller then starts
    the plan with the returned feet. Plan rows keep the plan's targets, use
    the stair cycle of their riser (stair_period), the plan decides which
    foot gets the swing window, and a target is only valid in its foot's own
    window. Rows that
    completed their plan roll on like flat tiles. ``terrain_z`` maps [N,K]
    course x to ground height for the rolling targets (None: 0).
    Returns (rolling_rows, walking, swing_schedule, handover).
    """
    riser = riser_height_per_env(env)
    stairs = riser > 0
    plan = stairs & state.on_plan
    rows = ~plan
    command = env.command_manager.get_command("base_velocity")
    walking = ((moving(command) & rows) | (plan & valid[:, 0])) & ~resets
    cfg = env.cfg.footstep_clock
    state.period_s = torch.where(
        rows, speed_period(command[:, 0], state.period_scale, cfg),
        stair_period(riser, state.period_scale, cfg))
    state.phase = follow_plan_foot(state.phase, state.swinging, feet[:, 0], plan)
    schedule, assigned = clock_schedule(state.phase)
    state.swinging = schedule.any(dim=-1)
    state.landed &= schedule                # a new window starts a new swing
    _, yaw = course_root_pose(env)
    # Patch 56: targets are laid along a heading that follows the commanded
    # turn at once and the root yaw over one cycle (footstep_clock.follow_heading).
    # Standing or just starting to walk: no twist to filter, take the root yaw.
    keep = state.heading_valid & state.active & walking
    state.heading = torch.where(keep, state.heading, yaw)
    state.heading = follow_heading(state.heading, yaw, command[:, 2], env.step_dt, state.period_s)
    state.heading_valid[:] = True
    yaw = state.heading
    # New target when the clock hands the swing to the other foot, and when
    # walking resumes after standing (the held target would be stale).
    new_window = rows & ((assigned != state.assigned) | (walking & ~state.active))
    index = torch.arange(env.num_envs, device=env.device)
    distance = first_plan_x - centre[index, 1 - assigned, 0]
    step_len = command[:, 0] * 0.5 * state.period_s
    steps_left = approach_steps_left(distance, step_len)
    approach = stairs & rows & ~state.climbed
    handover = approach & new_window & (steps_left <= 1)
    step_len = torch.where(approach, distance.clamp_min(0.0) / steps_left, step_len)
    state.on_plan |= handover
    new_window = new_window & ~handover
    state.assigned[plan | handover] = assigned[plan | handover]
    valid[plan, 0] &= own_window(state.phase, feet[:, 0])[plan]
    valid[handover] = False                 # the plan's targets start next step
    if bool(new_window.any()):
        # Flat and platform rows continue from the stance foot's previous
        # target (an overshoot does not carry forward); the stair approach
        # counts down from the stance foot itself.
        stance_xy = centre[index, 1 - assigned, :2]
        continuing = (rows & ~approach & state.active & (state.window > 0)
                      & (state.feet[:, 0] == 1 - assigned))
        origin = rolling_origin(stance_xy, state.targets[:, 0, :2], continuing)
        fresh, fresh_feet = heading_targets(
            origin, assigned, yaw,
            command, state.period_s, NOMINAL_FOOT_SEPARATION_M, step_len,
        )
        if terrain_z is not None:
            fresh[..., 2] = terrain_z(fresh[..., 0])
        state.targets[new_window] = fresh[new_window]
        state.feet[new_window] = fresh_feet[new_window]
        state.window[new_window] += 1
        state.acquired[new_window] = False
        state.assigned[new_window] = assigned[new_window]
    state.active = walking
    rolling = rows & ~handover
    targets[rolling] = state.targets[rolling]
    feet[rolling] = state.feet[rolling]
    valid[rolling, 0] = walking[rolling] & ~state.acquired[rolling]
    valid[rolling, 1] = walking[rolling]
    return rolling, walking, schedule, handover


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
    # Body indices never change during a run; ContactSensor.body_names rebuilds
    # the list from PhysX prim paths on every call (py-spy, run 20261109:
    # 25% of the training thread when looked up every step).
    ids = getattr(env, "_g1_foot_body_ids", None)
    if ids is None:
        ids = ([list(robot.body_names).index(name) for name in FOOT_BODY_NAMES],
               [list(sensor.body_names).index(name) for name in FOOT_BODY_NAMES])
        env._g1_foot_body_ids = ids
    foot_ids, sensor_ids = ids
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
    # Flat tiles under the gait clock get a target on every step. The
    # target id is the clock window, so progress resets per target.
    clock_state = ensure_rolling_clock(env)
    rolling_rows = None
    if clock_state is not None:
        resets_now = (env.termination_manager.terminated
                      | env.termination_manager.time_outs)
        targets, feet, valid = targets.clone(), feet.clone(), valid.clone()
        approach = coordinator.runtime.approach_for(flat)
        rolling_rows, rolling_walking, clock_swing, handover = apply_rolling_clock(
            env, clock_state, centre=centre, targets=targets, feet=feet,
            valid=valid, resets=resets_now,
            first_plan_x=approach_target_x(
                course["first_riser_x"], course["tread_depth"], -approach),
            terrain_z=lambda x: stair_ground_height(
                x, course["first_riser_x"], course["tread_depth"],
                course["riser_height"], coordinator.runtime.step_count),
        )
        coordinator.runtime.target_index[rolling_rows] = clock_state.window[rolling_rows]
        # Hand-over: the stair plan starts at its first target, with the foot
        # the clock is about to swing.
        coordinator.runtime.target_index[handover] = 0
        coordinator.runtime.first_foot[handover] = clock_state.assigned[handover]
    env._g1_rolling_rows = rolling_rows
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
    if rolling_rows is not None:
        rolling_qualified = qualified_flat_mask(
            sole_centre=centre, sole_bounds=bounds,
            vertical_force=force_z, targets=targets,
            feet=feet, valid=valid,
            half_width=float(cfg.size[1]) / 2,
        )
        qualified = torch.where(rolling_rows[:, None], rolling_qualified, qualified)
        if not flat:
            strict = torch.where(rolling_rows[:, None], rolling_qualified, strict)
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
    if rolling_rows is not None:
        # Clock rewards come from the rolling clock (the phase the policy
        # observed) on all tiles; standing rows and finished stair plans
        # request double stance and still feet.
        # A foot that reached its target is neutral for the rest of its
        # window: lifting again earns nothing, and loading earns nothing
        # either (as stance it paid for dropping the weight onto the new
        # foot at once: peak body acceleration after a hit 7.9 -> 11.4 m/s2,
        # run 20261031_083022).
        scheduled = clock_swing & rolling_walking[:, None]
        clock_swing_f = (scheduled & ~clock_state.landed).to(force_z.dtype)
        clock_stance_f = (~scheduled).to(force_z.dtype)
        clock_stance, clock_motion = clock_contact_scores(
            force_z, centre, sole_velocity, targets, feet, valid,
            clock_swing_f, clock_stance_f,
        )
        keep = ~resets
        env._g1_footstep_stance_support = torch.where(
            keep, clock_stance, torch.zeros_like(clock_stance))
        env._g1_footstep_directed_swing = torch.where(
            keep, clock_motion, torch.zeros_like(clock_motion))
        # Uncommanded rotation: last full cycle's average yaw-rate error.
        from .footstep_clock import yaw_drift_cost
        drift = yaw_drift_cost(clock_state.cycle_yaw_error)
        env._g1_footstep_yaw_drift = torch.where(keep, drift, torch.zeros_like(drift))
        # Swing clearance: each swinging sole above the ground under it, as a
        # fraction of what it must clear (its target level, or the ground it
        # is over if higher) plus a margin. Flat tiles: riser 0, ground 0.
        from .footstep_rewards import swing_clearance, swing_ground
        # Patch 60: the ground is also taken 2 cm ahead of the toe, so a
        # toe about to reach the next riser is paid only for height above
        # that tread: up first, then forward. Run 20261110_085422
        # model_27200 at 10-12 cm: swings that scuffed moved the toe forward
        # faster than they rose (at 12 cm the sole was 1.4 cm below the next
        # tread top when the toe reached the riser; clean swings 1.7 cm above).
        ground = swing_ground(
            centre[..., 0], bounds[..., 1], course["first_riser_x"],
            course["tread_depth"], course["riser_height"],
            coordinator.runtime.step_count)
        own_slot = (feet[:, 0:1] == torch.arange(2, device=env.device)[None, :])
        goal = torch.maximum(
            ground, torch.where(own_slot, targets[:, 0, 2:3], targets[:, 1, 2:3]))
        cap = late_swing_cap(swing_progress(clock_state.phase))
        lift = swing_clearance(bounds[..., 4], ground, goal, clock_swing_f, cap)
        env._g1_footstep_swing_clearance = torch.where(keep, lift, torch.zeros_like(lift))
        # Off-beat steps on the stair plan: one foot per window, so one tread
        # per step. Extra steps were free and the policy kept its short flat
        # stride, putting both feet on every tread (run 20261031_112519).
        from .footstep_clock import off_beat_steps
        _, turn_foot = clock_schedule(clock_state.phase)
        on_turn = turn_foot[:, None] == torch.arange(2, device=env.device)[None, :]
        plan_rows = (riser_height_per_env(env) > 0) & clock_state.on_plan & keep
        off_beat, clock_state.liftoff_on_turn, clock_state.liftoff_x = off_beat_steps(
            clock_state.prev_supported, supported, clock_state.liftoff_on_turn,
            clock_state.liftoff_x, centre[..., 0], on_turn, plan_rows)
        # Landing impact on all tiles (before prev_supported moves on).
        sole_vz = sole_velocity[..., 2]
        impact = landing_impact(
            clock_state.prev_supported, supported, clock_state.prev_sole_vz,
            sole_vz, keep & clock_state.contact_armed)
        env._g1_footstep_landing_impact = impact
        # Patch 57: toe strikes and nose landings this episode (clean_climb),
        # read by the clean-climb curriculum at reset.
        from .clean_climb import NOSE_LANDING, TOE_STRIKE, nose_landings, toe_strikes
        stair_rows = (course["riser_height"] > 0) & keep
        push = torch.linalg.vector_norm(
            sensor.data.net_forces_w[:, sensor_ids, :2], dim=-1)
        clock_state.violations[:, TOE_STRIKE] |= toe_strikes(
            bounds, push, course["first_riser_x"], course["tread_depth"],
            coordinator.runtime.step_count, stair_rows)
        nose, clock_state.nose_window = nose_landings(
            supported & ~clock_state.prev_supported & clock_state.contact_armed[:, None],
            bounds, push, clock_state.nose_window, course["first_riser_x"],
            course["tread_depth"], coordinator.runtime.step_count, stair_rows)
        clock_state.violations[:, NOSE_LANDING] |= nose
        clock_state.prev_sole_vz = sole_vz.clone()
        clock_state.contact_armed = keep.clone()
        touchdown_before = (clock_state.prev_supported.clone(), plan_rows)  # patch 52, below
        clock_state.prev_supported = supported.clone()
        env._g1_footstep_off_beat = off_beat
        clock_state.off_beat_since_hit |= off_beat > 0
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

    # Platform targets: a landing at or past them also counts (the tread
    # mask accepts it; the distance check must too), so one overshoot on
    # the platform does not leave the plan unfinished (run 20261030_133621:
    # last target missed by 1 mm, then 15 cm past). Patch 59: the first
    # platform target too. Run 20261110_085422 model_27200 at 12 cm: 6/10
    # robots stood on the platform after landing 9-11 cm past it and 8-12 cm
    # wide (12.5-15.9 cm from it, radius 12.5 cm), the plan never finished
    # and the episode ended as a stall.
    #
    # Patch 60: on the stair plan the tread mask is the whole check along x
    # (tread edges, heel overhang, toe before the next riser, platform
    # overshoot), so the distance check keeps only a sideways limit of
    # STAIR_HIT_LATERAL_M. Run 20261110_085422 model_27200 at 12 cm: platform
    # landings that passed the tread mask 13.6 and 17.6 cm to the side were
    # missed by the 12.5 cm circle; the sideways offset on tread 2 grows
    # about 1 cm per cm of riser (10 cm at 12 cm). Rows under the rolling
    # clock keep the 12.5 cm circle.
    step_targets = targets
    xy_tolerance = 0.08 if flat else 0.125
    if not flat:
        plan = valid[:, 0].clone()
        if rolling_rows is not None:
            plan &= ~rolling_rows
        if bool(plan.any()):
            step_targets = targets.clone()
            foot_x = centre[rows, feet[:, 0].long(), 0]
            step_targets[plan, 0, 0] = foot_x[plan]
            xy_tolerance = torch.where(
                plan, torch.full_like(foot_x, STAIR_HIT_LATERAL_M),
                torch.full_like(foot_x, 0.125))
    advanced = coordinator.step(
        step_id=int(env.common_step_counter), dt=float(env.step_dt),
        targets=step_targets, feet=feet, valid=valid, sole_xyz=centre,
        supported=supported, qualified_tread=qualified,
        xy_tolerance=xy_tolerance,
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
    env._g1_footstep_acquisition_paid = advanced
    # Patch 53: stair tread hits pay by placement (1.0 on the planned centre,
    # 0.5 at the heel-overhang limit); flat, approach and rolling rows 1.0.
    quality = torch.ones(env.num_envs, device=env.device)
    if not flat:
        quality = tread_placement_quality(
            sole_bounds=bounds, targets=targets, feet=feet,
            first_riser_x=course['first_riser_x'],
            tread_depth=course['tread_depth'],
            riser_height=course['riser_height'],
            step_count=coordinator.runtime.step_count,
        )
        if rolling_rows is not None:
            quality = torch.where(rolling_rows, torch.ones_like(quality), quality)
    env._g1_footstep_acquisition_quality = quality
    env._g1_footstep_landing_audit["acquisition_quality"] = quality.clone()
    if rolling_rows is not None:
        # A stair hit after an off-beat step advances the plan but pays no
        # acquisition: step-to walking (both feet on every tread) earned the
        # same hits as alternating, only a step later (run 20261031_112519).
        plan_hit = advanced & clock_state.on_plan
        env._g1_footstep_acquisition_paid = advanced & ~(
            plan_hit & clock_state.off_beat_since_hit)
        from .clean_climb import EXTRA_STEP
        clock_state.violations[:, EXTRA_STEP] |= plan_hit & clock_state.off_beat_since_hit
        clock_state.off_beat_since_hit &= ~plan_hit
        # The first touchdown is the attempt (patch 52): the re-plant that
        # hits after a missed touchdown pays no acquisition, like a hit after
        # an off-beat step. Before, a 2-3 cm re-plant was not a step
        # (OFF_BEAT_MIN_MOVE_M) and paid in full: in run 20261106_122429
        # 8/10 4 cm climbs landed the first tread 1-2 cm short and shuffled
        # forward; all 100 plan hits there landed on the touchdown frame.
        from .footstep_clock import missed_touchdown
        prev_supported, plan_rows = touchdown_before
        clock_state.off_beat_since_hit |= missed_touchdown(
            prev_supported, supported, feet[:, 0], plan_rows, valid[:, 0], advanced)
        # One acquisition per clock window; the id stays the window.
        clock_state.acquired |= advanced & rolling_rows
        # The foot that reached its target ends its swing.
        hit = advanced[:, None] & (
            feet[:, 0:1] == torch.arange(2, device=env.device)[None, :])
        clock_state.landed |= hit
        # Plan complete: walk on like a flat tile, straight ahead at the
        # current forward command (the stair steering's side and turn
        # corrections are cleared).
        finished = (clock_state.on_plan & ~resets
                    & (coordinator.runtime.target_index
                       >= coordinator.runtime.plan_length(flat)))
        if bool(finished.any()):
            clock_state.on_plan[finished] = False
            clock_state.climbed[finished] = True
            clock_state.assigned[finished] = -1     # new rolling window next step
            # Force a progress reset on the first rolling target: its window
            # id can equal the finished plan's last index, whose best
            # distance was just set to infinity.
            env._g1_footstep_progress_target_index[finished] = -1
            env.command_manager.get_command("base_velocity")[finished, 1:] = 0.0
        # Stair rows mark the best distance infinite because their target
        # id changes next step and resets it. A rolling target keeps its id
        # until the clock hands over, so mark it reached instead (no further
        # progress credit, finite); the next window resets it.
        env._g1_footstep_best_distance[advanced & rolling_rows] = 0.0
        coordinator.runtime.target_index[rolling_rows] = clock_state.window[rolling_rows]
        command = env.command_manager.get_command("base_velocity")
        clock_state.advance(
            float(env.step_dt), rolling_walking,
            yaw_rate_error=data.root_ang_vel_b[:, 2] - command[:, 2],
        )
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
    steer_stair_command(env, coordinator.runtime, flat, resets,
                        course["tread_depth"], clock_state)
    if clock_state is None:
        # Without the clock a finished plan stands still; with it the row
        # walks on (above).
        stop_command_after_plan(env, coordinator.runtime, flat, resets)
    return advanced


def steer_stair_command(env, runtime, flat_ground, resets, tread_depth, clock_state):
    """Stair envs with an active plan (or still approaching it): lateral and
    yaw command steer back to the centre line facing the stairs
    (footstep_mixing.stair_steering), like an operator lining the robot up.
    With the gait clock, the forward command is the climbing speed, one
    tread per stair step: 2 x tread / stair cycle time, already on the
    approach. Without it, the sampled stair speed stays.

    Runs every policy step in training and evaluation (same hook), so a
    command resample is overwritten within one step. Flat tiles keep their
    joystick command.
    """
    riser = riser_height_per_env(env)
    stairs = (riser > 0) & ~resets
    plan_active = runtime.target_index < runtime.plan_length(flat_ground)
    if clock_state is not None:
        # Approach rows (no plan yet) walk at the same speed as the climb;
        # rows past the stairs follow their own command.
        plan_active = (~clock_state.on_plan | plan_active) & ~clock_state.climbed
    active = stairs & plan_active
    if bool(active.any()):
        xyz, yaw = course_root_pose(env)
        command = env.command_manager.get_command("base_velocity")
        if clock_state is not None:
            period = stair_period(riser, clock_state.period_scale, env.cfg.footstep_clock)
            command[active, 0] = 2.0 * tread_depth[active] / period[active]
        command[active, 1:3] = stair_steering(xyz[active, 1], yaw[active])
    return active


def stop_command_after_plan(env, runtime, flat_ground, resets):
    """Plan complete: zero the velocity command so the prior stands still.

    Replaces the removed clock stand mode. The command term may resample
    on its own timer; this runs every policy step, so at most one step can
    observe the resampled value before it is zeroed again.
    """
    done = (runtime.target_index >= runtime.plan_length(flat_ground)) & ~resets
    rolling = getattr(env, "_g1_rolling_rows", None)
    if rolling is not None:
        # Clock-driven flat tiles never complete: they follow the joystick.
        done = done & ~rolling
    if bool(done.any()):
        command = env.command_manager.get_command("base_velocity")
        command[done] = 0.0
    return done
