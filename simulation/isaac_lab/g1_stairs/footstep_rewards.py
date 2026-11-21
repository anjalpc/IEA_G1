"""Paper-inspired, tensor-only footstep objectives; no old stair reward dependencies.

Learning Bipedal Walking on Planned Footsteps:
step acquisition/progress, foot force/velocity phase, posture/height.
These are diagnostic raw scores, NOT registered reward terms or calibrated weights.
"""
import torch


def footstep_target_score(sole_xyz, targets, feet, valid, acquired, *, sigma_m=0.25):
    """[N] positive ONLY for externally qualified acquisition of assigned foot."""
    n = sole_xyz.shape[0]
    if sole_xyz.shape != (n, 2, 3) or targets.shape != (n, 2, 4):
        raise ValueError("invalid footstep geometry")
    if feet.shape != (n, 2) or valid.shape != (n, 2) or acquired.shape != (n,):
        raise ValueError("invalid target state")
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    idx = feet[:, 0].long()
    assigned = sole_xyz[torch.arange(n,device=sole_xyz.device),idx]
    error = torch.linalg.vector_norm(assigned-targets[:,0,:3],dim=-1)
    return torch.exp(-error/sigma_m) * (acquired & valid[:,0]).to(error.dtype)


def clock_contact_scores(force_z, sole_xyz, sole_vel_xyz, targets, feet, valid,
                         swing, stance, *, threshold_n=30.0, speed_scale=0.3):
    """Signed periodic GRF and foot-speed scores for BOTH legs.

    Positive GRF: scheduled stance; negative GRF: scheduled swing.
    Positive speed: scheduled swing; negative speed: scheduled stance.
    Target assignment does not determine which leg receives gait credit.
    Swing/stance masks fully define the mode: after the plan is complete the
    clock requests double stance (stand still, both feet loaded), as in the
    reference STANDING mode. Reset rows are zeroed by the caller.
    """
    n = force_z.shape[0]
    if (force_z.shape != (n, 2) or sole_xyz.shape != (n, 2, 3)
            or sole_vel_xyz.shape != (n, 2, 3) or targets.shape != (n, 2, 4)
            or feet.shape != (n, 2) or valid.shape != (n, 2)
            or swing.shape != (n, 2) or stance.shape != (n, 2)):
        raise ValueError("invalid contact/footstep tensor shapes")
    if threshold_n <= 0 or speed_scale <= 0:
        raise ValueError("invalid force or speed scale")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")

    # Bounded physical magnitudes. Both legs contribute independently.
    force = (force_z.clamp_min(0) / threshold_n).clamp(max=1)
    speed = (
        torch.linalg.vector_norm(sole_vel_xyz, dim=-1) / speed_scale
    ).clamp(max=1)

    force_score = ((stance - swing) * force).sum(dim=1) / 2
    speed_score = ((swing - stance) * speed).sum(dim=1) / 2
    return force_score, speed_score


# Stage 19 intentionally does not change PPO reward registration, losses, or training.

"""Isolation contract for a NEW paper-inspired task (not legacy reward retuning).

Registration is intentionally blocked until physics touchdown and reward-activation
diagnostics are verified. No existing 76-term rewards may be inherited implicitly.
"""
REQUIRED_NEW_FAMILIES = (
    "footstep_acquisition",
    "footstep_target_progress",
    "clock_stance_support",
    "clock_swing_motion",
    "body_orientation_height",
    "upper_body_alignment",
)

def assert_paper_only_rewards(active_names):
    """Fail closed if any active reward has not been explicitly redesigned."""
    active = tuple(active_names)
    unexpected = sorted(set(active) - set(REQUIRED_NEW_FAMILIES))
    if unexpected:
        raise RuntimeError("Legacy or unreviewed reward terms enabled: "+", ".join(unexpected))
    return True


def _per_event(env, value):
    """Cancel the reward manager's dt so weight == reward per event/metre unit.

    Same convention as geometry_failure_termination_event. Use only for
    one-step events and non-recycling increments, never for per-step rates.
    """
    return value / float(env.step_dt)


def cached_acquisition(env):
    """Qualified assigned-foot touchdown; 1.0 per acquisition after dt scaling.

    Pays ``_g1_footstep_acquisition_paid`` when the hook sets it (stair hits
    after an off-beat step are not paid), else every advance."""
    value = getattr(env, "_g1_footstep_acquisition_paid", None)
    if value is None:
        value = getattr(env, "_g1_footstep_last_advanced", None)
    if value is None:
        raise RuntimeError("footstep acquisition reward evaluated before physics hook")
    if value.shape != (env.num_envs,):
        raise RuntimeError("invalid cached acquisition shape")
    value = value.float()
    quality = getattr(env, "_g1_footstep_acquisition_quality", None)
    if quality is not None:
        # Patch 53: stair tread hits pay by placement (footstep_tread).
        value = value * quality
    return _per_event(env, value)


def cached_directed_swing(env):
    """Read target-directed swing measurement; never update coordinator."""
    value = getattr(env, "_g1_footstep_directed_swing", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing post-physics swing measurement")
    return value


def new_target_progress(previous_distance, current_distance, best_distance, valid, *, scale_m=0.10):
    """One-way improvement against best distance reached for the CURRENT target.

    Returns (score, updated_best).  Distance increases never pay, and
    retracing previously rewarded ground cannot pay again. The caller must
    reset best distance only when the target changes or an episode resets.
    """
    if (previous_distance.shape != current_distance.shape
            or previous_distance.shape != best_distance.shape
            or previous_distance.shape != valid.shape):
        raise ValueError("progress tensors have different shapes")
    if scale_m <= 0:
        raise ValueError("scale_m must be positive")
    if not torch.isfinite(previous_distance).all() or not torch.isfinite(current_distance).all():
        raise ValueError("nonfinite target distance")
    if not torch.isfinite(best_distance).all():
        raise ValueError("nonfinite best distance")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    new_best = torch.minimum(best_distance, current_distance)
    improvement = (best_distance - new_best).clamp(min=0)
    reward = torch.where(valid, improvement / scale_m, torch.zeros_like(improvement))
    return reward, torch.where(valid, new_best, best_distance)


def cached_net_progress(env):
    """Non-recycling progress; 1.0 per 0.10 m of new best distance after dt scaling."""
    value = getattr(env, "_g1_footstep_net_progress", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached footstep net progress")
    return _per_event(env, value)


def cached_yaw_drift(env):
    """Uncommanded rotation cost >= 0 (cycle-averaged yaw-rate error, Huber)."""
    value = getattr(env, "_g1_footstep_yaw_drift", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached yaw drift")
    return value


def cached_stance_support(env):
    """Observed support during assigned gait-clock stance."""
    value = getattr(env, "_g1_footstep_stance_support", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached stance support")
    return value


# Swing clearance (gait-clock stages). Full credit when the swinging sole
# is this far above what it has to clear. Over a higher target level: 5 cm
# above it (at 2 cm risers with the prior's ~1.5 cm lift the toe jammed,
# run 20261030_092747). On level ground: 2 cm, the Unitree prior's own lift
# (1.2-1.8 cm at 0.25-0.6 m/s, model_0 sweeps 20261028); 5 cm there made
# the residual fight the prior on every flat step (run 20261031_083022:
# flat lift 5-6.7 cm, pitch rate sd 26 vs 7 deg/s for the prior).
SWING_CLEARANCE_MARGIN_M = 0.05
SWING_CLEARANCE_FLAT_MARGIN_M = 0.02
# Swing clearance ground is also read this far ahead of the modelled toe:
# the real foot touches a riser 1.5 cm before it (TOE_CONTACT_REACH_M), plus
# 0.5 cm so the credit drops just before contact, not at it.
SWING_TOE_LOOKAHEAD_M = 0.02


def stair_ground_height(x, first_riser_x, tread_depth, riser_height, step_count):
    """[N,K] terrain height at course x: 0 before the first riser, tread k
    at k risers, the platform at step_count risers. Zero on flat tiles."""
    level = torch.floor((x - first_riser_x[:, None]) / tread_depth[:, None]) + 1.0
    return level.clamp(0.0, float(step_count)) * riser_height[:, None]


def swing_ground(centre_x, toe_x, first_riser_x, tread_depth, riser_height, step_count,
                 lookahead_m=SWING_TOE_LOOKAHEAD_M):
    """[N,K] ground a swing sole is measured from: the higher of the terrain
    under its centre and ``lookahead_m`` ahead of its toe. A toe about to
    reach the next riser is paid only for height above that tread (patch 60:
    up first, then forward)."""
    return torch.maximum(
        stair_ground_height(centre_x, first_riser_x, tread_depth, riser_height, step_count),
        stair_ground_height(toe_x + lookahead_m, first_riser_x, tread_depth,
                            riser_height, step_count))

def swing_clearance(sole_bottom, ground, goal, swing, cap=None,
                    *, margin_m=SWING_CLEARANCE_MARGIN_M,
                    flat_margin_m=SWING_CLEARANCE_FLAT_MARGIN_M):
    """[N] in [0, 1]: lift of each scheduled swing sole above the ground
    under it, as a fraction of the lift needed to clear ``goal`` + margin
    (``flat_margin_m`` when the goal is the ground itself).

    All inputs [N,2]; ``swing`` is the clock's swing schedule (0/1). Lift is
    measured from the ground, not the goal, so a high riser still gives a
    signal from the first centimetre. A loaded or still foot gets nothing:
    only the clock's swing foot counts, and higher than the margin pays no
    more. ``cap`` [N,2] (optional) limits the credit per foot (late in the
    window: a gradual descent pays as much as hovering).
    """
    if not (sole_bottom.shape == ground.shape == goal.shape == swing.shape):
        raise ValueError("swing clearance tensors have different shapes")
    if margin_m <= 0 or flat_margin_m <= 0:
        raise ValueError("margins must be positive")
    rise = (goal - ground).clamp_min(0.0)
    need = rise + torch.where(rise > 1e-3, torch.full_like(rise, margin_m),
                              torch.full_like(rise, flat_margin_m))
    fraction = ((sole_bottom - ground) / need).clamp(0.0, 1.0)
    if cap is not None:
        if cap.shape != fraction.shape:
            raise ValueError("swing clearance cap has a different shape")
        fraction = torch.minimum(fraction, cap)
    return (fraction * swing).sum(dim=-1)


def foot_stumble(env, sensor_cfg):
    """Unitree's feet_stumble (unitree_rl_lab mdp): 1 per step while a foot's
    horizontal contact force exceeds 4x its vertical force (a foot pushing
    into a vertical face). Event scaled: weight = cost per contact step.

    Run 20261031_083022 (model_2499): active in 1.2% of stair steps (swing
    toes driven into riser faces, up to 1.8 kN), never on flat ground or for
    the Unitree prior.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    hit = (horizontal > 4.0 * forces[..., 2].abs()) & (horizontal > 1.0)
    return _per_event(env, hit.any(dim=-1).to(forces.dtype))


def cached_off_beat_step(env):
    """Stair-plan steps lifted off outside their foot's turn; 1.0 per step
    after dt scaling."""
    value = getattr(env, "_g1_footstep_off_beat", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached off-beat steps")
    return _per_event(env, value)


def body_rate_l2(env, command_name="base_velocity"):
    """Pelvis rate cost [N]: roll^2 + pitch^2 + (yaw - commanded yaw)^2 in
    rad^2/s^2 (body frame).

    Isaac Lab's ang_vel_xy_l2 (Unitree's base_angular_velocity) leaves yaw
    out, and uncommanded_yaw averages over a gait cycle, so a step-by-step
    pelvis twist cost nothing: run 20261102_090334 model_5600 twisted
    3.4 deg sd (yaw rate sd 35 deg/s) vs Unitree 0.2 deg (2 deg/s). Yaw is
    measured against the command, so commanded turning stays free.
    """
    w = env.scene["robot"].data.root_ang_vel_b
    yaw_cmd = env.command_manager.get_command(command_name)[:, 2]
    return w[:, 0].square() + w[:, 1].square() + (w[:, 2] - yaw_cmd).square()


def cached_landing_impact(env):
    """Touchdown impact cost (sum over landing feet of the squared excess
    downward speed); per event after dt scaling."""
    value = getattr(env, "_g1_footstep_landing_impact", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached landing impact")
    return _per_event(env, value)


def cached_swing_clearance(env):
    """Clock swing-foot clearance in [0, 1], per second."""
    value = getattr(env, "_g1_footstep_swing_clearance", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached swing clearance")
    return value

# Patch 64: stance knees bent beyond this off the stair plan are charged.
# Scheduled-stance knee, flat, model_35200 (natural gait): median 0.20,
# p99 0.36 rad; model_81200: median 0.51 rad.
STANCE_KNEE_LIMIT_RAD = 0.35


def stance_knee_bend(knee_pos, stance, rows, limit=STANCE_KNEE_LIMIT_RAD):
    """Crouch measure [N]: knee flexion beyond ``limit`` [rad], summed over
    the feet the clock schedules as stance, on ``rows`` only.

    Patch 64. The flat, approach and platform gait sank into a crouch as the
    stair curriculum went up (flat stance knee 0.20 rad at model_35200 ->
    0.58 at model_81200, step 16 -> 13.4 cm, 0.38 -> 0.32 m/s at 0.35
    command, stance knee torque 7.5 -> 11.9 Nm). residual_magnitude cannot
    hold the posture (patch 63): the teacher follows the crouched state, so
    executed - teacher stays small. This measures the posture itself. Mean
    per step on flat: 0.001 rad at model_35200, 0.19 rad at model_81200.
    Scheduled stance, not contact, so unloading a foot does not avoid it.
    Not charged on the stair plan, where deep stance flexion is the climb.
    """
    excess = torch.clamp(knee_pos - limit, min=0.0) * stance.to(knee_pos.dtype)
    return torch.where(rows, excess.sum(dim=-1), torch.zeros_like(excess[:, 0]))


def cached_stance_knee_bend(env):
    """Stance knee flexion beyond the limit off the stair plan, per second."""
    value = getattr(env, "_g1_footstep_stance_knee_bend", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached stance knee bend")
    return value


# Patch 71: pelvis pitch free up to this on level ground. Unitree teacher
# alone on flat: -0.1 deg; model_94100 6.8, 91700 9.8, 96100 13.4 deg.
LEVEL_PITCH_DEADBAND_RAD = 0.0524  # 3 deg


def level_pitch(projected_gravity_b, rows, deadband=LEVEL_PITCH_DEADBAND_RAD):
    """Pelvis pitch beyond ``deadband`` [rad], either sign, on ``rows`` [N].

    Patch 71. Pitch = asin(gravity x in the pelvis frame), + = forward lean.
    Linear, unlike upright (flat_orientation_l2, quadratic): at 13 deg the
    upright term charged 0.16/s per extra degree while the lean bought about
    0.24/s per degree on the paid platform (model_91700 -> 96100: swing-leg
    residual 1.80 -> 0.85 action units, residual_magnitude +0.33/s, target
    hits +0.32/s, upright -0.35/s; doc g1-p70-lean-mechanism). The lean is
    made at the stance hip and adds leg reach in place of residual. Level
    ground only (flat tiles, platform after the climb): the stair approach
    keeps its forward pitch (+2.2 deg before the 20 cm step at 91700, 10/10).
    """
    g_x = projected_gravity_b[:, 0].clamp(-1.0, 1.0)
    excess = torch.clamp(torch.asin(g_x).abs() - deadband, min=0.0)
    return torch.where(rows, excess, torch.zeros_like(excess))


def cached_level_pitch(env):
    """Pelvis pitch beyond the deadband on level ground, per second."""
    value = getattr(env, "_g1_footstep_level_pitch", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached level pitch")
    return value


# Patch 73: stance foot yaw against the pelvis free up to this on level
# ground. Level stance toe-out p90: model_91700 4.3 deg, 94100 mean -1.3;
# treads 3.8-5.1 deg at every checkpoint since 91700.
LEVEL_FOOT_YAW_DEADBAND_RAD = 0.0873  # 5 deg


def _yaw_wxyz(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def level_foot_yaw(foot_quat_w, root_quat_w, supported, rows,
                   deadband=LEVEL_FOOT_YAW_DEADBAND_RAD):
    """Stance foot yaw against the pelvis beyond ``deadband`` [rad], either
    sign, summed over the feet in contact, on ``rows`` [N].

    Patch 73. Level-ground toe-out grew 2.8 deg (91700) -> 6.3 (103600) ->
    10.4 (108600) -> 11.3 (114100) at the hip yaw, while tread toe-out held
    3.8-5.1 deg and did not separate strict from failed climbs (doc
    g1-stair-clock-resume-run-20261120-100022-findings). Flat excess per
    frame: 0.001 rad at 91700, 0.140 at 114100. Same rows as level_pitch.
    """
    rel = _yaw_wxyz(foot_quat_w) - _yaw_wxyz(root_quat_w)[:, None]
    rel = torch.atan2(torch.sin(rel), torch.cos(rel))
    excess = torch.clamp(rel.abs() - deadband, min=0.0) * supported.to(rel.dtype)
    return torch.where(rows, excess.sum(dim=-1), torch.zeros_like(excess[:, 0]))


def cached_level_foot_yaw(env):
    """Stance foot yaw beyond the deadband on level ground, per second."""
    value = getattr(env, "_g1_footstep_level_foot_yaw", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached level foot yaw")
    return value


# Sole clearance above the riser edge that earns full lift credit. The
# Unitree prior's swing sole clearance is ~1.5-2 cm, so a lift capped at
# exactly the riser height still stubs the edge.
TREAD_LIFT_MARGIN_M = 0.03


def tread_lift(sole_bottom_z, target_z, riser_height, plan_index, step_count,
               *, margin_m=TREAD_LIFT_MARGIN_M):
    """Assigned-sole lift above the level it climbs from, for tread targets.

    Returns (lift [N] in [0, goal], goal [N]) with goal = riser + margin.
    Ground-approach and platform targets (no riser to cross) return 0, and
    lift is measured from the tread below, not from the ground.
    """
    if not (sole_bottom_z.shape == target_z.shape == riser_height.shape
            == plan_index.shape):
        raise ValueError("tread lift tensors have different shapes")
    if margin_m <= 0:
        raise ValueError("margin_m must be positive")
    is_tread = (plan_index >= 0) & (plan_index < step_count) & (riser_height > 0)
    goal = riser_height + margin_m
    lift = torch.minimum((sole_bottom_z - (target_z - riser_height)).clamp_min(0), goal)
    return torch.where(is_tread, lift, torch.zeros_like(lift)), goal


def cached_clearance_progress(env):
    """New peak tread lift (riser + margin); at most 1.0 per tread after dt scaling."""
    value = getattr(env, "_g1_footstep_clearance_progress", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached footstep clearance")
    return _per_event(env, value)


def flat_feet_clearance(env, asset_cfg, target_height, std, tanh_mult):
    """Unitree's foot_clearance_reward, flat tiles only.

    It targets an absolute world height. It must not depend on position
    within a stair tile: it is ~1 whenever the feet are still or near the
    target height, so paying it only before the riser would reward stopping
    there. Stair envs get 0; their lift comes from footstep clearance.
    """
    from unitree_rl_lab.tasks.locomotion import mdp
    from .terrain_state import riser_height_per_env

    value = mdp.foot_clearance_reward(env, asset_cfg, target_height, std, tanh_mult)
    return value * (riser_height_per_env(env) <= 0).to(value.dtype)


def cached_stall_event(env):
    """Stair plan stall (target time limit); 1.0 per stall after dt scaling.

    Added on top of the generic failure penalty. Note: under gamma 0.99
    a stall penalty 4 s ahead is heavily discounted, so it only discourages
    stalling near the limit; it does not make an unclimbable riser worth
    attempting.
    """
    value = getattr(env, "_g1_footstep_stall_event", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached stall event")
    return _per_event(env, value)


def cached_failure_event(env):
    """Non-timeout termination; 1.0 per failure after dt scaling."""
    value = getattr(env, "_g1_footstep_failure_event", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached failure event")
    return _per_event(env, value)


def residual_magnitude(env):
    """Squared size of the correction applied on top of the frozen Unitree
    action, in raw action units (x 0.25 rad), summed over the 29 joints.

    Executed minus teacher action, i.e. what actually reached the robot
    after clipping. With a negative weight the correction stays near zero
    wherever it earns nothing (flat walking stays Unitree's) and is spent
    only where it pays (lift and placement at the stairs).
    """
    teacher = getattr(env, "_g1_balance_teacher_action", None)
    executed = getattr(env, "_g1_balance_executed_action", None)
    if teacher is None or executed is None:
        raise RuntimeError(
            "residual_magnitude needs the frozen-teacher residual wrapper"
        )
    if teacher.shape != executed.shape or teacher.shape[0] != env.num_envs:
        raise RuntimeError("invalid teacher/executed action shapes")
    return torch.sum(torch.square(executed - teacher), dim=1)
