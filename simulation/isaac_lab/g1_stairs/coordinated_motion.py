"""Opt-in C6 coordinated motion shaping. Robot actions are never overwritten."""
import math
import torch
from isaaclab.managers import ManagerTermBase
from .motion_reference import (
    sole_state, swing_reference, bounded_square, normalized_huber, rotate,
    target_support_mask, transition_quality_escrow_update,
    strict_tread_landing_mask,
)
from .contracts import G1_SOLE_SPHERE_CENTRES_M, G1_SOLE_SPHERE_RADIUS_M
from .terrain_state import stair_cfg, riser_height_per_env


class CoordinatedMotion(ManagerTermBase):
    """Explicit swing/stance/foothold objective tied to canonical outcomes."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, device = env.num_envs, env.device
        self.target = torch.full((n,), -1, dtype=torch.long, device=device)
        self.preferred = torch.full_like(self.target, -1)
        self.progress = torch.zeros(n, device=device)
        self.start = torch.zeros(n, 3, device=device)
        self.landing = torch.zeros_like(self.start)
        self.lane = torch.zeros(n, 2, device=device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=device)
        self.quality_sum = torch.zeros(n, device=device)
        self.quality_count = torch.zeros(n, device=device)
        self.previous_reference_velocity = torch.zeros(n, 3, device=device)
        self.last_step = -1
        self.trace = {}
        env._g1_coordinated_motion = self
        names = list(env.scene['robot'].joint_names)
        self.arm_ids = [i for i, name in enumerate(names)
                        if any(x in name for x in ('shoulder', 'elbow', 'wrist'))]
        if len(self.arm_ids) != 14:
            raise ValueError('C6 requires the 14 G1 arm joints')
        self.pitch_ids = [names.index(s+'_shoulder_pitch_joint') for s in ('left', 'right')]

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.target[ids] = -1
        self.preferred[ids] = -1
        self.progress[ids] = 0
        self.start[ids] = 0
        self.landing[ids] = 0
        self.lane[ids] = 0
        self.initialized[ids] = False
        self.quality_sum[ids] = 0
        self.quality_count[ids] = 0
        self.previous_reference_velocity[ids] = 0
        if hasattr(self, 'total'):
            self.total[ids] = 0
        for value in self.trace.values():
            value[ids] = 0
        for name in ('reference_target', 'reference_foot'):
            if name in self.trace:
                self.trace[name][ids] = -1

    def _landing_target(self, cfg, height, target, lane, dtype, device):
        land = torch.zeros((target.shape[0], 3), dtype=dtype, device=device)
        neutral_span = (max(p[0] for p in G1_SOLE_SPHERE_CENTRES_M)
                        - min(p[0] for p in G1_SOLE_SPHERE_CENTRES_M)
                        + 2*G1_SOLE_SPHERE_RADIUS_M)
        free = float(cfg.tread_depth) - neutral_span
        if free < 0.05 - 1e-6:
            raise ValueError('coordinated foothold needs at least 2.5 cm sole margin per edge')
        land[:, 0] = (float(cfg.start_clearance)
                      + target.to(dtype)*float(cfg.tread_depth)
                      + 0.5*float(cfg.tread_depth))
        land[:, 1] = lane
        land[:, 2] = (target+1).to(dtype)*height
        return land

    def __call__(self, env, swing_weight=.35, stance_weight=.15,
                 prelift_weight=.20, placement_weight=.25, transfer_weight=.15,
                 lateral_weight=.10, arm_weight=.15, finish_weight=.15,
                 clearance_m=.03, position_scale_m=.06,
                 arm_amplitude_rad=.20, direct_quality_weight=0.0,
                 motion_miss_weight=.25, ascent_only_auxiliary=False, adaptive_timing=False):
        values = (swing_weight, stance_weight, prelift_weight, placement_weight, transfer_weight,
                  lateral_weight, arm_weight, finish_weight, clearance_m,
                  position_scale_m, arm_amplitude_rad, direct_quality_weight,
                  motion_miss_weight)
        if any(not math.isfinite(x) or x < 0 for x in values) or position_scale_m <= 0:
            raise ValueError('C6 parameters must be finite/nonnegative; scale positive')
        runtime = getattr(env, '_g1_v04_acquisition', None)
        step = int(env.common_step_counter)
        if runtime is None or runtime.last_step != step:
            raise RuntimeError('C6 must run after acquisition rewards')
        if self.last_step == step:
            return self.total
        self.last_step = step
        self._env = env
        cfg = stair_cfg(env)
        data = runtime.robot.data
        foot = data.body_pos_w[:, runtime.robot_foot_ids] - env.scene.env_origins[:, None]
        quat = data.body_quat_w[:, runtime.robot_foot_ids]
        centre, bounds, surface_velocity = sole_state(
            foot, quat, data.body_lin_vel_w[:, runtime.robot_foot_ids],
            data.body_ang_vel_w[:, runtime.robot_foot_ids],
        )
        force = runtime.sensor.data.net_forces_w[:, runtime.sensor_foot_ids, 2].clamp_min(0)
        self.lane.copy_(torch.where(self.initialized[:, None], self.lane, centre[..., 1]))
        self.initialized.fill_(True)
        height = riser_height_per_env(env)
        rows = torch.arange(env.num_envs, device=env.device)
        gait = env.action_manager.get_term('gait_frequency')

        # C6 first-step guidance: once the gait runtime observes a real unloaded
        # approach foot near the first riser, install target 0. No canonical
        # sequence credit is available for this bootstrap transition.
        if adaptive_timing:
            approach_foot = gait.approach_preferred_foot

            # gait_runtime may invalidate a provisional first-step reference
            # after that foot makes an ordinary pre-riser approach landing.
            # Drop the matching motion reference before selecting the next swing.
            invalidate_first = (
                (self.target == 0)
                & (runtime.target_step[:, 0] == 0)
                & (approach_foot < 0)
            )
            self.target.copy_(torch.where(
                invalidate_first,
                torch.full_like(self.target, -1),
                self.target,
            ))
            self.preferred.copy_(torch.where(
                invalidate_first,
                torch.full_like(self.preferred, -1),
                self.preferred,
            ))
            self.progress.copy_(torch.where(
                invalidate_first,
                torch.zeros_like(self.progress),
                self.progress,
            ))
            self.previous_reference_velocity.copy_(torch.where(
                invalidate_first[:, None],
                torch.zeros_like(self.previous_reference_velocity),
                self.previous_reference_velocity,
            ))

            install_first = ((self.target < 0) & (runtime.target_step[:, 0] == 0)
                             & (approach_foot >= 0) & (height > 1e-8))
            first_idx = approach_foot.clamp(0, 1)
            first_start = centre[rows, first_idx]
            first_target = torch.zeros_like(self.target)
            first_land = self._landing_target(
                cfg, height, first_target, self.lane[rows, first_idx],
                foot.dtype, foot.device)
            self.start.copy_(torch.where(install_first[:, None], first_start, self.start))
            self.landing.copy_(torch.where(install_first[:, None], first_land, self.landing))
            self.target.copy_(torch.where(install_first, first_target, self.target))
            self.preferred.copy_(torch.where(install_first, approach_foot, self.preferred))
            self.progress.copy_(torch.where(install_first, torch.zeros_like(self.progress), self.progress))

        idx = self.preferred.clamp(0, 1)
        actual = centre[rows, idx]
        selected = ((self.target >= 0) & (self.target < int(cfg.step_count))
                    & (self.preferred >= 0) & (height > 1e-8))
        expired = gait.reference_expired if adaptive_timing else torch.zeros_like(selected)
        swing_started = gait.swing_started if adaptive_timing else selected
        valid = selected & ~expired
        tracking_active = valid & swing_started

        # Before actual liftoff, keep updating the selected foot's start state;
        # next-foot selection and physical swing start are intentionally separate.
        self.start.copy_(torch.where((valid & ~swing_started)[:, None], actual, self.start))

        start_phase = torch.where(idx == 0, .5, 0.)
        before_progress = (2*(gait.reward_phase-start_phase)).clamp(0, 1)
        self.progress.copy_(torch.where(
            tracking_active,
            (before_progress + 2*float(env.step_dt)*gait.processed_actions[:, 0]).clamp(0, 1),
            torch.zeros_like(self.progress),
        ))
        reference = swing_reference(self.start, self.landing, self.progress, clearance_m)
        previous_reference = swing_reference(self.start, self.landing, before_progress, clearance_m)
        reference_velocity = torch.where(
            tracking_active[:, None],
            (reference-previous_reference)/float(env.step_dt),
            torch.zeros_like(reference),
        )
        reference_acceleration = torch.where(
            tracking_active[:, None],
            (reference_velocity-self.previous_reference_velocity)/float(env.step_dt),
            torch.zeros_like(reference_velocity),
        )
        self.previous_reference_velocity.copy_(reference_velocity)
        error = actual-reference
        swing = normalized_huber(error, position_scale_m).mean(-1) * tracking_active

        support_idx = 1-idx
        support_force = force[rows, support_idx]
        support_slip = surface_velocity[rows, support_idx, :, :2].norm(dim=-1).amax(-1)
        prelift_active = valid & ~swing_started
        prelift = (
            (force[rows, idx] / 30.0).clamp(0, 1)
            + (1.0 - support_force / 30.0).clamp(0, 1)
        ) * 0.5 * prelift_active
        swing_contact = (((self.progress > .10) & (self.progress < .90)).to(foot.dtype)
                         * (force[rows, idx]/30.).clamp(0, 1))
        stance = (bounded_square(support_slip, .10)
                  + (1-support_force/30.).clamp(0, 1) + swing_contact)/3*tracking_active

        tread_start = float(cfg.start_clearance) + self.target.to(foot.dtype)*float(cfg.tread_depth)
        tread_end = tread_start + float(cfg.tread_depth)
        tread_top = (self.target+1).to(foot.dtype)*height
        selected_bounds = bounds[rows, idx]
        margins = torch.stack((selected_bounds[:, 0]-tread_start,
                               tread_end-selected_bounds[:, 1],
                               selected_bounds[:, 2]+float(cfg.size[1])/2,
                               float(cfg.size[1])/2-selected_bounds[:, 3]), -1)
        # C7 first-step landing guidance must follow physical progress, not
        # wait for the nominal swing clock to reach 65%.  This keeps the
        # existing swing/clearance reference unchanged while progressively
        # exposing placement, orientation, and speed costs as the committed
        # foot closes horizontal distance to its tread target.
        phase_landing_gate = ((self.progress-.65)/.35).clamp(0, 1)*tracking_active
        horizontal_remaining_m = (
            actual[:, :2] - self.landing[:, :2]
        ).norm(dim=-1)
        horizontal_span_m = (
            self.start[:, :2] - self.landing[:, :2]
        ).norm(dim=-1).clamp_min(1e-6)
        spatial_landing_progress = (
            1.0 - horizontal_remaining_m / horizontal_span_m
        ).clamp(0, 1)

        committed_first = (
            tracking_active
            & (self.target == 0)
            & gait.approach_committed
        ) if adaptive_timing else torch.zeros_like(tracking_active)

        landing_gate = torch.where(
            committed_first,
            spatial_landing_progress,
            phase_landing_gate,
        )

        edge_cost = bounded_square((.025-margins).clamp_min(0), .025).mean(-1)
        centre_error_xy = (actual[:, :2] - self.landing[:, :2]).norm(dim=-1)
        centre_cost = bounded_square(centre_error_xy, position_scale_m)
        up = rotate(quat, foot.new_tensor([0., 0., 1.]).expand_as(foot))
        tilt = torch.acos(up[..., 2].clamp(-1, 1))
        forward = rotate(quat, foot.new_tensor([1., 0., 0.]).expand_as(foot))
        yaw = torch.atan2(forward[..., 1], forward[..., 0])
        surface_speed = surface_velocity.norm(dim=-1).amax(-1)
        placement_raw = (centre_cost + edge_cost
                         + bounded_square(tilt[rows, idx], .15)
                         + bounded_square(yaw[rows, idx], .20)
                         + bounded_square(surface_speed[rows, idx], .20))/5
        placement = landing_gate*placement_raw

        load_ratio = force / force.sum(-1, keepdim=True).clamp_min(1.)
        episode_runtime = getattr(env, '_g1_geometry_episode', None)
        strict_gait = getattr(episode_runtime, 'strict_gait', None)
        if strict_gait is not None:
            on_target_support = strict_tread_landing_mask(
                selected_bounds, force[rows, idx], surface_speed[rows, idx],
                tilt[rows, idx], yaw[rows, idx],
                tread_start=tread_start, tread_end=tread_end, tread_top=tread_top,
                lane_target=self.lane[rows, idx],
                terrain_half_width=float(cfg.size[1]) / 2,
                riser_height=height, tread_margin_m=strict_gait.TREAD_MARGIN_M,
                center_tolerance_m=.010,
                lane_tolerance_m=strict_gait.LANE_TOLERANCE_M,
                min_force_n=strict_gait.SUPPORT_FORCE_N,
                max_speed_mps=strict_gait.TOUCHDOWN_SPEED_MPS,
                max_tilt_rad=strict_gait.MAX_TILT_RAD,
                max_yaw_rad=strict_gait.MAX_YAW_RAD,
            ) & valid
        else:
            on_target_support = target_support_mask(
                selected_bounds, force[rows, idx], tread_start=tread_start,
                tread_end=tread_end, tread_top=tread_top,
                terrain_half_width=float(cfg.size[1])/2, min_force_n=30.,
                height_tolerance_m=.025,
            ) & valid
        load_mechanics = ((1-load_ratio[rows, idx]/.45).clamp(0, 1)
                          + (1-force[rows, idx]/30.).clamp(0, 1))*.5
        # Loading the old tread or another surface can never look like a good
        # support transfer, even if force magnitude is high.
        load_cost = torch.where(on_target_support, load_mechanics, torch.ones_like(load_mechanics))
        transfer = tracking_active*(self.progress >= .80)*load_cost

        lane_error = (centre[..., 1]-self.lane).abs()
        lateral = (bounded_square((lane_error-.04).clamp_min(0), .05)
                   * (force < 30)).mean(-1)
        if ascent_only_auxiliary:
            lateral = lateral * selected
        platform_x = float(cfg.start_clearance)+(int(cfg.step_count)-1)*float(cfg.tread_depth)
        root = data.root_pos_w-env.scene.env_origins
        command = env.command_manager.get_command('base_velocity')
        finish_gate = ((runtime.target_step[:, 0] >= int(cfg.step_count))
                       & (root[:, 0] >= platform_x)).to(foot.dtype)
        braking = finish_gate*(1-command[:, 0].abs()/.35).clamp(0, 1)
        arm_reference = data.default_joint_pos.clone()
        clock = torch.sin(2*math.pi*gait.reward_phase)
        amplitude = arm_amplitude_rad * selected * (1-braking)
        arm_reference[:, self.pitch_ids[0]] -= amplitude*clock
        arm_reference[:, self.pitch_ids[1]] += amplitude*clock
        arm_error = data.joint_pos[:, self.arm_ids]-arm_reference[:, self.arm_ids]
        target_error = data.joint_pos_target[:, self.arm_ids]-arm_reference[:, self.arm_ids]
        arm = .5*(bounded_square(arm_error, .25).mean(-1)
                  + bounded_square(target_error, .25).mean(-1))*(1+braking)
        if ascent_only_auxiliary:
            arm = arm * (selected | (finish_gate > 0)).to(foot.dtype)
        finish = braking*(bounded_square(data.root_lin_vel_w[:, :2]-command[:, :2], .15).mean(-1)
                         + bounded_square(data.root_ang_vel_w, .30).mean(-1))*.5
        costs = {'swing': swing, 'stance': stance, 'prelift': prelift,
                 'placement': placement, 'transfer': transfer, 'lateral': lateral,
                 'arm': arm, 'finish': finish}
        weights = (swing_weight, stance_weight, prelift_weight, placement_weight,
                   transfer_weight, lateral_weight, arm_weight, finish_weight)
        dense_cost = sum(w*c for w,c in zip(weights, costs.values()))
        motion_miss = (
            gait.transition_motion_miss_event.to(foot.dtype)
            if adaptive_timing else torch.zeros_like(dense_cost)
        )
        # RewardManager multiplies the term by env.step_dt. These are one-shot
        # events, so convert configured event magnitudes to per-second rates.
        event_rate = 1.0 / float(env.step_dt)
        dense_cost = dense_cost + motion_miss_weight * motion_miss * event_rate

        event_code = getattr(runtime, 'preferred_transition_event_code', None)
        sequence_failed = getattr(runtime, 'clean_step_over_sequence_failed', None)
        if direct_quality_weight > 0 and (event_code is None or sequence_failed is None):
            raise RuntimeError('C6 requires canonical transition and sequence-failure telemetry')
        if event_code is None:
            event_code = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        if sequence_failed is None:
            sequence_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        transition_weight = (swing_weight + stance_weight + lateral_weight
                             + prelift_weight*prelift_active.to(foot.dtype)
                             + placement_weight*landing_gate
                             + transfer_weight*(self.progress >= .80).to(foot.dtype))
        transition_cost = (swing_weight*swing + stance_weight*stance
                           + prelift_weight*prelift + lateral_weight*lateral
                           + placement_weight*placement + transfer_weight*transfer)
        transition_quality = torch.where(
            tracking_active & (transition_weight > 0),
            torch.exp(-transition_cost/transition_weight.clamp_min(1e-6)),
            torch.zeros_like(dense_cost),
        )
        foothold_center_cost = bounded_square(actual-self.landing, position_scale_m).mean(-1)
        terminal_quality = torch.where(
            on_target_support,
            (1-(foothold_center_cost + placement_raw + load_mechanics)/3).clamp(0, 1),
            torch.zeros_like(dense_cost),
        )
        credit_active = tracking_active & (self.target > 0) & ~sequence_failed
        discard = expired | sequence_failed
        self.quality_sum, self.quality_count, direct_quality = transition_quality_escrow_update(
            self.quality_sum, self.quality_count, transition_quality, terminal_quality,
            credit_active, event_code, discard=discard,
        )
        self.total = dense_cost - direct_quality_weight * direct_quality * event_rate
        self.trace = {**{'cost_'+k: v for k,v in costs.items()},
                      'total_cost': self.total, 'dense_cost': dense_cost,
                      'reference_selected': selected.clone(),
                      'reference_active': tracking_active.clone(),
                      'reference_expired': expired.clone(),
                      'liftoff_seen': swing_started.clone(),
                      'filtered_frequency_hz': gait.processed_actions[:, 0].clone(),
                      'transition_quality_sample': transition_quality,
                      'transition_quality_sum': self.quality_sum.clone(),
                      'transition_quality_samples': self.quality_count.clone(),
                      'direct_quality_credit': direct_quality,
                      'direct_terminal_quality': terminal_quality,
                      'transition_motion_miss_event': motion_miss,
                      'transition_motion_miss_latched': gait.transition_motion_miss.clone() if adaptive_timing else torch.zeros_like(selected),
                      'transition_motion_miss_reason': gait.transition_motion_miss_reason.clone() if adaptive_timing else torch.zeros_like(self.target),
                      'selection_wait_s': gait.selection_wait_s.clone() if adaptive_timing else torch.zeros_like(self.progress),
                      'approach_unload_dwell_s': gait.approach_unload_dwell_s.clone() if adaptive_timing else torch.zeros(env.num_envs, 2, device=env.device),
                      'approach_simultaneous_unload': gait.approach_simultaneous_unload.clone() if adaptive_timing else torch.zeros_like(selected),
                      'approach_committed': gait.approach_committed.clone() if adaptive_timing else torch.zeros_like(selected),
                      'approach_clear_pending': gait.approach_clear_pending.clone() if adaptive_timing else torch.zeros_like(selected),
                      'foothold_center_cost': foothold_center_cost,
                      'target_support_valid': on_target_support,
                      'reference_target': self.target.clone(),
                      'reference_foot': self.preferred.clone(),
                      'swing_progress': self.progress.clone(),
                      'landing_guidance_gate': landing_gate.clone(),
                      'landing_spatial_progress': spatial_landing_progress.clone(),
                      'landing_remaining_xy_m': horizontal_remaining_m.clone(),
                      'desired_sole_centre_local': reference,
                      'actual_sole_centre_local': actual,
                      'swing_error_xyz_m': error,
                      'desired_sole_velocity_mps': reference_velocity,
                      'desired_sole_acceleration_mps2': reference_acceleration,
                      'actual_sole_velocity_mps': surface_velocity[rows, idx].mean(-2),
                      'sole_bounds_local': bounds, 'sole_surface_velocity_w': surface_velocity,
                      'landing_margins_m': margins, 'foot_tilt_rad': tilt, 'foot_yaw_rad': yaw,
                      'foot_load_ratio': load_ratio, 'finish_gate': finish_gate,
                      'braking_gate': braking, 'arm_reference_rad': arm_reference[:, self.arm_ids],
                      'arm_error_rad': arm_error, 'arm_target_error_rad': target_error}

        # Score the OLD transition before installing the next canonical target.
        event = runtime.step_acquired_event
        unique = runtime.physical_touchdown_event.sum(-1) == 1
        next_target = runtime.target_step[:, 0]
        new = event & unique & (next_target < int(cfg.step_count))
        next_foot = runtime.preferred_foot.clamp(0, 1)
        next_start = centre[rows, next_foot]
        next_land = self._landing_target(
            cfg, height, next_target, self.lane[rows, next_foot], foot.dtype, foot.device)
        self.start.copy_(torch.where(new[:, None], next_start, self.start))
        self.landing.copy_(torch.where(new[:, None], next_land, self.landing))
        self.target.copy_(torch.where(event, torch.where(new, next_target, -1), self.target))
        self.preferred.copy_(torch.where(event, torch.where(new, runtime.preferred_foot, -1), self.preferred))
        self.progress.copy_(torch.where(event, 0., self.progress))
        self.previous_reference_velocity.copy_(torch.where(
            event[:, None], torch.zeros_like(self.previous_reference_velocity),
            self.previous_reference_velocity,
        ))
        return self.total
