"""Opt-in C4 coordinated motion shaping. Robot actions are never overwritten."""
import math
import torch
from isaaclab.managers import ManagerTermBase
from .motion_reference import sole_state, swing_reference, bounded_square, rotate
from .contracts import G1_SOLE_SPHERE_CENTRES_M, G1_SOLE_SPHERE_RADIUS_M
from .terrain_state import stair_cfg, riser_height_per_env



class CoordinatedMotion(ManagerTermBase):
    """One coherent cost with individually traceable components.

    A transition reference starts once after verified acquisition. It does not
    restart on replants or clock wrap. Missed timing holds the landing target.
    The existing classifier, safety terminations and finish contract remain.
    """
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
        self.last_step = -1
        self.trace = {}
        env._g1_coordinated_motion = self
        names = list(env.scene['robot'].joint_names)
        self.arm_ids = [i for i, name in enumerate(names)
                        if any(x in name for x in ('shoulder', 'elbow', 'wrist'))]
        if len(self.arm_ids) != 14:
            raise ValueError('C4 requires the 14 G1 arm joints')
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
        # Keep the global step stamp: re-running all rows would double-process
        # unreset environments and could consume an old acquisition event.
        # Reset rows instead become neutral until the next physics step.
        if hasattr(self, 'total'):
            self.total[ids] = 0
        for value in self.trace.values():
            value[ids] = 0
        for name in ('reference_target', 'reference_foot'):
            if name in self.trace:
                self.trace[name][ids] = -1

    def __call__(self, env, swing_weight=.50, stance_weight=.15,
                 placement_weight=.25, transfer_weight=.15,
                 lateral_weight=.10, arm_weight=.15, finish_weight=.15,
                 clearance_m=.03, position_scale_m=.06,
                 arm_amplitude_rad=.20):
        values = (swing_weight, stance_weight, placement_weight, transfer_weight,
                  lateral_weight, arm_weight, finish_weight, clearance_m,
                  position_scale_m, arm_amplitude_rad)
        if any(not math.isfinite(x) or x < 0 for x in values) or position_scale_m <= 0:
            raise ValueError('C4 parameters must be finite/nonnegative; scale positive')
        runtime = getattr(env, '_g1_v04_acquisition', None)
        step = int(env.common_step_counter)
        if runtime is None or runtime.last_step != step:
            raise RuntimeError('C4 must run after acquisition rewards')
        if self.last_step == step:
            return self.total
        self.last_step = step
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
        idx = self.preferred.clamp(0, 1)
        actual = centre[rows, idx]
        valid = ((self.target > 0) & (self.target < int(cfg.step_count))
                 & (self.preferred >= 0) & (height > 1e-8))
        gait = env.action_manager.get_term('gait_frequency')
        # Both reward and reference use the action's PRE-step phase snapshot.
        # Only GaitFrequencyAction advances/writes the next observed phase.
        start_phase = torch.where(idx == 0, .5, 0.)
        before_progress = (2*(gait.reward_phase-start_phase)).clamp(0, 1)
        self.progress.copy_((before_progress +
            2*float(env.step_dt)*gait.processed_actions[:, 0]).clamp(0, 1))
        reference = swing_reference(self.start, self.landing, self.progress, clearance_m)
        previous_reference = swing_reference(
            self.start, self.landing,
            before_progress,
            clearance_m)
        reference_velocity = (reference-previous_reference)/float(env.step_dt)
        error = actual - reference
        swing = bounded_square(error, position_scale_m).mean(-1) * valid
        # Keep tracking the landing endpoint after the nominal swing duration.
        support_idx = 1-idx
        support_force = force[rows, support_idx]
        support_slip = surface_velocity[rows, support_idx, :, :2].norm(dim=-1).amax(-1)
        swing_contact = ((self.progress > .20) & (self.progress < .80)).to(foot.dtype) * (force[rows, idx]/30.).clamp(0, 1)
        stance = (bounded_square(support_slip, .10)
                  + (1-support_force/30.).clamp(0, 1) + swing_contact)/3*valid
        target_x = float(cfg.start_clearance) + self.target.to(foot.dtype)*float(cfg.tread_depth)
        selected = bounds[rows, idx]
        margins = torch.stack((selected[:, 0]-target_x,
                               torch.where(self.target == int(cfg.step_count)-1,
                                   torch.full_like(target_x, float(cfg.size[0])-float(cfg.approach_length)+float(cfg.start_clearance)),
                                   target_x+float(cfg.tread_depth))-selected[:, 1],
                               selected[:, 2]+float(cfg.size[1])/2,
                               float(cfg.size[1])/2-selected[:, 3]), -1)
        landing_gate = ((self.progress-.70)/.30).clamp(0, 1)*valid
        edge_cost = bounded_square((.015-margins).clamp_min(0), .03).mean(-1)
        up = rotate(quat, foot.new_tensor([0., 0., 1.]).expand_as(foot))
        tilt = torch.acos(up[..., 2].clamp(-1, 1))
        forward = rotate(quat, foot.new_tensor([1., 0., 0.]).expand_as(foot))
        yaw = torch.atan2(forward[..., 1], forward[..., 0])
        surface_speed = surface_velocity.norm(dim=-1).amax(-1)
        placement = landing_gate*(edge_cost + bounded_square(tilt[rows, idx], .15)
                      + bounded_square(yaw[rows, idx], .20)
                      + bounded_square(surface_speed[rows, idx], .20))/4
        load_ratio = force / force.sum(-1, keepdim=True).clamp_min(1.)
        transfer = valid*(self.progress >= 1)*(
            (1-load_ratio[rows, idx]/.45).clamp(0, 1)
            + (1-force[rows, idx]/30.).clamp(0, 1))*.5
        # Straight-course lateral corridor also covers approach swings.
        lane_error = (centre[..., 1]-self.lane).abs()
        lateral = (bounded_square((lane_error-.04).clamp_min(0), .05)
                   * (force < 30)).mean(-1)
        platform_x = float(cfg.start_clearance)+(int(cfg.step_count)-1)*float(cfg.tread_depth)
        root = data.root_pos_w-env.scene.env_origins
        command = env.command_manager.get_command('base_velocity')
        # Progression, not observed stillness, activates finish pressure.
        finish_gate = ((runtime.target_step[:, 0] >= int(cfg.step_count))
                       & (root[:, 0] >= platform_x)).to(foot.dtype)
        braking = finish_gate*(1-command[:, 0].abs()/.35).clamp(0, 1)
        arm_reference = data.default_joint_pos.clone()
        clock = torch.sin(2*math.pi*gait.reward_phase)
        # Opposite shoulder pitch signs. Reference remains soft, never a command.
        amplitude = arm_amplitude_rad * valid * (1-braking)
        arm_reference[:, self.pitch_ids[0]] -= amplitude*clock
        arm_reference[:, self.pitch_ids[1]] += amplitude*clock
        arm_error = data.joint_pos[:, self.arm_ids]-arm_reference[:, self.arm_ids]
        target_error = data.joint_pos_target[:, self.arm_ids]-arm_reference[:, self.arm_ids]
        arm = .5*(bounded_square(arm_error, .25).mean(-1)
                   + bounded_square(target_error, .25).mean(-1))*(1+braking)
        finish = braking*(bounded_square(data.root_lin_vel_w[:, :2]-command[:, :2], .15).mean(-1)
                         + bounded_square(data.root_ang_vel_w, .30).mean(-1))*.5
        costs = {'swing': swing, 'stance': stance, 'placement': placement,
                 'transfer': transfer, 'lateral': lateral, 'arm': arm, 'finish': finish}
        weights = (swing_weight, stance_weight, placement_weight, transfer_weight,
                   lateral_weight, arm_weight, finish_weight)
        self.total = sum(w*c for w,c in zip(weights, costs.values()))
        self.trace = {**{'cost_'+k: v for k,v in costs.items()},
                      'total_cost': self.total, 'reference_active': valid.clone(),
                      'reference_target': self.target.clone(),
                      'reference_foot': self.preferred.clone(),
                      'swing_progress': self.progress.clone(),
                      'desired_sole_centre_local': reference,
                      'actual_sole_centre_local': actual,
                      'swing_error_xyz_m': error,
                      'desired_sole_velocity_mps': reference_velocity,
                      'actual_sole_velocity_mps': surface_velocity[rows, idx].mean(-2),
                      'sole_bounds_local': bounds, 'sole_surface_velocity_w': surface_velocity,
                      'landing_margins_m': margins, 'foot_tilt_rad': tilt, 'foot_yaw_rad': yaw,
                      'foot_load_ratio': load_ratio, 'finish_gate': finish_gate,
                      'braking_gate': braking, 'arm_reference_rad': arm_reference[:, self.arm_ids],
                      'arm_error_rad': arm_error, 'arm_target_error_rad': target_error}
        # Score the OLD transition before installing the next target.
        event = runtime.step_acquired_event
        unique = runtime.physical_touchdown_event.sum(-1) == 1
        next_target = runtime.target_step[:, 0]
        new = event & unique & (next_target < int(cfg.step_count))
        next_foot = runtime.preferred_foot.clamp(0, 1)
        start = centre[rows, next_foot]
        land = start.clone()
        # Match the existing insertion objective instead of opposing it.
        rear_margin = float(env.cfg.coordinated_landing_rear_margin_m)
        neutral_span = (max(p[0] for p in G1_SOLE_SPHERE_CENTRES_M)
                        - min(p[0] for p in G1_SOLE_SPHERE_CENTRES_M)
                        + 2*G1_SOLE_SPHERE_RADIUS_M)
        low = rear_margin + neutral_span/2
        high = float(cfg.tread_depth)-.015-neutral_span/2
        if low > high+1e-6:
            raise ValueError('C4 tread cannot fit the full sole and configured insertion margins')
        land[:, 0] = float(cfg.start_clearance)+next_target.to(foot.dtype)*float(cfg.tread_depth)+(low+high)/2
        land[:, 1] = self.lane[rows, next_foot]
        land[:, 2] = (next_target+1)*height
        self.start.copy_(torch.where(new[:, None], start, self.start))
        self.landing.copy_(torch.where(new[:, None], land, self.landing))
        self.target.copy_(torch.where(event, torch.where(new, next_target, -1), self.target))
        self.preferred.copy_(torch.where(event, torch.where(new, runtime.preferred_foot, -1), self.preferred))
        self.progress.copy_(torch.where(event, 0., self.progress))
        return self.total
