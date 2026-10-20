"""Opt-in C3 internal cadence action; physical joint control is unchanged."""

import math
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg, RewardTermCfg
from isaaclab.utils import configclass

from .contracts import BODY_JOINTS
from .gait_phase import (
    advance_gait_phase,
    gait_frequency,
    smooth_gait_frequency,
    gait_phase_observation,
    phase_swing_contact_loss,
)
from .terrain_state import stair_cfg, riser_height_per_env
from .motion_reference import sole_state


def physical_last_action(env):
    return env.action_manager.action[:, :len(BODY_JOINTS)]


def physical_action_rate(env):
    count = len(BODY_JOINTS)
    delta = (
        env.action_manager.action[:, :count]
        - env.action_manager.prev_action[:, :count]
    )
    return delta.square().sum(dim=-1)


class GaitFrequencyAction(ActionTerm):
    """One internal action. Never writes joint targets or torques."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 1), device=env.device)
        self._frequency = torch.ones_like(self._raw)
        self.phase = torch.zeros(env.num_envs, device=env.device)
        self.active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.aligned = torch.zeros_like(self.active)
        self.reward_phase = torch.zeros_like(self.phase)
        self.reward_active = torch.zeros_like(self.active)
        self.contact_loss = torch.zeros_like(self.phase)
        self.swing_started = torch.zeros_like(self.active)
        self.reference_expired = torch.zeros_like(self.active)
        self.endpoint_hold_s = torch.zeros_like(self.phase)
        self.selection_wait_s = torch.zeros_like(self.phase)
        self.approach_unload_dwell_s = torch.zeros((env.num_envs, 2), device=env.device)
        self.transition_motion_miss = torch.zeros_like(self.active)
        self.transition_motion_miss_event = torch.zeros_like(self.active)
        self.transition_motion_miss_reason = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.approach_simultaneous_unload = torch.zeros_like(self.active)
        self.approach_committed = torch.zeros_like(self.active)
        self.approach_clear_pending = torch.zeros_like(self.active)
        self.approach_preferred_foot = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.last_finished_step = -1
        self._checked_contract = False

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._frequency

    def process_actions(self, actions):
        if not self._checked_contract:
            manager = self._env.action_manager
            if list(manager.active_terms) != [
                "JointPositionAction", "gait_frequency"
            ]:
                raise RuntimeError("C3 requires 29 joint actions followed by cadence")
            joint_term = manager.get_term("JointPositionAction")
            if joint_term.action_dim != len(BODY_JOINTS):
                raise RuntimeError("C3 physical action dimension changed")
            if list(joint_term._joint_names) != list(BODY_JOINTS):
                raise RuntimeError("C3 physical joint ordering changed")
            if manager.total_action_dim != 30:
                raise RuntimeError("C3 requires exactly 30 policy outputs")
            dimensions = self._env.observation_manager.group_obs_dim
            expected_policy = int(self.cfg.policy_observation_dim)
            expected_critic = int(self.cfg.critic_observation_dim)
            if tuple(dimensions["policy"]) != (expected_policy,):
                raise RuntimeError(
                    f"gait actor dimensions: {dimensions['policy']} != {(expected_policy,)}"
                )
            if tuple(dimensions["critic"]) != (expected_critic,):
                raise RuntimeError(
                    f"gait critic dimensions: {dimensions['critic']} != {(expected_critic,)}"
                )
            self._checked_contract = True

        self._raw.copy_(actions)
        if float(self.cfg.smoothing_tau_s) > 0:
            filtered = smooth_gait_frequency(
                self._frequency, actions, dt=float(self._env.step_dt),
                tau_s=float(self.cfg.smoothing_tau_s), learned=self.cfg.learned,
            )
            if self.cfg.adaptive_transition_clock:
                # Frequency is not part of the actor observation; avoid a hidden
                # preloaded filter state while the gait clock is inactive.
                filtered = torch.where(self.active[:, None], filtered, torch.ones_like(filtered))
            self._frequency.copy_(filtered)
        else:
            self._frequency.copy_(
                gait_frequency(actions, learned=self.cfg.learned)
            )
        # Snapshot the phase seen when this action was chosen.
        self.reward_phase.copy_(self.phase)
        self.reward_active.copy_(self.active)

    def apply_actions(self):
        # Cadence is internal; only JointPositionAction controls the robot.
        pass

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._frequency[ids] = 1.0
        self.phase[ids] = 0.0
        self.active[ids] = False
        self.aligned[ids] = False
        self.reward_phase[ids] = 0.0
        self.reward_active[ids] = False
        self.contact_loss[ids] = 0.0
        self.swing_started[ids] = False
        self.reference_expired[ids] = False
        self.endpoint_hold_s[ids] = 0.0
        self.selection_wait_s[ids] = 0.0
        self.approach_unload_dwell_s[ids] = 0.0
        self.transition_motion_miss[ids] = False
        self.transition_motion_miss_event[ids] = False
        self.transition_motion_miss_reason[ids] = 0
        self.approach_simultaneous_unload[ids] = False
        self.approach_committed[ids] = False
        self.approach_clear_pending[ids] = False
        self.approach_preferred_foot[ids] = -1

    def finish_step(self, runtime, touchdown_force_n):
        """Score old phase, then advance/realign for the next observation."""
        env = self._env
        step = int(env.common_step_counter)
        if self.last_finished_step == step:
            return
        if self.cfg.preservation_mode:
            # Architecture-preservation control: keep the 30th internal action
            # and two gait observation channels present without introducing C3
            # acquisition, phase, or reward semantics into the V07 task.
            self.phase.zero_()
            self.active.zero_()
            self.aligned.zero_()
            self.reward_phase.zero_()
            self.reward_active.zero_()
            self.contact_loss.zero_()
            self.last_finished_step = step
            return
        if not runtime.dynamic_alternating_rewards:
            raise RuntimeError("C3 requires dynamic alternating acquisition")
        if not runtime.global_stair_acquisition:
            raise RuntimeError("C3 requires synchronized global stair targets")

        cfg = stair_cfg(env)
        event = runtime.step_acquired_event
        contacts = runtime.physical_touchdown_event
        unique = event & (contacts.sum(dim=1) == 1)
        target = runtime.target_step[:, 0]
        height = riser_height_per_env(env)
        vertical_force = runtime.sensor.data.net_forces_w[
            :, runtime.sensor_foot_ids, 2
        ].clamp_min(0.0)

        if not self.cfg.adaptive_transition_clock:
            self.aligned.copy_(torch.where(event, unique, self.aligned))
            valid = (
                self.aligned
                & (runtime.preferred_foot >= 0)
                & (runtime.preferred_foot < 2)
                & (target > 0)
                & (target < int(cfg.step_count))
                & (height > 1.0e-8)
            )
            self.reward_active.logical_and_(valid)
            self.contact_loss.copy_(phase_swing_contact_loss(
                self.reward_phase, self.reward_active, vertical_force,
                touchdown_force_n=float(touchdown_force_n),
            ))
            acquiring_foot = torch.where(
                unique, contacts.to(torch.int64).argmax(dim=1),
                torch.full_like(target, -1),
            )
            advance = advance_gait_phase
            if getattr(env.cfg, "coordinated_motion_enabled", False):
                from .motion_reference import advance_reference_phase
                advance = lambda phase, frequency, active, acquired, dt: advance_reference_phase(
                    phase, frequency, active, acquired, dt, runtime.sample_preferred_foot
                )
            self.phase.copy_(advance(
                self.phase, self._frequency[:, 0], valid, acquiring_foot,
                float(env.step_dt),
            ))
            self.active.copy_(valid)
            self.last_finished_step = step
            return

        # C6.1: first-step selection requires sustained asymmetric unloading
        # while the base is in the local pre-riser window. A one-sample contact
        # dropout or simultaneous unloading cannot select a foot.
        # A committed failed approach is kept alive for one reward sample, then
        # cleared here on the following step.
        clear_pending = self.approach_clear_pending.clone()
        self.approach_preferred_foot.copy_(torch.where(
            clear_pending, torch.full_like(self.approach_preferred_foot, -1),
            self.approach_preferred_foot,
        ))
        self.swing_started.copy_(torch.where(
            clear_pending, torch.zeros_like(self.swing_started), self.swing_started
        ))
        self.active.copy_(torch.where(
            clear_pending, torch.zeros_like(self.active), self.active
        ))
        self.phase.copy_(torch.where(
            clear_pending, torch.zeros_like(self.phase), self.phase
        ))
        self.reference_expired.copy_(torch.where(
            clear_pending, torch.zeros_like(self.reference_expired), self.reference_expired
        ))
        self.endpoint_hold_s.copy_(torch.where(
            clear_pending, torch.zeros_like(self.endpoint_hold_s), self.endpoint_hold_s
        ))
        self.selection_wait_s.copy_(torch.where(
            clear_pending, torch.zeros_like(self.selection_wait_s), self.selection_wait_s
        ))
        self.approach_committed.copy_(torch.where(
            clear_pending, torch.zeros_like(self.approach_committed), self.approach_committed
        ))
        self.approach_clear_pending.zero_()
        self.transition_motion_miss_event.zero_()
        root_x = runtime.robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        first_riser_x = float(cfg.start_clearance)
        near_first = (
            (root_x >= first_riser_x - float(self.cfg.first_step_activation_distance_m))
            & (root_x <= first_riser_x)
        )
        left_unloaded = vertical_force[:, 0] <= float(self.cfg.liftoff_force_n)
        right_unloaded = vertical_force[:, 1] <= float(self.cfg.liftoff_force_n)
        left_ready = left_unloaded & (vertical_force[:, 1] >= float(self.cfg.support_force_n))
        right_ready = right_unloaded & (vertical_force[:, 0] >= float(self.cfg.support_force_n))

        # A first-step reference selected from approach unloading is provisional.
        # If that foot completes an ordinary re-landing wholly before the first
        # riser, it was only an approach walking step. Re-open first-step
        # selection so the next genuine swing foot can own tread 1.
        foot_pos = (
            runtime.robot.data.body_pos_w[:, runtime.robot_foot_ids]
            - env.scene.env_origins[:, None]
        )
        sole_centre, sole_bounds, _ = sole_state(
            foot_pos,
            runtime.robot.data.body_quat_w[:, runtime.robot_foot_ids],
            runtime.robot.data.body_lin_vel_w[:, runtime.robot_foot_ids],
            runtime.robot.data.body_ang_vel_w[:, runtime.robot_foot_ids],
        )
        rows = torch.arange(env.num_envs, device=env.device)
        approach_idx = self.approach_preferred_foot.clamp(0, 1)
        approach_front_x = sole_bounds[rows, approach_idx, 1]
        approach_force = vertical_force[rows, approach_idx]
        approach_support_force = vertical_force[rows, 1 - approach_idx]

        # Commitment is deliberately later than preview selection but earlier
        # than riser contact. Reuse the same 5 cm pre-riser interaction window
        # used by the strict clearance contract: the selected foot must already
        # be unloaded with valid opposite support while physically entering that
        # zone. Ordinary approach swings farther from the riser remain previews.
        approach_commit_now = (
            (target == 0)
            & (height > 1.0e-8)
            & (self.approach_preferred_foot >= 0)
            & (approach_force <= float(self.cfg.liftoff_force_n))
            & (approach_support_force >= float(self.cfg.support_force_n))
            & (approach_front_x >= first_riser_x - 0.05)
        )
        commit_max_travel = getattr(self.cfg, "first_step_commit_max_travel_m", None)
        if commit_max_travel is None:
            self.approach_committed.logical_or_(approach_commit_now)

        replant_before_riser = (
            (target == 0)
            & (height > 1.0e-8)
            & (self.approach_preferred_foot >= 0)
            & self.swing_started
            & ~event
            & (approach_force >= float(self.cfg.support_force_n))
            & (approach_front_x < first_riser_x)
        )
        committed_replant = replant_before_riser & self.approach_committed
        pre_riser_replant = replant_before_riser & ~self.approach_committed

        # A committed attempt may not silently erase its stair objective. Keep
        # the reference alive for this reward sample, emit one explicit miss,
        # and defer cleanup until the next finish_step call.
        new_committed_miss = committed_replant & ~self.transition_motion_miss
        self.transition_motion_miss_event.logical_or_(new_committed_miss)
        self.transition_motion_miss.logical_or_(new_committed_miss)
        self.transition_motion_miss_reason.copy_(torch.where(
            new_committed_miss,
            torch.full_like(self.transition_motion_miss_reason, 3),
            self.transition_motion_miss_reason,
        ))
        self.approach_clear_pending.logical_or_(new_committed_miss)

        self.approach_preferred_foot.copy_(torch.where(
            pre_riser_replant,
            torch.full_like(self.approach_preferred_foot, -1),
            self.approach_preferred_foot,
        ))
        self.swing_started.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.swing_started),
            self.swing_started,
        ))
        self.active.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.active),
            self.active,
        ))
        self.phase.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.phase),
            self.phase,
        ))
        self.reference_expired.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.reference_expired),
            self.reference_expired,
        ))
        self.endpoint_hold_s.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.endpoint_hold_s),
            self.endpoint_hold_s,
        ))
        self.selection_wait_s.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.selection_wait_s),
            self.selection_wait_s,
        ))
        self.transition_motion_miss.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.transition_motion_miss),
            self.transition_motion_miss,
        ))
        self.transition_motion_miss_reason.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.transition_motion_miss_reason),
            self.transition_motion_miss_reason,
        ))
        self.approach_committed.copy_(torch.where(
            pre_riser_replant,
            torch.zeros_like(self.approach_committed),
            self.approach_committed,
        ))

        first_open = (
            (target == 0)
            & (height > 1.0e-8)
            & near_first
            & (self.approach_preferred_foot < 0)
        )
        self.approach_simultaneous_unload.copy_(first_open & left_unloaded & right_unloaded)
        left_dwell = torch.where(
            first_open & left_ready & ~right_ready,
            self.approach_unload_dwell_s[:, 0] + float(env.step_dt),
            torch.zeros_like(self.approach_unload_dwell_s[:, 0]),
        )
        right_dwell = torch.where(
            first_open & right_ready & ~left_ready,
            self.approach_unload_dwell_s[:, 1] + float(env.step_dt),
            torch.zeros_like(self.approach_unload_dwell_s[:, 1]),
        )
        self.approach_unload_dwell_s[:, 0].copy_(left_dwell)
        self.approach_unload_dwell_s[:, 1].copy_(right_dwell)
        dwell = float(self.cfg.first_step_unload_dwell_s)
        left_selected = first_open & (left_dwell >= dwell)
        right_selected = first_open & (right_dwell >= dwell)
        select_first = left_selected ^ right_selected
        first_foot = torch.where(left_selected, torch.zeros_like(target), torch.ones_like(target))
        self.approach_preferred_foot.copy_(torch.where(
            select_first, first_foot, self.approach_preferred_foot
        ))

        if commit_max_travel is not None:
            # Diagnostic eligibility bound, not a kinematic reach certificate.
            # Decide once at selection; never convert an ongoing approach
            # preview into a committed stair swing as it nears the riser.
            landing_x = first_riser_x + 0.5 * float(cfg.tread_depth)
            travel_x = landing_x - sole_centre[rows, first_foot, 0]
            eligible = (
                select_first
                & (travel_x > 0.0)
                & (travel_x <= commit_max_travel)
                & (sole_bounds[rows, first_foot, 1] < first_riser_x)
            )
            self.approach_committed.logical_or_(eligible)

        # The reference clock is foot-relative:
        # left swing occupies phase 0.5 -> 1.0,
        # right swing occupies phase 0.0 -> 0.5.
        # A newly selected first-step foot must start from its own aligned
        # phase; otherwise a reselected left foot waits half a gait cycle
        # before coordinated-motion progress can advance.
        first_phase = torch.where(
            first_foot == 0,
            torch.full_like(self.phase, 0.5),
            torch.zeros_like(self.phase),
        )
        self.phase.copy_(torch.where(select_first, first_phase, self.phase))
        self.reward_phase.copy_(torch.where(
            select_first, first_phase, self.reward_phase
        ))

        self.approach_unload_dwell_s.copy_(torch.where(
            select_first[:, None], torch.zeros_like(self.approach_unload_dwell_s),
            self.approach_unload_dwell_s,
        ))

        # Canonical acquisition selects the opposite foot for the next tread.
        self.aligned.copy_(torch.where(event, unique, self.aligned))
        canonical = (
            self.aligned
            & (runtime.preferred_foot >= 0)
            & (runtime.preferred_foot < 2)
            & (target > 0)
            & (target < int(cfg.step_count))
            & (height > 1.0e-8)
        )
        approach = (target == 0) & (self.approach_preferred_foot >= 0) & (height > 1.0e-8)
        selected = torch.where(canonical, runtime.preferred_foot, self.approach_preferred_foot)
        valid = canonical | approach
        idx = selected.clamp(0, 1)
        rows = torch.arange(env.num_envs, device=env.device)
        selected_force = vertical_force[rows, idx]
        support_force = vertical_force[rows, 1-idx]
        liftoff = valid & (selected_force <= float(self.cfg.liftoff_force_n)) & (support_force >= float(self.cfg.support_force_n))
        # A new acquisition starts a new selected transition but its next swing
        # does not start until that selected foot actually unloads.
        next_swing_started = torch.where(
            event, torch.zeros_like(self.swing_started), self.swing_started
        ) | liftoff
        self.swing_started.copy_(next_swing_started)

        failure_code = getattr(runtime, 'preferred_transition_event_code', torch.zeros_like(target))
        failed_without_advance = (~event) & (failure_code >= 2) & (failure_code <= 5)
        self.reference_expired.logical_or_(failed_without_advance)
        self.reference_expired.copy_(torch.where(event, torch.zeros_like(self.reference_expired), self.reference_expired))
        self.transition_motion_miss.copy_(torch.where(
            event, torch.zeros_like(self.transition_motion_miss), self.transition_motion_miss
        ))
        self.transition_motion_miss_reason.copy_(torch.where(
            event, torch.zeros_like(self.transition_motion_miss_reason), self.transition_motion_miss_reason
        ))

        waiting = valid & ~self.swing_started & ~self.reference_expired & ~event
        self.selection_wait_s.copy_(torch.where(
            waiting, self.selection_wait_s + float(env.step_dt),
            torch.zeros_like(self.selection_wait_s),
        ))
        liftoff_timeout_now = (
            waiting
            & (self.selection_wait_s >= float(self.cfg.liftoff_timeout_s))
            & ~self.transition_motion_miss
        )
        self.transition_motion_miss_event.logical_or_(liftoff_timeout_now)
        self.transition_motion_miss.logical_or_(liftoff_timeout_now)
        self.transition_motion_miss_reason.copy_(torch.where(
            liftoff_timeout_now, torch.ones_like(self.transition_motion_miss_reason),
            self.transition_motion_miss_reason,
        ))
        self.reference_expired.logical_or_(liftoff_timeout_now)

        self.reward_active.logical_and_(valid & self.swing_started & ~self.reference_expired)
        self.contact_loss.copy_(phase_swing_contact_loss(
            self.reward_phase, self.reward_active, vertical_force,
            touchdown_force_n=float(touchdown_force_n),
        ))

        acquiring_foot = torch.where(
            unique, contacts.to(torch.int64).argmax(dim=1),
            torch.full_like(target, -1),
        )
        from .motion_reference import advance_reference_phase
        preferred_before = torch.where(
            runtime.sample_preferred_foot >= 0,
            runtime.sample_preferred_foot,
            self.approach_preferred_foot,
        )
        proposed = advance_reference_phase(
            self.phase, self._frequency[:, 0], valid, acquiring_foot,
            float(env.step_dt), preferred_before,
            swing_started=self.swing_started, expired=self.reference_expired,
        )
        endpoint = torch.where(selected == 0, torch.ones_like(proposed), torch.full_like(proposed, .5))
        at_endpoint = valid & self.swing_started & ~event & ~self.reference_expired & ((proposed-endpoint).abs() < 1e-6)
        self.endpoint_hold_s.copy_(torch.where(
            at_endpoint, self.endpoint_hold_s + float(env.step_dt),
            torch.zeros_like(self.endpoint_hold_s),
        ))
        expired_now = self.endpoint_hold_s > float(self.cfg.touchdown_grace_s)
        touchdown_timeout_now = (
            expired_now & ~event & ~self.reference_expired & ~self.transition_motion_miss
        )
        self.transition_motion_miss_event.logical_or_(touchdown_timeout_now)
        self.transition_motion_miss.logical_or_(touchdown_timeout_now)
        self.transition_motion_miss_reason.copy_(torch.where(
            touchdown_timeout_now, torch.full_like(self.transition_motion_miss_reason, 2),
            self.transition_motion_miss_reason,
        ))
        self.reference_expired.logical_or_(expired_now)
        proposed = torch.where(self.reference_expired, torch.zeros_like(proposed), proposed)
        self.phase.copy_(proposed)
        self.active.copy_(valid & ~self.reference_expired)
        # First acquisition ends approach selection; subsequent preferred foot
        # comes only from the canonical acquisition state machine.
        self.approach_preferred_foot.copy_(torch.where(
            event, torch.full_like(self.approach_preferred_foot, -1),
            self.approach_preferred_foot,
        ))
        self.approach_committed.copy_(torch.where(
            event, torch.zeros_like(self.approach_committed), self.approach_committed
        ))
        self.approach_clear_pending.copy_(torch.where(
            event, torch.zeros_like(self.approach_clear_pending), self.approach_clear_pending
        ))
        self.last_finished_step = step

    def observation(self):
        return gait_phase_observation(self.phase, self.active)


@configclass
class GaitFrequencyActionCfg(ActionTermCfg):
    class_type: type = GaitFrequencyAction
    asset_name: str = "robot"
    learned: bool = True
    smoothing_tau_s: float = 0.0
    adaptive_transition_clock: bool = False
    liftoff_force_n: float = 15.0
    support_force_n: float = 30.0
    first_step_activation_distance_m: float = 0.35
    first_step_commit_max_travel_m: float | None = None
    first_step_unload_dwell_s: float = 0.06
    liftoff_timeout_s: float = 0.30
    touchdown_grace_s: float = 0.30
    policy_observation_dim: int = 489
    critic_observation_dim: int = 691
    preservation_mode: bool = False


def phase_swing_contact(env):
    term = env.action_manager.get_term("gait_frequency")
    if term.last_finished_step != int(env.common_step_counter):
        raise RuntimeError("C3 contact reward ran before acquisition update")
    return term.contact_loss


def configure_gait(env_cfg, resolved):
    """Called by the shared training/replay reward configuration boundary."""
    options = resolved.get("gait", {})
    if not options.get("enabled", False):
        return

    mode = options.get("mode")
    if mode not in ("learned", "fixed"):
        raise ValueError("gait.mode must be learned or fixed")

    observations = resolved.get("observations", {})
    if not observations.get("preferred_transition_foot", False):
        raise ValueError("C3 requires preferred_transition_foot")
    if (
        observations.get("preferred_foothold", False)
        or observations.get("preferred_foothold_ablation", False)
    ):
        raise ValueError("gait mode cannot include legacy C2 foothold channels")
    coordinated_foothold_xyz = bool(
        observations.get("coordinated_foothold_xyz", False)
    )
    if observations.get("actor_height_scan", True):
        raise ValueError("C3 requires actor_height_scan=false")
    if not observations.get("critic_height_scan", False):
        raise ValueError("C3 requires critic_height_scan=true")

    initialization = resolved.get("policy_initialization", {})
    expected_policy_dim = 489 + (3 if coordinated_foothold_xyz else 0)
    expected_critic_dim = 691 + (3 if coordinated_foothold_xyz else 0)
    if initialization.get("stair_observations") != expected_policy_dim:
        raise ValueError(
            f"gait initialization must declare {expected_policy_dim} actor inputs"
        )
    if initialization.get("actions") != 30:
        raise ValueError("C3 initialization must declare 30 policy outputs")
    if resolved.get("ppo", {}).get("observation_normalization", False):
        raise ValueError("C3 preserves unnormalized observations")

    outcome_conditioned = bool(options.get("outcome_conditioned_motion", False))
    direct_quality_weight = float(options.get("direct_quality_weight", 0.0))
    ascent_only_auxiliary = bool(options.get("ascent_only_auxiliary", False))
    adaptive_timing = bool(options.get("adaptive_timing", False))
    preservation_mode = bool(options.get("preservation_mode", False))
    smoothing_tau_s = float(options.get("frequency_smoothing_tau_s", 0.0))
    first_step_activation_distance_m = float(
        options.get("first_step_activation_distance_m", 0.35)
    )
    if not math.isfinite(first_step_activation_distance_m) or first_step_activation_distance_m <= 0.0:
        raise ValueError("gait.first_step_activation_distance_m must be finite and positive")
    commit_max_travel = options.get("first_step_commit_max_travel_m")
    if commit_max_travel is not None:
        commit_max_travel = float(commit_max_travel)
        if not adaptive_timing or not math.isfinite(commit_max_travel) or commit_max_travel <= 0.0:
            raise ValueError("gait.first_step_commit_max_travel_m requires adaptive timing and a finite positive distance")
    first_step_unload_dwell_s = float(options.get("first_step_unload_dwell_s", 0.06))
    liftoff_timeout_s = float(options.get("liftoff_timeout_s", 0.30))
    touchdown_grace_s = float(options.get("touchdown_grace_s", 0.30))
    motion_miss_weight = float(options.get("motion_miss_weight", 0.25))
    coordinated_weights = {
        "swing_weight": float(options.get("swing_weight", 0.35)),
        "stance_weight": float(options.get("stance_weight", 0.15)),
        "prelift_weight": float(options.get("prelift_weight", 0.20)),
        "placement_weight": float(options.get("placement_weight", 0.25)),
        "transfer_weight": float(options.get("transfer_weight", 0.15)),
        "lateral_weight": float(options.get("lateral_weight", 0.10)),
        "arm_weight": float(options.get("arm_weight", 0.15)),
        "finish_weight": float(options.get("finish_weight", 0.15)),
    }
    if any(value < 0.0 for value in coordinated_weights.values()):
        raise ValueError("coordinated-motion weights cannot be negative")
    if outcome_conditioned and not options.get("coordinated_motion", False):
        raise ValueError("outcome_conditioned_motion requires coordinated_motion")
    if direct_quality_weight < 0.0:
        raise ValueError("gait.direct_quality_weight cannot be negative")
    if outcome_conditioned and direct_quality_weight <= 0.0:
        raise ValueError("outcome-conditioned motion requires positive direct_quality_weight")
    if preservation_mode:
        if mode != "fixed":
            raise ValueError("gait.preservation_mode requires gait.mode=fixed")
        if options.get("coordinated_motion", False):
            raise ValueError("gait.preservation_mode cannot enable coordinated_motion")
        if adaptive_timing:
            raise ValueError("gait.preservation_mode cannot enable adaptive_timing")
        if outcome_conditioned:
            raise ValueError("gait.preservation_mode cannot enable outcome_conditioned_motion")
    if adaptive_timing and mode != "learned":
        raise ValueError("adaptive_timing requires gait.mode=learned")
    if adaptive_timing and smoothing_tau_s <= 0.0:
        raise ValueError("adaptive_timing requires positive frequency_smoothing_tau_s")
    if adaptive_timing and (first_step_unload_dwell_s <= 0.0 or liftoff_timeout_s <= 0.0 or touchdown_grace_s <= 0.0):
        raise ValueError("adaptive_timing requires positive dwell/timeout/grace values")
    if motion_miss_weight < 0.0:
        raise ValueError("gait.motion_miss_weight cannot be negative")

    dense_cost_scale = float(options.get("dense_cost_scale", 1.0))
    if not 0.0 <= dense_cost_scale <= 1.0:
        raise ValueError("gait.dense_cost_scale must be in [0, 1]")

    if options.get("coordinated_motion", False):
        if mode != "fixed" and not adaptive_timing:
            raise ValueError("coordinated motion requires fixed cadence unless adaptive_timing is enabled")
        from .coordinated_motion import CoordinatedMotion
        env_cfg.coordinated_motion_enabled = True
        insertion = resolved["reward_parameters"]["forward_insertion"]
        env_cfg.coordinated_landing_rear_margin_m = (
            float(insertion["tread_margin_m"]) + float(insertion["insertion_target_m"])
        )
        env_cfg.rewards.coordinated_motion = RewardTermCfg(
            func=CoordinatedMotion, weight=-1.0,
            params=dict(**coordinated_weights,
                        clearance_m=.03, position_scale_m=.06,
                        arm_amplitude_rad=.20,
                        direct_quality_weight=direct_quality_weight if outcome_conditioned else 0.0,
                        motion_miss_weight=motion_miss_weight if adaptive_timing else 0.0,
                        dense_cost_scale=dense_cost_scale,
                        ascent_only_auxiliary=ascent_only_auxiliary if outcome_conditioned else False,
                        adaptive_timing=adaptive_timing),
        )
    env_cfg.gait_phase_enabled = True
    env_cfg.actions.gait_frequency = GaitFrequencyActionCfg(
        learned=(mode == "learned"),
        smoothing_tau_s=smoothing_tau_s if adaptive_timing else 0.0,
        adaptive_transition_clock=adaptive_timing,
        first_step_activation_distance_m=first_step_activation_distance_m,
        first_step_commit_max_travel_m=commit_max_travel,
        first_step_unload_dwell_s=first_step_unload_dwell_s,
        liftoff_timeout_s=liftoff_timeout_s,
        touchdown_grace_s=touchdown_grace_s,
        policy_observation_dim=expected_policy_dim,
        critic_observation_dim=expected_critic_dim,
        preservation_mode=preservation_mode,
    )
    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        if group.last_action.history_length != 5:
            raise ValueError("C3 requires five-frame physical action history")
        group.last_action.func = physical_last_action
        group.last_action.params = {}

    env_cfg.rewards.action_rate.func = physical_action_rate
    env_cfg.rewards.action_rate.params = {}
    if not preservation_mode:
        env_cfg.rewards.phase_swing_contact = RewardTermCfg(
            func=phase_swing_contact, weight=-0.10, params={}
        )
