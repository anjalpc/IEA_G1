"""Opt-in C3 internal cadence action; physical joint control is unchanged."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg, RewardTermCfg
from isaaclab.utils import configclass

from .contracts import BODY_JOINTS
from .gait_phase import (
    advance_gait_phase,
    gait_frequency,
    gait_phase_observation,
    phase_swing_contact_loss,
)
from .terrain_state import stair_cfg, riser_height_per_env


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
            if tuple(dimensions["policy"]) != (489,):
                raise RuntimeError(f"C3 actor dimensions: {dimensions['policy']}")
            if tuple(dimensions["critic"]) != (691,):
                raise RuntimeError(f"C3 critic dimensions: {dimensions['critic']}")
            self._checked_contract = True

        self._raw.copy_(actions)
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

    def finish_step(self, runtime, touchdown_force_n):
        """Score old phase, then advance/realign for the next observation."""
        env = self._env
        step = int(env.common_step_counter)
        if self.last_finished_step == step:
            return
        if not runtime.dynamic_alternating_rewards:
            raise RuntimeError("C3 requires dynamic alternating acquisition")
        if not runtime.global_stair_acquisition:
            raise RuntimeError("C3 requires synchronized global stair targets")

        cfg = stair_cfg(env)
        event = runtime.step_acquired_event
        contacts = runtime.physical_touchdown_event
        unique = event & (contacts.sum(dim=1) == 1)

        # Ambiguous simultaneous acquisition cannot establish a support foot.
        self.aligned.copy_(
            torch.where(event, unique, self.aligned)
        )
        target = runtime.target_step[:, 0]
        valid = (
            self.aligned
            & (runtime.preferred_foot >= 0)
            & (runtime.preferred_foot < 2)
            & (target > 0)
            & (target < int(cfg.step_count))
            & (riser_height_per_env(env) > 1.0e-8)
        )

        vertical_force = runtime.sensor.data.net_forces_w[
            :, runtime.sensor_foot_ids, 2
        ].clamp_min(0.0)

        # No penalty before first acquisition or after final acquisition.
        self.reward_active.logical_and_(valid)
        self.contact_loss.copy_(
            phase_swing_contact_loss(
                self.reward_phase,
                self.reward_active,
                vertical_force,
                touchdown_force_n=float(touchdown_force_n),
            )
        )

        acquiring_foot = torch.where(
            unique,
            contacts.to(torch.int64).argmax(dim=1),
            torch.full_like(target, -1),
        )
        self.phase.copy_(
            advance_gait_phase(
                self.phase,
                self._frequency[:, 0],
                valid,
                acquiring_foot,
                float(env.step_dt),
            )
        )
        self.active.copy_(valid)
        self.last_finished_step = step

    def observation(self):
        return gait_phase_observation(self.phase, self.active)


@configclass
class GaitFrequencyActionCfg(ActionTermCfg):
    class_type: type = GaitFrequencyAction
    asset_name: str = "robot"
    learned: bool = True


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
        raise ValueError("C3 cannot include either C2 foothold mode")
    if observations.get("actor_height_scan", True):
        raise ValueError("C3 requires actor_height_scan=false")
    if not observations.get("critic_height_scan", False):
        raise ValueError("C3 requires critic_height_scan=true")

    initialization = resolved.get("policy_initialization", {})
    if initialization.get("stair_observations") != 489:
        raise ValueError("C3 initialization must declare 489 actor inputs")
    if initialization.get("actions") != 30:
        raise ValueError("C3 initialization must declare 30 policy outputs")
    if resolved.get("ppo", {}).get("observation_normalization", False):
        raise ValueError("C3 preserves unnormalized observations")

    env_cfg.gait_phase_enabled = True
    env_cfg.actions.gait_frequency = GaitFrequencyActionCfg(
        learned=(mode == "learned")
    )
    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        if group.last_action.history_length != 5:
            raise ValueError("C3 requires five-frame physical action history")
        group.last_action.func = physical_last_action
        group.last_action.params = {}

    env_cfg.rewards.action_rate.func = physical_action_rate
    env_cfg.rewards.action_rate.params = {}
    env_cfg.rewards.phase_swing_contact = RewardTermCfg(
        func=phase_swing_contact, weight=-0.10, params={}
    )
