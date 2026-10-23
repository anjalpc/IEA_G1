
"""Read-only state sampling after reward computation and before reset."""
import json
from pathlib import Path


class GeometryTrace:
    def __init__(self, env, output):
        from simulation.isaac_lab.g1_stairs.contracts import (
            G1_SOLE_SPHERE_CENTRES_M, G1_SOLE_SPHERE_RADIUS_M,
        )
        from simulation.isaac_lab.g1_stairs.terrain_state import (
            stair_cfg, riser_height_per_env,
        )
        from simulation.isaac_lab.g1_stairs.motion_reference import sole_state
        self.sole_state = sole_state
        self.env = env
        self.robot = env.scene["robot"]
        self.sensor = env.scene.sensors["contact_forces"]
        self.manager = env.reward_manager
        self.original_compute = self.manager.compute
        self.feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_ids = [self.robot.body_names.index(n) for n in self.feet]
        self.knees = ["left_knee_link", "right_knee_link"]
        self.knee_ids = [self.robot.body_names.index(n) for n in self.knees]
        cfg = stair_cfg(env)
        metadata = {
            "schema_version": 1,
            "sampling": "after reward computation, before automatic reset",
            "step_dt_s": float(env.step_dt),
            "physics_dt_s": float(env.physics_dt),
            "quaternion_order": "wxyz",
            "positions": "world coordinates; subtract env_origin_w for course coordinates",
            "foot_names": self.feet,
            "knee_names": self.knees,
            "joint_names": list(self.robot.joint_names),
            "default_joint_pos_rad": self.values(self.robot.data.default_joint_pos[0]),
            "diagnostic_schema_version": 2,
            "sole_geometry_semantics": "collision-sphere envelope, not measured contact-patch area",
            "contact_body_names": list(self.sensor.body_names),
            "reward_names": list(self.manager.active_terms),
            "sole_sphere_centres_body_m": G1_SOLE_SPHERE_CENTRES_M,
            "sole_sphere_radius_m": G1_SOLE_SPHERE_RADIUS_M,
            "first_riser_x_local_m": float(cfg.start_clearance),
            "tread_depth_m": float(cfg.tread_depth),
            "step_count": int(cfg.step_count),
            "riser_height_m": float(riser_height_per_env(env)[0].item()),
            "reward_contributions": "weighted term values multiplied by step_dt",
        }
        output = Path(output)
        with (output / "trace_metadata.json").open("x") as stream:
            json.dump(metadata, stream, indent=2)
        self.stream = (output / "trace.jsonl").open("x", buffering=1)
        self.capture(None)
        self.manager.compute = self.compute

    @staticmethod
    def values(tensor):
        return tensor.detach().cpu().tolist()

    def capture(self, reward):
        env = self.env
        data = self.robot.data
        row = {
            "sample": "initial" if reward is None else "post_physics_pre_reset",
            "global_step": int(env.common_step_counter),
            "episode_step": int(env.episode_length_buf[0].item()),
            "episode_time_s": float(env.episode_length_buf[0].item() * env.step_dt),
            "env_origin_w": self.values(env.scene.env_origins[0]),
            "root_pos_w": self.values(data.root_pos_w[0]),
            "root_quat_w": self.values(data.root_quat_w[0]),
            "root_lin_vel_w": self.values(data.root_lin_vel_w[0]),
            "root_ang_vel_w": self.values(data.root_ang_vel_w[0]),
            "command_base_velocity": self.values(
                env.command_manager.get_command("base_velocity")[0]
            ),
            "foot_pos_w": self.values(data.body_pos_w[0, self.foot_ids]),
            "foot_quat_w": self.values(data.body_quat_w[0, self.foot_ids]),
            "foot_lin_vel_w": self.values(data.body_lin_vel_w[0, self.foot_ids]),
            "foot_ang_vel_w": self.values(data.body_ang_vel_w[0, self.foot_ids]),
            "knee_pos_w": self.values(data.body_pos_w[0, self.knee_ids]),
            "knee_lin_vel_w": self.values(data.body_lin_vel_w[0, self.knee_ids]),
            "contact_net_forces_w": self.values(self.sensor.data.net_forces_w[0]),
            "joint_pos_rad": self.values(data.joint_pos[0]),
            "joint_vel_rad_s": self.values(data.joint_vel[0]),
            "applied_torque_nm": self.values(data.applied_torque[0]),
            "joint_pos_target_rad": self.values(data.joint_pos_target[0]),
            "action": self.values(
                env.action_manager.action[0, :29]
                if getattr(env.cfg, "gait_phase_enabled", False)
                else env.action_manager.action[0]
            ),
        }
        centre, bounds, surface_velocity = self.sole_state(
            data.body_pos_w[0:1, self.foot_ids] - env.scene.env_origins[0:1, None],
            data.body_quat_w[0:1, self.foot_ids],
            data.body_lin_vel_w[0:1, self.foot_ids],
            data.body_ang_vel_w[0:1, self.foot_ids],
        )
        row["motion_geometry"] = {
            "sole_centre_local": self.values(centre[0]),
            "sole_bounds_local": self.values(bounds[0]),
            "surface_velocity_w": self.values(surface_velocity[0]),
            "surface_speed_mps": self.values(surface_velocity.norm(dim=-1).amax(-1)[0]),
            "surface_slip_mps": self.values(surface_velocity[..., :2].norm(dim=-1).amax(-1)[0]),
        }
        row["knee_ang_vel_w"] = self.values(data.body_ang_vel_w[0, self.knee_ids])
        row["knee_quat_w"] = self.values(data.body_quat_w[0, self.knee_ids])
        history = getattr(self.sensor.data, "net_forces_w_history", None)
        if history is not None:
            row["contact_force_history_w"] = self.values(history[0])
        motion = getattr(env, "_g1_coordinated_motion", None)
        if motion is not None and reward is not None:
            row["coordinated_motion"] = {
                key: self.values(value[0]) for key, value in motion.trace.items()
            }
        if getattr(env.cfg, "gait_phase_enabled", False):
            gait = env.action_manager.get_term("gait_frequency")
            clock = gait.observation()[0]
            row.update({
                "raw_frequency_action": self.values(gait.raw_actions[0, 0]),
                "frequency_hz": self.values(gait.processed_actions[0, 0]),
                "phase": self.values(gait.phase[0]),
                "sin_phase": self.values(clock[0]),
                "cos_phase": self.values(clock[1]),
                "gait_active": self.values(gait.active[0]),
                "gait_reward_phase": self.values(gait.reward_phase[0]),
                "gait_reward_active": self.values(gait.reward_active[0]),
                "phase_swing_contact_loss": self.values(gait.contact_loss[0]),
                "approach_committed": self.values(gait.approach_committed[0]),
                "approach_clear_pending": self.values(gait.approach_clear_pending[0]),
            })
        if reward is not None:
            row.update({
                "terminated": bool(env.termination_manager.terminated[0].item()),
                "time_out": bool(env.termination_manager.time_outs[0].item()),
                "reward_total": float(reward[0].item()),
                "reward_contributions": self.values(
                    self.manager._step_reward[0] * env.step_dt
                ),
            })
        acquisition = getattr(env, "_g1_v04_acquisition", None)
        if acquisition is not None:
            def trace_value(name):
                value = getattr(acquisition, name, None)
                return (
                    self.values(value[0])
                    if value is not None
                    else None
                )

            row.update({
                "sample_target_index": trace_value("sample_target_index"),
                "sample_preferred_foot": trace_value("sample_preferred_foot"),
                "target_step_after_update": trace_value("target_step"),
                "preferred_foot_after_update": trace_value("preferred_foot"),
                "step_over_target": trace_value(
                    "step_over_trace_target"
                ),
                "step_over_preferred_foot": trace_value(
                    "step_over_trace_preferred"
                ),
                "step_over_event_code": trace_value(
                    "step_over_trace_event_code"
                ),
                "preferred_transition_event_code": trace_value(
                    "preferred_transition_event_code"
                ),
                "preferred_transition_active": trace_value(
                    "preferred_transition_active"
                ),
                "preferred_transition_lifted": trace_value(
                    "preferred_transition_lifted"
                ),
                "preferred_transition_new_lift": trace_value(
                    "preferred_transition_new_lift"
                ),
                "clean_step_over_event": trace_value(
                    "clean_step_over_event"
                ),
                "clean_step_over_sequence_failed": trace_value(
                    "clean_step_over_sequence_failed"
                ),
                "sequence_active": trace_value(
                    "sequence_active"
                ),
                "sequence_first_direct": trace_value(
                    "sequence_first_direct"
                ),
                "sequence_failed": trace_value(
                    "sequence_failed"
                ),
                "sequence_clean_paid": trace_value(
                    "sequence_clean_paid"
                ),
                "step_over_lifted": trace_value(
                    "step_over_trace_lifted"
                ),
                "step_over_cleared": trace_value(
                    "step_over_trace_cleared"
                ),
                "step_over_clear_lift": trace_value(
                    "step_over_trace_clear_lift"
                ),
                "step_over_clear_riser": trace_value(
                    "step_over_trace_clear_riser"
                ),
                "step_over_old_contact": trace_value(
                    "step_over_trace_old_contact"
                ),
                "step_over_next_contact": trace_value(
                    "step_over_trace_next_contact"
                ),
                "step_over_consolidation": trace_value(
                    "step_over_trace_consolidation"
                ),
                "step_over_pref_force_n": trace_value(
                    "step_over_trace_pref_force_n"
                ),
                "step_over_pref_rear_x": trace_value(
                    "step_over_trace_pref_rear_x"
                ),
                "step_over_pref_front_x": trace_value(
                    "step_over_trace_pref_front_x"
                ),
                "step_over_pref_bottom_z": trace_value(
                    "step_over_trace_pref_bottom_z"
                ),
                "step_acquired_event": self.values(
                    acquisition.step_acquired_event[0]
                ),
                "acquired_target_index": self.values(
                    acquisition.acquired_target_index[0]
                ),
                "physical_touchdown_event": self.values(
                    acquisition.physical_touchdown_event[0]
                ),
                "riser_face_strike_sample": self.values(
                    acquisition.riser_face_strike_sample[0]
                ),
                "riser_strike_force_x": self.values(
                    acquisition.riser_strike_force_x[0]
                ),
                "crossing_attempt_invalid": self.values(
                    acquisition.crossing_attempt_invalid[0]
                ),
                "clearance_potential": trace_value(
                    "clearance_potential"
                ),
            })

        runtime = getattr(env, "_g1_geometry_episode", None)
        progress = getattr(runtime, "supported_ascent", None)
        if progress is not None:
            row["supported_ascent"] = {
                "next_step": int(progress.next_step[0].item()),
                "last_foot": int(progress.last_foot[0].item()),
                "pending_time_s": self.values(progress.pending_time[0]),
                "candidate_foot": int(progress.candidate_foot[0].item()),
                "transfer_time_s": float(progress.transfer_time[0].item()),
                "transfer_start_potential": float(
                    progress.transfer_start_potential[0].item()
                ),
                "motion_potential": float(progress.swing_potential[0].item()),
                "swing_potential": float(progress.swing_potential[0].item()),
                "swing_progress": float(progress.last_swing_progress[0].item()),
                "first": bool(progress.first[0].item()),
                "second": bool(progress.second[0].item()),
                "third": bool(progress.third[0].item()),
                "alternating_complete": bool(progress.alternating_complete[0].item()),
            }
        row["finish_dwell_s"] = (
            float(runtime.tracker.dwell[0].item())
            if runtime is not None else None
        )
        episode_runtime = getattr(env, "_g1_geometry_episode", None)
        strict = getattr(episode_runtime, "strict_gait", None)
        if strict is not None:
            from simulation.isaac_lab.g1_stairs.motion_reference import riser_contact_masks
            from simulation.isaac_lab.g1_stairs.terrain_state import riser_height_per_env
            # Recompute physical masks: strict classification freezes after failure.
            height = riser_height_per_env(env)
            strike, bridge = riser_contact_masks(
                bounds, self.sensor.data.net_forces_w[:, episode_runtime.sensor_feet],
                first_riser=strict.first, tread=strict.tread, steps=strict.steps,
                riser_height=height, active=height > 0,
                support_force_n=strict.SUPPORT_FORCE_N,
            )
            acquisition = getattr(env, "_g1_v04_acquisition", None)
            external = getattr(acquisition, "step_over_contract_violation", None)
            row["training_recovery"] = {
                "duration_s": episode_runtime.contract.training_recovery_s,
                "failure_latched": bool(episode_runtime.strict_failure_latched[0].item()),
                "elapsed_s": float(episode_runtime.recovery_elapsed_s[0].item()),
                "live_strike": bool(strike[0].item()),
                "live_bridge": bool(bridge[0].item()),
                "external_violation": bool(external[0].item()) if external is not None else False,
                "episode_reason": int(episode_runtime.tracker.reason[0].item()),
            }
            row["strict_stair_gait"] = {
                "acquisitions": int(strict.count[0].item()),
                "last_foot": int(strict.last_foot[0].item()),
                "failed": bool(strict.failed[0].item()),
                "reason": int(strict.reason[0].item()),
                "pending_foot": int(strict.pending[0].item()),
                "support_dwell_s": float(strict.dwell[0].item()),
            }

            decision = getattr(strict, "trace", None)
            if decision:
                row["strict_landing_decision"] = {
                    key: self.values(value[0])
                    for key, value in decision.items()
                }

        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def compute(self, dt):
        reward = self.original_compute(dt=dt)
        self.capture(reward)
        return reward

    def close(self):
        self.manager.compute = self.original_compute
        self.stream.close()
