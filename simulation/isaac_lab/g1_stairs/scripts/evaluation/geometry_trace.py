
"""Read-only state sampling after reward computation and before reset."""
import json
from pathlib import Path


class GeometryTrace:
    def __init__(self, env, outputs):
        """outputs: one directory, or one per robot (env index order)."""
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
            "reward_contributions": "weighted term values multiplied by step_dt",
        }
        outputs = [Path(outputs)] if isinstance(outputs, (str, Path)) else [Path(o) for o in outputs]
        heights = riser_height_per_env(env)
        self.streams = []
        for i, output in enumerate(outputs):
            entry = dict(metadata, env_index=i, riser_height_m=float(heights[i].item()))
            with (output / "trace_metadata.json").open("x") as stream:
                json.dump(entry, stream, indent=2)
            self.streams.append((output / "trace.jsonl").open("x", buffering=1))
        self.capture(None)
        self.manager.compute = self.compute

    @staticmethod
    def values(tensor):
        return tensor.detach().cpu().tolist()

    def capture(self, reward):
        for i, stream in enumerate(self.streams):
            stream.write(json.dumps(self.row(reward, i), allow_nan=False) + "\n")

    def row(self, reward, i):
        """One trace row for robot (env) i."""
        env = self.env
        data = self.robot.data
        row = {
            "sample": "initial" if reward is None else "post_physics_pre_reset",
            "global_step": int(env.common_step_counter),
            "episode_step": int(env.episode_length_buf[i].item()),
            "episode_time_s": float(env.episode_length_buf[i].item() * env.step_dt),
            "env_origin_w": self.values(env.scene.env_origins[i]),
            "root_pos_w": self.values(data.root_pos_w[i]),
            "root_quat_w": self.values(data.root_quat_w[i]),
            "root_lin_vel_w": self.values(data.root_lin_vel_w[i]),
            "root_ang_vel_w": self.values(data.root_ang_vel_w[i]),
            "command_base_velocity": self.values(
                env.command_manager.get_command("base_velocity")[i]
            ),
            "foot_pos_w": self.values(data.body_pos_w[i, self.foot_ids]),
            "foot_quat_w": self.values(data.body_quat_w[i, self.foot_ids]),
            "foot_lin_vel_w": self.values(data.body_lin_vel_w[i, self.foot_ids]),
            "foot_ang_vel_w": self.values(data.body_ang_vel_w[i, self.foot_ids]),
            "knee_pos_w": self.values(data.body_pos_w[i, self.knee_ids]),
            "knee_lin_vel_w": self.values(data.body_lin_vel_w[i, self.knee_ids]),
            "contact_net_forces_w": self.values(self.sensor.data.net_forces_w[i]),
            "joint_pos_rad": self.values(data.joint_pos[i]),
            "joint_vel_rad_s": self.values(data.joint_vel[i]),
            "applied_torque_nm": self.values(data.applied_torque[i]),
            "joint_pos_target_rad": self.values(data.joint_pos_target[i]),
            "action": self.values(
                env.action_manager.action[i, :29]
                if getattr(env.cfg, "gait_phase_enabled", False)
                else env.action_manager.action[i]
            ),
        }
        centre, bounds, surface_velocity = self.sole_state(
            data.body_pos_w[:, self.foot_ids] - env.scene.env_origins[:, None],
            data.body_quat_w[:, self.foot_ids],
            data.body_lin_vel_w[:, self.foot_ids],
            data.body_ang_vel_w[:, self.foot_ids],
        )
        row["motion_geometry"] = {
            "sole_centre_local": self.values(centre[i]),
            "sole_bounds_local": self.values(bounds[i]),
            "surface_velocity_w": self.values(surface_velocity[i]),
            "surface_speed_mps": self.values(surface_velocity.norm(dim=-1).amax(-1)[i]),
            "surface_slip_mps": self.values(surface_velocity[..., :2].norm(dim=-1).amax(-1)[i]),
        }
        row["knee_ang_vel_w"] = self.values(data.body_ang_vel_w[i, self.knee_ids])
        row["knee_quat_w"] = self.values(data.body_quat_w[i, self.knee_ids])
        history = getattr(self.sensor.data, "net_forces_w_history", None)
        if history is not None:
            row["contact_force_history_w"] = self.values(history[i])
        motion = getattr(env, "_g1_coordinated_motion", None)
        if motion is not None and reward is not None:
            row["coordinated_motion"] = {
                key: self.values(value[i]) for key, value in motion.trace.items()
            }
        if getattr(env.cfg, "gait_phase_enabled", False):
            gait = env.action_manager.get_term("gait_frequency")
            clock = gait.observation()[i]
            row.update({
                "raw_frequency_action": self.values(gait.raw_actions[i, 0]),
                "frequency_hz": self.values(gait.processed_actions[i, 0]),
                "phase": self.values(gait.phase[i]),
                "sin_phase": self.values(clock[0]),
                "cos_phase": self.values(clock[1]),
                "gait_active": self.values(gait.active[i]),
                "gait_reward_phase": self.values(gait.reward_phase[i]),
                "gait_reward_active": self.values(gait.reward_active[i]),
                "phase_swing_contact_loss": self.values(gait.contact_loss[i]),
                "approach_committed": self.values(gait.approach_committed[i]),
                "approach_clear_pending": self.values(gait.approach_clear_pending[i]),
            })
        if reward is not None:
            row.update({
                "terminated": bool(env.termination_manager.terminated[i].item()),
                "time_out": bool(env.termination_manager.time_outs[i].item()),
                "reward_total": float(reward[i].item()),
                "reward_contributions": self.values(
                    self.manager._step_reward[i] * env.step_dt
                ),
            })
        acquisition = getattr(env, "_g1_v04_acquisition", None)
        if acquisition is not None:
            def trace_value(name):
                value = getattr(acquisition, name, None)
                return (
                    self.values(value[i])
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
                    acquisition.step_acquired_event[i]
                ),
                "acquired_target_index": self.values(
                    acquisition.acquired_target_index[i]
                ),
                "physical_touchdown_event": self.values(
                    acquisition.physical_touchdown_event[i]
                ),
                "riser_face_strike_sample": self.values(
                    acquisition.riser_face_strike_sample[i]
                ),
                "riser_strike_force_x": self.values(
                    acquisition.riser_strike_force_x[i]
                ),
                "crossing_attempt_invalid": self.values(
                    acquisition.crossing_attempt_invalid[i]
                ),
                "clearance_potential": trace_value(
                    "clearance_potential"
                ),
            })

        runtime = getattr(env, "_g1_geometry_episode", None)
        progress = getattr(runtime, "supported_ascent", None)
        if progress is not None:
            row["supported_ascent"] = {
                "next_step": int(progress.next_step[i].item()),
                "last_foot": int(progress.last_foot[i].item()),
                "pending_time_s": self.values(progress.pending_time[i]),
                "candidate_foot": int(progress.candidate_foot[i].item()),
                "transfer_blocked": bool(progress.transfer_blocked[i].item()),
                "transfer_time_s": float(progress.transfer_time[i].item()),
                "transfer_start_potential": float(
                    progress.transfer_start_potential[i].item()
                ),
                "motion_potential": float(progress.swing_potential[i].item()),
                "swing_potential": float(progress.swing_potential[i].item()),
                "swing_progress": float(progress.last_swing_progress[i].item()),
                "first": bool(progress.first[i].item()),
                "second": bool(progress.second[i].item()),
                "third": bool(progress.third[i].item()),
                "alternating_complete": bool(progress.alternating_complete[i].item()),
            }
        row["finish_dwell_s"] = (
            float(runtime.tracker.dwell[i].item())
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
                "failure_latched": bool(episode_runtime.strict_failure_latched[i].item()),
                "elapsed_s": float(episode_runtime.recovery_elapsed_s[i].item()),
                "live_strike": bool(strike[i].item()),
                "live_bridge": bool(bridge[i].item()),
                "external_violation": bool(external[i].item()) if external is not None else False,
                "episode_reason": int(episode_runtime.tracker.reason[i].item()),
            }
            row["strict_stair_gait"] = {
                "acquisitions": int(strict.count[i].item()),
                "last_foot": int(strict.last_foot[i].item()),
                "failed": bool(strict.failed[i].item()),
                "reason": int(strict.reason[i].item()),
                "pending_foot": int(strict.pending[i].item()),
                "support_dwell_s": float(strict.dwell[i].item()),
            }

            decision = getattr(strict, "trace", None)
            if decision:
                row["strict_landing_decision"] = {
                    key: self.values(value[i])
                    for key, value in decision.items()
                }


        coordinator = getattr(env, "_g1_footstep_coordinator", None)
        if coordinator is not None:
            from simulation.isaac_lab.g1_stairs.footstep_isaac_adapter import course_tensors
            from simulation.isaac_lab.g1_stairs.footstep_tensor import NOMINAL_FOOT_SEPARATION_M

            runtime = coordinator.runtime
            flat = bool(getattr(env.cfg, "flat_ground_footsteps", False))
            course = course_tensors(
                env,
                foot_separation_m=NOMINAL_FOOT_SEPARATION_M,
                flat_ground=flat,
            )
            targets, feet, valid = runtime.targets(**course, flat_ground=flat)

            advanced = getattr(env, "_g1_footstep_last_advanced", None)

            row["planned_footsteps"] = {
                "target_index": int(runtime.target_index[i].item()),
                "assigned_foot_index": int(feet[i, 0].item()),
                "target_xyz": self.values(targets[i, 0, :3]),
                "target_valid": bool(valid[i, 0].item()),
                "first_foot_index": int(runtime.first_foot[i].item()),
                "completed": int(runtime.completed[i].item()),
                "attempted": int(runtime.attempted[i].item()),
                "advanced_this_step": (
                    bool(advanced[i].item())
                    if advanced is not None else None
                ),
            }


        rolling = getattr(env, "_g1_rolling_clock", None)
        if rolling is not None:
            from simulation.isaac_lab.g1_stairs.footstep_clock import clock_schedule

            swing, assigned = clock_schedule(rolling.phase)
            rows = getattr(env, "_g1_rolling_rows", None)
            row["footstep_rolling_clock"] = {
                "rolling_row": (bool(rows[i].item()) if rows is not None else None),
                "phase": float(rolling.phase[i].item()),
                "period_s": float(rolling.period_s[i].item()),
                "period_scale": float(rolling.period_scale[i].item()),
                "walking": bool(rolling.active[i].item()),
                "scheduled_swing_lr": self.values(swing[i]),
                "assigned_foot": int(assigned[i].item()),
                "window": int(rolling.window[i].item()),
                "acquired": bool(rolling.acquired[i].item()),
                "target_xyz": self.values(rolling.targets[i, 0, :3]),
                "next_target_xyz": self.values(rolling.targets[i, 1, :3]),
                "cycle_yaw_error_radps": float(rolling.cycle_yaw_error[i].item()),
                "on_stair_plan": bool(rolling.on_plan[i].item()),
                "stairs_climbed": bool(rolling.climbed[i].item()),
                "swing_landed_lr": self.values(rolling.landed[i]),
            }

        coordinator = getattr(env, "_g1_footstep_coordinator", None)
        if coordinator is not None:
            from simulation.isaac_lab.g1_stairs.footstep_isaac_adapter import (
                course_tensors,
            )
            from simulation.isaac_lab.g1_stairs.footstep_tensor import NOMINAL_FOOT_SEPARATION_M

            flat = bool(getattr(env.cfg, "flat_ground_footsteps", False))
            course = course_tensors(
                env,
                foot_separation_m=NOMINAL_FOOT_SEPARATION_M,
                flat_ground=flat,
            )
            targets, feet, valid = coordinator.runtime.targets(**course, flat_ground=flat)

            phase, swing, stance = coordinator.clock.features(
                target_foot=feet[:, 0],
                valid=valid[:, 0],
            )

            sensor_ids = [
                self.sensor.body_names.index(name)
                for name in self.feet
            ]

            force_z = self.sensor.data.net_forces_w[
                0, sensor_ids, 2
            ]

            def cached(name):
                tensor = getattr(env, name, None)
                return (
                    float(tensor[i].item())
                    if tensor is not None else None
                )

            row["footstep_clock_audit"] = {
                "elapsed_s": float(
                    coordinator.clock.elapsed_s[i].item()
                ),
                "period_s": coordinator.clock.step_period_s,
                "sin_cos": self.values(phase[i]),
                "scheduled_swing_lr": self.values(swing[i]),
                "scheduled_stance_lr": self.values(stance[i]),
                "assigned_target_foot": int(feet[i, 0].item()),
                "target_index": int(
                    coordinator.runtime.target_index[i].item()
                ),
                "target_xyz": self.values(targets[i, 0, :3]),
                "target_valid": bool(valid[i, 0].item()),
                "vertical_force_lr_n": self.values(force_z),
                "supported_lr_30n": self.values(force_z >= 30.0),
                "foot_speed_lr_mps": self.values(
                    data.body_lin_vel_w[i, self.foot_ids].norm(dim=-1)
                ),
                "cached_stance_reward": cached(
                    "_g1_footstep_stance_support"
                ),
                "cached_swing_reward": cached(
                    "_g1_footstep_directed_swing"
                ),
                "cached_swing_clearance": cached(
                    "_g1_footstep_swing_clearance"
                ),
                "cached_off_beat_steps": cached(
                    "_g1_footstep_off_beat"
                ),
                "cached_landing_impact": cached(
                    "_g1_footstep_landing_impact"
                ),
                "acquisition_paid": (
                    bool(env._g1_footstep_acquisition_paid[i].item())
                    if hasattr(env, "_g1_footstep_acquisition_paid")
                    else None
                ),
                "qualified_acquisition": (
                    bool(env._g1_footstep_last_advanced[i].item())
                    if hasattr(env, "_g1_footstep_last_advanced")
                    else None
                ),
            }

        landing = getattr(env, "_g1_footstep_landing_audit", None)
        if landing is not None and reward is not None:
            row["footstep_landing_audit"] = {
                key: self.values(value[i])
                for key, value in landing.items()
            }

        return row

    def compute(self, dt):
        reward = self.original_compute(dt=dt)
        self.capture(reward)
        return reward

    def close(self):
        self.manager.compute = self.original_compute
        for stream in self.streams:
            stream.close()
