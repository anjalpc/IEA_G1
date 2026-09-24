"""Live Isaac Lab/RSL-RL metrics bridge for Experiment A."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from simulation.isaac_lab.g1_stairs.evaluation.metrics import RunMetricsWriter


def recover_raw_reward_step(weighted_rate, dt: float, weight: float):
    """Undo RewardManager weighting when the raw value is identifiable."""
    if weight == 0.0:
        return None
    return weighted_rate * dt / weight


class InstrumentedRslRlVecEnvWrapper:
    """Proxy an RSL-RL wrapper while accumulating rollout diagnostics."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._approach_teacher = None
        self._approach_teacher_cfg = None
        self._previous_actions = None
        self._previous_joint_acc = None
        self.reset_metrics()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def reset_metrics(self) -> None:
        self._steps = 0
        self._reward_total = 0.0
        self._reward_components: dict[str, float] = {}
        self._action_rate = 0.0
        self._joint_jerk = 0.0
        self._mechanical_power = 0.0
        self._action_saturation_count = 0
        self._joint_limit_violation_count = 0
        self._velocity_limit_violation_count = 0
        self._torque_limit_violation_count = 0
        self._nan_count = 0
        self._maximum_x = -math.inf
        self._maximum_height = -math.inf
        self._maximum_step = 0
        self._platform_reached = False
        self._fall_count = 0
        self._forbidden_contact_count = 0
        self._finite_action_env_steps = 0
        self._physical043_counts = {}
        self._teacher_gate_sum = 0.0
        self._teacher_error_sum = 0.0
        self._teacher_reward_sum = 0.0

    def configure_approach_teacher(self, teacher, config) -> None:
        self._approach_teacher = teacher
        self._approach_teacher_cfg = config

    def step(self, actions):
        import torch

        teacher_raw = None
        teacher_gate = None
        if self._approach_teacher is not None:
            from simulation.isaac_lab.g1_stairs.approach_teacher import (
                smooth_l1_per_environment,
                smooth_release_gate,
                teacher_observation,
            )
            current = self._wrapped.get_observations()
            locomotion_obs = teacher_observation(current)
            with torch.no_grad():
                target = self._approach_teacher(locomotion_obs)

            env = self.unwrapped
            root_x = (
                env.scene["robot"].data.root_pos_w[:, 0]
                - env.scene.env_origins[:, 0]
            )
            stair = (
                env.cfg.scene.terrain.terrain_generator
                .sub_terrains["three_step_ascent"]
            )
            distance = float(stair.start_clearance) - root_x
            teacher_gate = smooth_release_gate(
                distance,
                self._approach_teacher_cfg.release_near_m,
                self._approach_teacher_cfg.release_far_m,
            )
            teacher_error = smooth_l1_per_environment(
                actions,
                target,
                self._approach_teacher_cfg.huber_delta,
            )
            teacher_raw = -teacher_gate * teacher_error

        result = self._wrapped.step(actions)
        observations, rewards, dones, extras = result
        if teacher_raw is not None:
            weighted = (
                self._approach_teacher_cfg.coefficient
                * teacher_raw
                * float(self.unwrapped.step_dt)
            )
            rewards = rewards + weighted
            result = (observations, rewards, dones, extras)
            self._teacher_gate_sum += float(teacher_gate.mean().item())
            self._teacher_error_sum += float((-teacher_raw).mean().item())
            self._teacher_reward_sum += float(
                (teacher_raw * float(self.unwrapped.step_dt)).mean().item()
            )
        env = self.unwrapped
        physical_runtime = getattr(env, "_physical_step043", None)
        if physical_runtime is not None:
            physical = physical_runtime.snapshot
            # Already computed before auto-reset; never recompute from reset poses.
            for name, value in physical["measurements"].items():
                key = "step043_" + name
                self._physical043_counts[key] = self._physical043_counts.get(key, 0) + int(value.sum().item())
            for name, value in (("episodes_finished", dones),
                                ("successful_episodes", dones & physical["success"]),
                                ("failed_episodes", dones & physical["failure"])):
                key = "step043_" + name
                self._physical043_counts[key] = self._physical043_counts.get(key, 0) + int(value.sum().item())
        robot = env.scene["robot"]
        data = robot.data
        dt = float(env.step_dt)
        self._steps += 1
        self._reward_total += float(rewards.mean().item())
        self._finite_action_env_steps += int(
            torch.count_nonzero(
                torch.isfinite(actions).reshape(actions.shape[0], -1).all(dim=1)
            ).item()
        )

        timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
        if timeouts is None:
            timeouts = torch.zeros_like(dones)
        self._fall_count += int(torch.count_nonzero(dones & ~timeouts).item())

        sensor = env.scene.sensors.get("contact_forces")
        if sensor is not None:
            forbidden_names = (
                "pelvis", "torso_link", "left_knee_link", "right_knee_link",
                "left_elbow_link", "right_elbow_link", "left_rubber_hand",
                "right_rubber_hand",
            )
            indices = [sensor.body_names.index(name) for name in forbidden_names]
            forces = sensor.data.net_forces_w[:, indices]
            contacts = torch.linalg.vector_norm(forces, dim=2).ge(5.0).any(dim=1)
            self._forbidden_contact_count += int(torch.count_nonzero(contacts).item())

        manager = env.reward_manager
        for index, name in enumerate(manager.active_terms):
            cfg = manager.get_term_cfg(name)
            weighted_rate = manager._step_reward[:, index]
            raw_step = recover_raw_reward_step(
                weighted_rate,
                dt,
                float(cfg.weight),
            )
            if raw_step is None:
                # RewardManager has already erased the raw value by
                # multiplying it by zero, so it cannot be reconstructed.
                continue
            self._reward_components[name] = (
                self._reward_components.get(name, 0.0)
                + float(raw_step.mean().item())
            )

        if self.clip_actions is not None:
            self._action_saturation_count += int(
                torch.count_nonzero(torch.abs(actions) > self.clip_actions).item()
            )
        if self._previous_actions is not None:
            self._action_rate += float(
                torch.mean(torch.square(actions - self._previous_actions)).item()
            )
        self._previous_actions = actions.detach().clone()

        if self._previous_joint_acc is not None:
            self._joint_jerk += float(
                torch.mean(torch.abs(data.joint_acc - self._previous_joint_acc) / dt).item()
            )
        self._previous_joint_acc = data.joint_acc.detach().clone()
        self._mechanical_power += float(
            torch.mean(torch.sum(torch.abs(data.applied_torque * data.joint_vel), dim=1)).item()
        )

        pos_limits = data.soft_joint_pos_limits
        self._joint_limit_violation_count += int(
            torch.count_nonzero(
                (data.joint_pos < pos_limits[:, :, 0])
                | (data.joint_pos > pos_limits[:, :, 1])
            ).item()
        )
        self._velocity_limit_violation_count += int(
            torch.count_nonzero(
                torch.abs(data.joint_vel) > data.soft_joint_vel_limits
            ).item()
        )
        self._torque_limit_violation_count += int(
            torch.count_nonzero(
                torch.abs(data.applied_torque) > data.joint_effort_limits
            ).item()
        )

        tensors = (
            actions,
            rewards,
            data.root_pos_w,
            data.root_quat_w,
            data.joint_pos,
            data.joint_vel,
            data.joint_acc,
            data.applied_torque,
        )
        self._nan_count += sum(
            int(torch.count_nonzero(~torch.isfinite(value)).item())
            for value in tensors
        )

        local_x = data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        local_z = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        course_x = local_x + 1.0
        self._maximum_x = max(self._maximum_x, float(course_x.max().item()))
        self._maximum_height = max(
            self._maximum_height, float(local_z.max().item())
        )
        level = sum(float(local_x.max().item()) >= edge for edge in (1.0, 1.25, 1.50))
        self._maximum_step = max(self._maximum_step, level)
        self._platform_reached = self._platform_reached or bool(
            torch.any(local_x >= 1.75).item()
        )
        return result

    def consume_metrics(self) -> tuple[dict, dict[str, dict[str, float]]]:
        steps = max(self._steps, 1)
        snapshot = {
            "episode_return_mean": self._reward_total,
            "maximum_x_mean_m": self._maximum_x,
            "maximum_base_height_mean_m": self._maximum_height,
            "maximum_step_reached_mean": self._maximum_step,
            "platform_reached_rate": float(self._platform_reached),
            "action_saturation_count": self._action_saturation_count,
            "joint_limit_violation_count": self._joint_limit_violation_count,
            "velocity_limit_violation_count": self._velocity_limit_violation_count,
            "torque_limit_violation_count": self._torque_limit_violation_count,
            "action_rate_mean": self._action_rate / steps,
            "joint_jerk_mean": self._joint_jerk / steps,
            "mechanical_power_mean": self._mechanical_power / steps,
            "nan_count": self._nan_count,
            "fall_rate": self._fall_count / (steps * self.num_envs),
            "forbidden_contact_rate": (
                self._forbidden_contact_count / (steps * self.num_envs)
            ),
            "finite_action_rate": (
                self._finite_action_env_steps / (steps * self.num_envs)
            ),
        }
        rewards = {}
        manager = self.unwrapped.reward_manager
        for name, raw in self._reward_components.items():
            rewards[name] = {
                "raw": raw,
                "weight": float(manager.get_term_cfg(name).weight),
            }
        if self._approach_teacher is not None:
            snapshot["approach_teacher_gate_mean"] = self._teacher_gate_sum / steps
            snapshot["approach_teacher_error_mean"] = self._teacher_error_sum / steps
            rewards["approach_teacher_prior"] = {
                "raw": self._teacher_reward_sum,
                "weight": self._approach_teacher_cfg.coefficient,
            }
        snapshot["base_above_rate"] = self._reward_components.get(
            "base_above_transition", 0.0
        )
        snapshot["both_feet_supported_rate"] = self._reward_components.get(
            "both_feet_supported_transition", 0.0
        )
        snapshot["completion_rate"] = self._reward_components.get(
            "completed_transition", 0.0
        )
        if self._physical043_counts:
            # Historical fields below are maxima/step frequencies, NOT their
            # advertised episode means/rates. Do not mislabel them for 043.
            for name in ("maximum_x_mean_m", "maximum_base_height_mean_m",
                         "maximum_step_reached_mean", "platform_reached_rate",
                         "fall_rate", "forbidden_contact_rate", "base_above_rate",
                         "both_feet_supported_rate", "completion_rate"):
                snapshot.pop(name, None)
            snapshot.update(self._physical043_counts)
            snapshot["step043_environment_steps"] = steps * self.num_envs
            finished = self._physical043_counts.get("step043_episodes_finished", 0)
            if finished:
                snapshot["success_rate"] = self._physical043_counts.get("step043_successful_episodes", 0) / finished
            else:
                snapshot["success_rate"] = None  # Not zero successes out of zero episodes.
        self.reset_metrics()
        return snapshot, rewards


class MetricsOnPolicyRunnerMixin:
    """Persist one structured metrics row after every RSL-RL iteration."""

    def configure_metrics(self, output_directory: Path, run_id: str) -> None:
        self._metrics_writer = RunMetricsWriter(output_directory, run_id)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        print("[POST_UPDATE] base_log ENTER", flush=True)
        super().log(locs, width=width, pad=pad)
        print("[POST_UPDATE] base_log EXIT", flush=True)

        print("[POST_UPDATE] consume_metrics ENTER", flush=True)
        snapshot, rewards = self.env.consume_metrics()
        if getattr(self, "writer", None) is not None:
            for name, value in snapshot.items():
                if name.startswith("step043_"):
                    self.writer.add_scalar("PhysicalStep043/" + name, value, locs["it"])
        print("[POST_UPDATE] consume_metrics EXIT", flush=True)
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        duration = locs["collection_time"] + locs["learn_time"]
        loss = locs["loss_dict"]
        snapshot.update(
            {
                "iteration": int(locs["it"]),
                "total_simulated_steps": int(self.tot_timesteps),
                "wall_time_s": float(self.tot_time),
                "episode_return_mean": (
                    statistics.mean(locs["rewbuffer"])
                    if locs["rewbuffer"]
                    else snapshot["episode_return_mean"]
                ),
                "episode_length_mean": (
                    statistics.mean(locs["lenbuffer"])
                    if locs["lenbuffer"]
                    else None
                ),
                "policy_loss": float(loss.get("surrogate", 0.0)),
                "value_loss": float(loss.get("value_function", 0.0)),
                "entropy": float(loss.get("entropy", 0.0)),
                "learning_rate": float(self.alg.learning_rate),
                "simulation_fps": float(collection_size / duration),
                "environment_steps_per_second": float(collection_size / duration),
            }
        )
        try:
            import torch

            snapshot["gpu_memory_used_mib"] = float(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            )
        except Exception:
            snapshot["gpu_memory_used_mib"] = None
        print("[POST_UPDATE] metrics_write ENTER", flush=True)
        self._metrics_writer.record_training_snapshot(snapshot, rewards)
        print("[POST_UPDATE] metrics_write EXIT", flush=True)
