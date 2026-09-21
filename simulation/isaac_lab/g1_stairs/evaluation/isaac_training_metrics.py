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

    def step(self, actions):
        import torch

        result = self._wrapped.step(actions)
        _, rewards, _, _ = result
        env = self.unwrapped
        robot = env.scene["robot"]
        data = robot.data
        dt = float(env.step_dt)
        self._steps += 1
        self._reward_total += float(rewards.mean().item())

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
        }
        rewards = {}
        manager = self.unwrapped.reward_manager
        for name, raw in self._reward_components.items():
            rewards[name] = {
                "raw": raw,
                "weight": float(manager.get_term_cfg(name).weight),
            }
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
