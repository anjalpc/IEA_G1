"""Live Isaac Lab/RSL-RL metrics bridge for the active stair task."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from simulation.isaac_lab.g1_stairs.action_diagnostics import (
    action_group_mean_abs,
)
from simulation.isaac_lab.g1_stairs.evaluation.metrics import RunMetricsWriter
from simulation.isaac_lab.g1_stairs.contact_sensor_contract import (
    FORBIDDEN_CONTACT_BODY_NAMES,
    resolve_available_body_indices,
)
from simulation.isaac_lab.g1_stairs.evaluation.stair_diagnostics import (
    NamedTensorStatsAccumulator,
)
from simulation.isaac_lab.g1_stairs.terrain_state import stair_cfg


def recover_raw_reward_step(weighted_rate, dt: float, weight: float):
    """Undo RewardManager weighting when the raw value is identifiable."""
    if weight == 0.0:
        return None
    # Isaac Lab stores _step_reward as raw_term * weight (the manager
    # divides the dt-scaled reward by dt before exposing this buffer).
    return weighted_rate / weight


class InstrumentedRslRlVecEnvWrapper:
    """Proxy an RSL-RL wrapper while accumulating rollout diagnostics."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._last_observations = None
        self._previous_actions = None
        self._previous_executed_actions = None
        self._previous_joint_acc = None

        self.reset_metrics()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def get_observations(self):
        observations = self._wrapped.get_observations()
        self._last_observations = observations
        return observations

    def reset_metrics(self) -> None:
        self._steps = 0
        self._reward_total = 0.0
        self._reward_components: dict[str, float] = {}
        self._action_rate = 0.0

        self._action_group_abs_sum = {
            "legs": 0.0,
            "arms": 0.0,
            "waist": 0.0,
        }
        self._executed_action_group_abs_sum = {
            "legs": 0.0,
            "arms": 0.0,
            "waist": 0.0,
        }
        self._executed_action_rate = 0.0

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

        # Distribution diagnostics are separate from the historical
        # reward-component accumulator. They never feed back into PPO.
        self._reward_distribution = None
        self._reward_distribution_weights = {}

    def step(self, actions):
        import torch

        executed_actions = actions

        result = self._wrapped.step(
            executed_actions
        )
        observations, rewards, dones, extras = result
        self._last_observations = observations
        env = self.unwrapped
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
            resolved = resolve_available_body_indices(
                sensor.body_names,
                FORBIDDEN_CONTACT_BODY_NAMES,
            )
            indices = list(resolved.values())
            if indices:
                forces = sensor.data.net_forces_w[:, indices]
                contacts = (
                    torch.linalg.vector_norm(
                        forces,
                        dim=2,
                    )
                    .ge(5.0)
                    .any(dim=1)
                )
                self._forbidden_contact_count += int(
                    torch.count_nonzero(
                        contacts
                    ).item()
                )

        manager = env.reward_manager

        reward_names = []
        reward_columns = []

        for index, name in enumerate(manager.active_terms):
            cfg = manager.get_term_cfg(name)
            weight = float(cfg.weight)
            weighted_rate = manager._step_reward[:, index]

            raw_step = recover_raw_reward_step(
                weighted_rate,
                dt,
                weight,
            )

            if raw_step is None:
                # RewardManager has already erased the raw value by
                # multiplying it by zero, so it cannot be reconstructed.
                continue

            self._reward_components[name] = (
                self._reward_components.get(name, 0.0)
                + float(raw_step.mean().item())
            )

            reward_names.append(name)
            reward_columns.append(raw_step)
            self._reward_distribution_weights[
                name
            ] = weight

        if reward_columns:
            reward_names = tuple(
                reward_names
            )

            values = torch.stack(
                reward_columns,
                dim=1,
            )

            if self._reward_distribution is None:
                self._reward_distribution = (
                    NamedTensorStatsAccumulator(
                        reward_names
                    )
                )

            if (
                self._reward_distribution.names
                != reward_names
            ):
                raise RuntimeError(
                    "active reward term order changed "
                    "during one training iteration"
                )

            self._reward_distribution.update(
                values
            )

        group_action_abs = action_group_mean_abs(
            actions
        )

        for name, value in (
            group_action_abs.items()
        ):
            self._action_group_abs_sum[
                name
            ] += float(value.item())

        executed_group_abs = action_group_mean_abs(
            executed_actions
        )
        for name, value in executed_group_abs.items():
            self._executed_action_group_abs_sum[
                name
            ] += float(value.item())

        if self._previous_executed_actions is not None:
            self._executed_action_rate += float(
                torch.mean(
                    torch.square(
                        executed_actions
                        - self._previous_executed_actions
                    )
                ).item()
            )
        self._previous_executed_actions = (
            executed_actions.detach().clone()
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
        # Report the same local-X coordinate used by the stair detector.
        # The old +1.0 presentation offset made a robot at local x=0.30 look
        # as if it had already crossed the 1.0 m riser plane.
        self._maximum_x = max(
            self._maximum_x,
            float(local_x.max().item()),
        )
        self._maximum_height = max(
            self._maximum_height, float(local_z.max().item())
        )
        terrain = stair_cfg(env)
        first_riser = float(terrain.start_clearance)
        tread_depth = float(terrain.tread_depth)
        step_count = int(terrain.step_count)
        maximum_x = float(local_x.max().item())
        level = sum(
            maximum_x >= first_riser + index * tread_depth
            for index in range(step_count)
        )
        self._maximum_step = max(self._maximum_step, level)
        platform_start = first_riser + step_count * tread_depth
        self._platform_reached = self._platform_reached or bool(
            torch.any(local_x >= platform_start).item()
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
            "action_legs_abs_mean": (
                self._action_group_abs_sum[
                    "legs"
                ]
                / steps
            ),
            "action_arms_abs_mean": (
                self._action_group_abs_sum[
                    "arms"
                ]
                / steps
            ),
            "action_waist_abs_mean": (
                self._action_group_abs_sum[
                    "waist"
                ]
                / steps
            ),
            "executed_action_legs_abs_mean": (
                self._executed_action_group_abs_sum["legs"]
                / steps
            ),
            "executed_action_arms_abs_mean": (
                self._executed_action_group_abs_sum["arms"]
                / steps
            ),
            "executed_action_waist_abs_mean": (
                self._executed_action_group_abs_sum["waist"]
                / steps
            ),
            "executed_action_rate_mean": (
                self._executed_action_rate / steps
            ),
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
        reward_diagnostics = {}

        if self._reward_distribution is not None:
            reward_stats = (
                self._reward_distribution.snapshot()
            )

            for name, values in reward_stats.items():
                weight = (
                    self._reward_distribution_weights[
                        name
                    ]
                )

                reward_diagnostics[name] = {
                    "sample_count": (
                        values["sample_count"]
                    ),
                    "mean": values["mean"],
                    "std": values["std"],
                    "min": values["min"],
                    "max": values["max"],
                    "nonzero_fraction": (
                        values[
                            "nonzero_fraction"
                        ]
                    ),
                    "positive_fraction": (
                        values[
                            "positive_fraction"
                        ]
                    ),
                    "negative_fraction": (
                        values[
                            "negative_fraction"
                        ]
                    ),
                    "weight": weight,
                    "weighted_mean": (
                        values["mean"]
                        * weight
                    ),
                    "weighted_abs_mean": (
                        values["mean_abs"]
                        * abs(weight)
                    ),
                }

        stair_diagnostics = {}

        self.reset_metrics()

        return (
            snapshot,
            rewards,
            reward_diagnostics,
            stair_diagnostics,
        )


class MetricsOnPolicyRunnerMixin:
    """Persist one structured metrics row after every RSL-RL iteration."""

    def configure_metrics(self, output_directory: Path, run_id: str) -> None:
        self._metrics_writer = RunMetricsWriter(output_directory, run_id)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        print("[POST_UPDATE] base_log ENTER", flush=True)
        super().log(locs, width=width, pad=pad)
        print("[POST_UPDATE] base_log EXIT", flush=True)

        print("[POST_UPDATE] consume_metrics ENTER", flush=True)
        (
            snapshot,
            rewards,
            reward_diagnostics,
            stair_diagnostics,
        ) = self.env.consume_metrics()
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
        print("[POST_UPDATE] metrics_write ENTER", flush=True)

        self._metrics_writer.record_training_snapshot(
            snapshot,
            rewards,
        )

        self._metrics_writer.record_reward_diagnostics(
            iteration=snapshot["iteration"],
            total_simulated_steps=snapshot[
                "total_simulated_steps"
            ],
            diagnostics=reward_diagnostics,
        )

        self._metrics_writer.record_stair_diagnostics(
            iteration=snapshot["iteration"],
            total_simulated_steps=snapshot[
                "total_simulated_steps"
            ],
            diagnostics=stair_diagnostics,
        )

        stair_event_counts = {
            key.removeprefix("stair_event_"): value
            for key, value in snapshot.items()
            if key.startswith("stair_event_")
        }
        self._metrics_writer.record_stair_event_counts(
            iteration=snapshot["iteration"],
            total_simulated_steps=snapshot["total_simulated_steps"],
            counts=stair_event_counts,
        )

        stair_episode_counts = {
            key.removeprefix("stair_episode_"): value
            for key, value in snapshot.items()
            if key.startswith("stair_episode_")
        }
        self._metrics_writer.record_stair_episode_counts(
            iteration=snapshot["iteration"],
            total_simulated_steps=snapshot["total_simulated_steps"],
            counts=stair_episode_counts,
        )

        print("[POST_UPDATE] metrics_write EXIT", flush=True)
