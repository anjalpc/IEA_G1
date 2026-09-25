"""Live Isaac Lab/RSL-RL metrics bridge for Experiment A."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from simulation.isaac_lab.g1_stairs.approach_teacher import (
    action_group_mean_abs,
)
from simulation.isaac_lab.g1_stairs.evaluation.metrics import RunMetricsWriter
from simulation.isaac_lab.g1_stairs.evaluation.stair_diagnostics import (
    NamedTensorStatsAccumulator,
    StairEventDiagnostics,
    physical_diagnostic_matrix,
)
from simulation.isaac_lab.g1_stairs.step_up_isaac import (
    FORBIDDEN_CONTACT_BODY_NAMES,
)


def recover_raw_reward_step(weighted_rate, dt: float, weight: float):
    """Undo RewardManager weighting when the raw value is identifiable."""
    if weight == 0.0:
        return None
    return weighted_rate * dt / weight


def resolve_available_body_indices(
    available_names,
    requested_names=FORBIDDEN_CONTACT_BODY_NAMES,
):
    """Resolve only contact bodies represented by the active sensor asset.

    Fixed bodies without collision shapes can legitimately be absent from a
    ContactSensor body inventory. Metrics must therefore follow the runtime
    sensor contract rather than assuming every articulation body is tracked.
    """
    index_by_name = {
        name: index
        for index, name in enumerate(available_names)
    }
    return [
        index_by_name[name]
        for name in requested_names
        if name in index_by_name
    ]


class InstrumentedRslRlVecEnvWrapper:
    """Proxy an RSL-RL wrapper while accumulating rollout diagnostics."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._teacher_action_blend = None
        self._teacher_action_blend_cfg = None
        self._last_observations = None
        self._previous_actions = None
        self._previous_executed_actions = None
        self._previous_joint_acc = None

        # Episode-local stair-event state must survive PPO iteration
        # boundaries. Only its iteration aggregates are consumed/reset.
        self._stair_event_diagnostics = None

        self.reset_metrics()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def get_observations(self):
        observations = self._wrapped.get_observations()
        self._last_observations = observations
        return observations

    def configure_teacher_action_blend(
        self,
        teacher,
        config,
    ) -> None:
        if teacher is None:
            raise RuntimeError(
                "teacher action blend requires frozen Unitree teacher"
            )

        self._teacher_action_blend = teacher
        self._teacher_action_blend_cfg = config

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
        self._teacher_execution_gate_sum = 0.0
        self._teacher_execution_gate_samples = 0

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
        self._structured044_counts = {}
        self._structured044_ascent_distance_sum = 0.0
        self._structured044_ascent_distance_count = 0

        # Distribution diagnostics are separate from the historical
        # reward-component accumulator. They never feed back into PPO.
        self._reward_distribution = None
        self._reward_distribution_weights = {}
        self._stair_distribution = None

    def step(self, actions):
        import torch

        executed_actions = actions
        execution_gate = None

        if self._teacher_action_blend is not None:
            from simulation.isaac_lab.g1_stairs.approach_teacher import (
                blend_teacher_student_actions,
                teacher_gate,
                teacher_observation,
            )

            current = self._last_observations
            if current is None:
                # Initialization fallback only. Normal rollouts cache exactly
                # the observations returned to RSL-RL for this action.
                current = self.get_observations()

            teacher_obs = teacher_observation(
                current
            )

            gate = teacher_gate(
                current
            )
            execution_gate = gate

            with torch.no_grad():
                teacher_actions = (
                    self._teacher_action_blend(
                        teacher_obs
                    )
                )

            executed_actions = (
                blend_teacher_student_actions(
                    actions,
                    teacher_actions,
                    gate,
                )
            )

        result = self._wrapped.step(
            executed_actions
        )
        observations, rewards, dones, extras = result
        self._last_observations = observations
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

            diagnostic_payload = physical.get("diagnostics")

            if diagnostic_payload is not None:
                names, values = physical_diagnostic_matrix(
                    diagnostic_payload
                )

                if self._stair_distribution is None:
                    self._stair_distribution = (
                        NamedTensorStatsAccumulator(
                            names
                        )
                    )

                if (
                    self._stair_distribution.names
                    != names
                ):
                    raise RuntimeError(
                        "physical diagnostic signal order changed "
                        "during one training iteration"
                    )

                self._stair_distribution.update(
                    values
                )

                if self._stair_event_diagnostics is None:
                    physical_config = (
                        physical_runtime.config
                    )

                    self._stair_event_diagnostics = (
                        StairEventDiagnostics(
                            env.num_envs,
                            env.device,
                            first_riser_m=(
                                physical_config.first_riser_m
                            ),
                            tread_m=(
                                physical_config.tread_m
                            ),
                            height_m=(
                                physical_config.height_m
                            ),
                            takeoff_distance_m=(
                                physical_config.takeoff_distance_m
                            ),
                        )
                    )

                self._stair_event_diagnostics.update(
                    diagnostic_payload,
                    dt=float(env.step_dt),
                )

                timeouts = (
                    extras.get("time_outs")
                    if isinstance(extras, dict)
                    else None
                )

                if timeouts is None:
                    timeouts = torch.zeros_like(
                        dones
                    )

                # RslRlVecEnvWrapper returns dones as torch.long
                # for RSL-RL compatibility. Convert once here so both
                # terminal classification and episode-local reset use
                # identical boolean-mask semantics.
                done_mask = dones.to(
                    dtype=torch.bool
                )

                # Classify completed episodes from the immutable
                # PRE-AUTORESET physical snapshot before clearing any
                # episode-local diagnostic evidence.
                self._stair_event_diagnostics.observe_terminations(
                    done_mask,
                    physical_failure=physical["failure"],
                    physical_success=physical["success"],
                    stage=physical["stage"],
                    timeouts=timeouts,
                )

                # Only after terminal classification may completed
                # environments rearm their episode-local diagnostics.
                self._stair_event_diagnostics.reset(
                    done_mask
                )

        # 044/047 has its own ordered state machine. Consume its immutable
        # pre-autoreset snapshot directly instead of pretending it is 043.
        structured_runtime = getattr(
            env,
            "_structured_step044",
            None,
        )

        if (
            structured_runtime is not None
            and structured_runtime.snapshot is not None
        ):
            structured = structured_runtime.snapshot
            diagnostics = structured["diagnostics"]

            for name, value in structured[
                "measurements"
            ].items():
                key = "step044_diag_" + name
                self._structured044_counts[key] = (
                    self._structured044_counts.get(key, 0)
                    + int(value.to(torch.long).sum().item())
                )

            ascent_event = diagnostics.get(
                "ascent_start_event"
            )
            distance_to_riser = diagnostics.get(
                "distance_to_riser"
            )

            if (
                ascent_event is not None
                and distance_to_riser is not None
            ):
                if bool(ascent_event.any()):
                    self._structured044_ascent_distance_sum += float(
                        distance_to_riser[
                            ascent_event
                        ].sum().item()
                    )
                    self._structured044_ascent_distance_count += int(
                        ascent_event.sum().item()
                    )

            done_mask = dones.to(dtype=torch.bool)
            foot_stage = structured["foot_stage"]

            from simulation.isaac_lab.g1_stairs import (
                structured_step044_runtime as step044_runtime,
            )

            episode_masks = {
                "episodes_finished_count": done_mask,
                "episodes_ascent_started_count": (
                    done_mask
                    & diagnostics["ascent_started"]
                ),
                "episodes_swing_count": (
                    done_mask
                    & (
                        foot_stage
                        >= step044_runtime.SWING
                    ).any(dim=1)
                ),
                "episodes_meaningful_lift_count": (
                    done_mask
                    & (
                        foot_stage
                        >= step044_runtime.LIFTED
                    ).any(dim=1)
                ),
                "episodes_riser_ready_count": (
                    done_mask
                    & (
                        foot_stage
                        >= step044_runtime.RISER_READY
                    ).any(dim=1)
                ),
                "episodes_valid_crossing_count": (
                    done_mask
                    & (
                        foot_stage
                        >= step044_runtime.CROSSED
                    ).any(dim=1)
                ),
                "episodes_good_touchdown_count": (
                    done_mask
                    & (
                        foot_stage
                        >= step044_runtime.TOUCHDOWN
                    ).any(dim=1)
                ),
                "episodes_stable_upper_count": (
                    done_mask
                    & (
                        foot_stage
                        >= step044_runtime.UPPER_SUPPORT
                    ).any(dim=1)
                ),
                "episodes_body_transfer_count": (
                    done_mask
                    & diagnostics["body_transfer_seen"]
                ),
                "episodes_both_feet_count": (
                    done_mask
                    & diagnostics["both_feet_seen"]
                ),
                "episodes_completion_count": (
                    done_mask
                    & structured["success"]
                ),
                "episodes_low_crossing_count": (
                    done_mask
                    & diagnostics[
                        "low_crossing_seen"
                    ].any(dim=1)
                ),
                "episodes_clearance_loss_count": (
                    done_mask
                    & diagnostics[
                        "clearance_loss_seen"
                    ].any(dim=1)
                ),
                "episodes_failure_count": (
                    done_mask
                    & structured["failure"]
                ),
            }

            for name, mask in episode_masks.items():
                key = "step044_episode_diag_" + name
                self._structured044_counts[key] = (
                    self._structured044_counts.get(key, 0)
                    + int(mask.sum().item())
                )

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
            indices = resolve_available_body_indices(
                sensor.body_names,
            )
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

        if execution_gate is not None:
            self._teacher_execution_gate_sum += float(
                execution_gate.mean().item()
            )
            self._teacher_execution_gate_samples += 1

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
            "teacher_execution_gate_mean": (
                self._teacher_execution_gate_sum
                / self._teacher_execution_gate_samples
                if self._teacher_execution_gate_samples
                else None
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

        if self._structured044_counts:
            snapshot.update(
                self._structured044_counts
            )
            snapshot["step044_environment_steps"] = (
                steps * self.num_envs
            )
            finished = self._structured044_counts.get(
                "step044_episode_diag_episodes_finished_count",
                0,
            )
            if finished:
                completed = self._structured044_counts.get(
                    "step044_episode_diag_episodes_completion_count",
                    0,
                )
                snapshot["success_rate"] = (
                    completed / finished
                )
            else:
                snapshot["success_rate"] = None

            if self._structured044_ascent_distance_count:
                snapshot[
                    "step044_ascent_start_distance_mean_m"
                ] = (
                    self._structured044_ascent_distance_sum
                    / self._structured044_ascent_distance_count
                )
            else:
                snapshot[
                    "step044_ascent_start_distance_mean_m"
                ] = None

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

        if self._stair_distribution is not None:
            stair_diagnostics = (
                self._stair_distribution.snapshot()
            )

        if self._stair_event_diagnostics is not None:
            (
                event_counts,
                event_diagnostics,
                episode_counts,
            ) = (
                self._stair_event_diagnostics
                .consume_iteration()
            )

            for name, value in event_counts.items():
                snapshot[
                    "step043_diag_" + name
                ] = value

            for name, value in episode_counts.items():
                snapshot[
                    "step043_episode_diag_" + name
                ] = value

            overlap = (
                stair_diagnostics.keys()
                & event_diagnostics.keys()
            )

            if overlap:
                raise RuntimeError(
                    "stair distribution/event diagnostic "
                    "name collision: "
                    + ", ".join(sorted(overlap))
                )

            stair_diagnostics.update(
                event_diagnostics
            )

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
        if getattr(self, "writer", None) is not None:
            for name, value in snapshot.items():
                if name.startswith("step043_"):
                    self.writer.add_scalar("PhysicalStep043/" + name, value, locs["it"])
                elif name.startswith("step044_") and value is not None:
                    self.writer.add_scalar("StructuredStep044/" + name, value, locs["it"])
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
        if "approach_teacher" in loss:
            snapshot["approach_teacher_loss"] = float(
                loss["approach_teacher"]
            )
            snapshot["approach_teacher_gate_mean"] = float(
                loss["approach_teacher_gate"]
            )
            snapshot["approach_teacher_error_mean"] = float(
                loss["approach_teacher_error"]
            )

            optional_teacher_metrics = {
                "approach_teacher_error_far": (
                    "approach_teacher_error_far"
                ),
                "approach_teacher_error_release": (
                    "approach_teacher_error_release"
                ),
                "approach_teacher_error_near": (
                    "approach_teacher_error_near"
                ),
                "approach_teacher_far_fraction": (
                    "approach_teacher_far_fraction"
                ),
                "approach_teacher_release_fraction": (
                    "approach_teacher_release_fraction"
                ),
                "approach_teacher_near_fraction": (
                    "approach_teacher_near_fraction"
                ),
                "approach_teacher_error_legs": (
                    "approach_teacher_error_legs"
                ),
                "approach_teacher_error_arms": (
                    "approach_teacher_error_arms"
                ),
                "approach_teacher_error_waist": (
                    "approach_teacher_error_waist"
                ),
            }

            for snapshot_name, loss_name in (
                optional_teacher_metrics.items()
            ):
                if loss_name in loss:
                    snapshot[
                        snapshot_name
                    ] = float(
                        loss[loss_name]
                    )
        try:
            import torch

            snapshot["gpu_memory_used_mib"] = float(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            )
        except Exception:
            snapshot["gpu_memory_used_mib"] = None
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

        # Preserve the historical copies in training_metrics.csv extra_json,
        # but also write a dedicated stable artifact for progression-funnel
        # and offline diagnostic reporting.
        stair_event_counts = {}
        for prefix in (
            "step043_diag_",
            "step044_diag_",
        ):
            stair_event_counts.update(
                {
                    key.removeprefix(prefix): value
                    for key, value in snapshot.items()
                    if key.startswith(prefix)
                }
            )

        self._metrics_writer.record_stair_event_counts(
            iteration=snapshot["iteration"],
            total_simulated_steps=snapshot[
                "total_simulated_steps"
            ],
            counts=stair_event_counts,
        )

        stair_episode_counts = {}
        for prefix in (
            "step043_episode_diag_",
            "step044_episode_diag_",
        ):
            stair_episode_counts.update(
                {
                    key.removeprefix(prefix): value
                    for key, value in snapshot.items()
                    if key.startswith(prefix)
                }
            )

        self._metrics_writer.record_stair_episode_counts(
            iteration=snapshot["iteration"],
            total_simulated_steps=snapshot[
                "total_simulated_steps"
            ],
            counts=stair_episode_counts,
        )

        print("[POST_UPDATE] metrics_write EXIT", flush=True)
