"""Live Isaac Lab/RSL-RL metrics bridge for the active stair task."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from simulation.isaac_lab.g1_stairs.action_diagnostics import (
    ACTION_GROUP_INDICES,
    action_group_mean_abs,
)
from simulation.isaac_lab.g1_stairs.evaluation.metrics import RunMetricsWriter
from simulation.isaac_lab.g1_stairs.contact_sensor_contract import (
    FOOT_BODY_NAMES,
    FORBIDDEN_CONTACT_BODY_NAMES,
    resolve_available_body_indices,
)
from simulation.isaac_lab.g1_stairs.contracts import (
    BODY_JOINTS,
    G1_SOLE_SPHERE_CENTRES_M,
    G1_SOLE_SPHERE_RADIUS_M,
)
from simulation.isaac_lab.g1_stairs.evaluation.stair_diagnostics import (
    NamedTensorStatsAccumulator,
)
from simulation.isaac_lab.g1_stairs.reward_math import tread_containment_score
from simulation.isaac_lab.g1_stairs.terrain_state import (
    riser_height_per_env,
    stair_cfg,
)


STAIR_DIAGNOSTIC_NAMES = (
    "root_distance_to_riser_m",
    "mean_foot_front_distance_to_riser_m",
    "lead_foot_front_distance_to_riser_m",
    "left_foot_front_distance_to_riser_m",
    "right_foot_front_distance_to_riser_m",
    "left_sole_bottom_m",
    "right_sole_bottom_m",
    "maximum_sole_bottom_m",
    "maximum_clearance_above_riser_m",
    "maximum_upper_containment",
    "maximum_vertical_foot_force_n",
    "left_vertical_foot_force_n",
    "right_vertical_foot_force_n",
    "minimum_foot_speed_mps",
    "maximum_supported_horizontal_foot_speed_mps",
    "root_height_m",
    "upright_gravity_z",
    "native_failure_termination",
    "motion_phase_index",
    "motion_phase_progress",
    "swing_foot_center_x_m",
    "swing_foot_center_y_m",
    "support_foot_center_x_m",
    "support_foot_center_y_m",
    "swing_target_x_error_m",
    "swing_lateral_error_m",
    "root_to_support_foot_y_m",
    "swing_foot_force_share",
    "support_foot_force_share",
    "both_feet_airborne",
    "swing_overheight_m",
    "support_foot_lift_m",
    "swing_backward_error_m",
    "swing_forward_overshoot_m",
    "swing_foot_vertical_speed_mps",
    "swing_foot_roll_rad",
    "swing_foot_pitch_rad",
    "swing_foot_yaw_rad",
    "support_foot_roll_rad",
    "support_foot_pitch_rad",
    "support_foot_yaw_rad",
)


POSTURE_BASE_DIAGNOSTIC_NAMES = (
    "posture_root_roll_rad",
    "posture_root_pitch_rad",
    "posture_root_yaw_rad",
    "posture_root_lin_vel_x_mps",
    "posture_root_lin_vel_y_mps",
    "posture_root_lin_vel_z_mps",
    "posture_root_ang_vel_x_radps",
    "posture_root_ang_vel_y_radps",
    "posture_root_ang_vel_z_radps",
    "posture_root_minus_mean_foot_x_m",
    "posture_root_minus_mean_foot_y_m",
    "posture_legs_error_rms_rad",
    "posture_arms_error_rms_rad",
    "posture_waist_error_rms_rad",
    "posture_legs_action_rms",
    "posture_arms_action_rms",
    "posture_waist_action_rms",
)


def _joint_posture_names(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}_{name.removesuffix('_joint')}"
        for name in BODY_JOINTS
    )


POSTURE_DIAGNOSTIC_NAMES = (
    *POSTURE_BASE_DIAGNOSTIC_NAMES,
    *_joint_posture_names("posture_joint_pos_rad"),
    *_joint_posture_names("posture_joint_default_rad"),
    *_joint_posture_names("posture_joint_error_rad"),
    *_joint_posture_names("posture_joint_vel_radps"),
    *_joint_posture_names("posture_policy_action"),
)


def _quat_to_euler_wxyz(quaternion):
    import torch

    w, x, y, z = quaternion.unbind(dim=1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = torch.asin(
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    )
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def _quat_rotate_wxyz(quaternion, vector):
    import torch

    q_w = quaternion[..., :1]
    q_v = quaternion[..., 1:]
    uv = torch.cross(q_v, vector, dim=-1)
    uuv = torch.cross(q_v, uv, dim=-1)
    return vector + 2.0 * (q_w * uv + uuv)


def _body_linear_velocity(data):
    if hasattr(data, "body_link_lin_vel_w"):
        return data.body_link_lin_vel_w
    if hasattr(data, "body_lin_vel_w"):
        return data.body_lin_vel_w
    raise RuntimeError(
        "robot articulation data has no body linear velocity tensor"
    )


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
        self._stair_body_indices = None
        self._stair_sensor_indices = None
        self._posture_joint_indices = None

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
        self._platform_reached_envs = None
        self._fall_count = 0
        self._forbidden_contact_count = 0
        self._finite_action_env_steps = 0

        # Distribution diagnostics are separate from the historical
        # reward-component accumulator. They never feed back into PPO.
        self._reward_distribution = None
        self._reward_distribution_weights = {}
        self._stair_distribution = NamedTensorStatsAccumulator(
            STAIR_DIAGNOSTIC_NAMES
        )
        self._posture_distribution = NamedTensorStatsAccumulator(
            POSTURE_DIAGNOSTIC_NAMES
        )

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
        if (
            self._platform_reached_envs is None
            or self._platform_reached_envs.shape[0] != local_x.shape[0]
        ):
            self._platform_reached_envs = torch.zeros(
                local_x.shape[0],
                dtype=torch.bool,
                device=local_x.device,
            )
        self._platform_reached_envs |= local_x >= platform_start

        # Reward-independent mechanics diagnostics. These signals never feed
        # PPO; they only make the current physical bottleneck observable.
        if self._stair_body_indices is None:
            body_index = {name: i for i, name in enumerate(robot.body_names)}
            sensor_index = {
                name: i for i, name in enumerate(sensor.body_names)
            } if sensor is not None else {}
            missing_bodies = [
                name for name in FOOT_BODY_NAMES if name not in body_index
            ]
            missing_sensor = [
                name for name in FOOT_BODY_NAMES if name not in sensor_index
            ]
            if missing_bodies or missing_sensor:
                raise RuntimeError(
                    "stair diagnostics require both ankle-roll bodies in "
                    "robot and contact sensor"
                )
            self._stair_body_indices = tuple(
                body_index[name] for name in FOOT_BODY_NAMES
            )
            self._stair_sensor_indices = tuple(
                sensor_index[name] for name in FOOT_BODY_NAMES
            )

        foot_ids = list(self._stair_body_indices)
        sensor_ids = list(self._stair_sensor_indices)
        foot_pos_w = data.body_pos_w[:, foot_ids]
        foot_quat_w = data.body_quat_w[:, foot_ids]
        foot_vel_w = _body_linear_velocity(data)[:, foot_ids]

        centres = torch.tensor(
            G1_SOLE_SPHERE_CENTRES_M,
            dtype=foot_pos_w.dtype,
            device=foot_pos_w.device,
        ).view(1, 1, 4, 3).expand(
            foot_pos_w.shape[0], 2, 4, 3
        )
        quaternions = foot_quat_w.unsqueeze(2).expand(
            foot_pos_w.shape[0], 2, 4, 4
        )
        sole_centres_w = foot_pos_w.unsqueeze(2) + _quat_rotate_wxyz(
            quaternions, centres
        )
        sole_local = (
            sole_centres_w
            - env.scene.env_origins[:, None, None, :]
        )
        rear_x = sole_local[..., 0].amin(dim=2)
        front_x = sole_local[..., 0].amax(dim=2)
        sole_center = sole_local.mean(dim=2)
        sole_bottom = (
            sole_local[..., 2] - G1_SOLE_SPHERE_RADIUS_M
        ).amin(dim=2)
        foot_roll = []
        foot_pitch = []
        foot_yaw = []
        for foot_index in range(2):
            r, p, y = _quat_to_euler_wxyz(foot_quat_w[:, foot_index])
            foot_roll.append(r)
            foot_pitch.append(p)
            foot_yaw.append(y)
        foot_roll = torch.stack(foot_roll, dim=1)
        foot_pitch = torch.stack(foot_pitch, dim=1)
        foot_yaw = torch.stack(foot_yaw, dim=1)
        foot_speed = torch.linalg.vector_norm(foot_vel_w, dim=2)
        horizontal_foot_speed = torch.linalg.vector_norm(
            foot_vel_w[..., :2], dim=2
        )
        vertical_force = sensor.data.net_forces_w[:, sensor_ids, 2].clamp_min(0.0)
        supported_horizontal_speed = torch.where(
            vertical_force >= 5.0,
            horizontal_foot_speed,
            torch.zeros_like(horizontal_foot_speed),
        )
        riser_height = riser_height_per_env(env)
        containment = tread_containment_score(
            rear_x,
            front_x,
            riser_x_m=first_riser,
            tread_depth_m=tread_depth,
            margin_m=0.015,
        )
        lead_front = front_x.amax(dim=1)
        mean_front = front_x.mean(dim=1)
        maximum_sole_bottom = sole_bottom.amax(dim=1)
        maximum_clearance = (
            maximum_sole_bottom - riser_height
        )
        native_failure = env.reset_terminated.to(dtype=local_x.dtype)

        swing_left = bool(getattr(env, "_g1_stair_swing_left", True))
        swing_index = 0 if swing_left else 1
        support_index = 1 - swing_index
        phase = getattr(env, "_g1_stair_motion_phase", None)
        if phase is None:
            phase = torch.zeros_like(local_x)
        else:
            phase = phase.to(dtype=local_x.dtype)
        phase_progress = getattr(env, "_g1_stair_motion_phase_progress", None)
        if phase_progress is None:
            phase_progress = torch.zeros_like(local_x)
        target_x = getattr(env, "_g1_stair_swing_target_x", None)
        if target_x is None:
            target_x = torch.full_like(local_x, first_riser + 0.5 * tread_depth)
        lane_y = getattr(env, "_g1_stair_swing_lane_y", None)
        if lane_y is None:
            lane_y = sole_center[:, swing_index, 1]

        total_foot_force = vertical_force.sum(dim=1).clamp_min(1.0)
        swing_force_share = vertical_force[:, swing_index] / total_foot_force
        support_force_share = vertical_force[:, support_index] / total_foot_force
        both_airborne = (
            (vertical_force[:, 0] < 5.0)
            & (vertical_force[:, 1] < 5.0)
            & (sole_bottom[:, 0] > 0.010)
            & (sole_bottom[:, 1] > 0.010)
        ).to(dtype=local_x.dtype)
        swing_overheight = (
            sole_bottom[:, swing_index] - (riser_height + 0.070)
        ).clamp_min(0.0)

        self._stair_distribution.update(
            torch.stack(
                (
                    first_riser - local_x,
                    first_riser - mean_front,
                    first_riser - lead_front,
                    first_riser - front_x[:, 0],
                    first_riser - front_x[:, 1],
                    sole_bottom[:, 0],
                    sole_bottom[:, 1],
                    maximum_sole_bottom,
                    maximum_clearance,
                    containment.amax(dim=1),
                    vertical_force.amax(dim=1),
                    vertical_force[:, 0],
                    vertical_force[:, 1],
                    foot_speed.amin(dim=1),
                    supported_horizontal_speed.amax(dim=1),
                    local_z,
                    -data.projected_gravity_b[:, 2],
                    native_failure,
                    phase,
                    phase_progress,
                    sole_center[:, swing_index, 0],
                    sole_center[:, swing_index, 1],
                    sole_center[:, support_index, 0],
                    sole_center[:, support_index, 1],
                    target_x - sole_center[:, swing_index, 0],
                    sole_center[:, swing_index, 1] - lane_y,
                    local_x.new_tensor(0.0).expand_as(local_x) + (
                        data.root_pos_w[:, 1]
                        - foot_pos_w[:, support_index, 1]
                    ),
                    swing_force_share,
                    support_force_share,
                    both_airborne,
                    swing_overheight,
                    sole_bottom[:, support_index].clamp_min(0.0),
                    (
                        getattr(env, "_g1_stair_swing_start_x", sole_center[:, swing_index, 0])
                        - sole_center[:, swing_index, 0]
                    ).clamp_min(0.0),
                    (sole_center[:, swing_index, 0] - (target_x + 0.050)).clamp_min(0.0),
                    foot_vel_w[:, swing_index, 2],
                    foot_roll[:, swing_index],
                    foot_pitch[:, swing_index],
                    foot_yaw[:, swing_index],
                    foot_roll[:, support_index],
                    foot_pitch[:, support_index],
                    foot_yaw[:, support_index],
                ),
                dim=1,
            )
        )

        # Full-body posture observability.  These signals never affect PPO.
        # They distinguish a configured/default posture from a posture that
        # the policy actively commands, including the persistent arm stance.
        if self._posture_joint_indices is None:
            joint_index = {
                name: index
                for index, name in enumerate(robot.joint_names)
            }
            missing = [
                name for name in BODY_JOINTS
                if name not in joint_index
            ]
            if missing:
                raise RuntimeError(
                    "posture diagnostics require BODY_JOINTS: "
                    + ", ".join(missing)
                )
            self._posture_joint_indices = tuple(
                joint_index[name] for name in BODY_JOINTS
            )

        posture_ids = list(self._posture_joint_indices)
        joint_pos = data.joint_pos[:, posture_ids]
        joint_default = data.default_joint_pos[:, posture_ids]
        joint_error = joint_pos - joint_default
        joint_vel = data.joint_vel[:, posture_ids]

        if actions.ndim != 2 or actions.shape[1] != len(BODY_JOINTS):
            raise RuntimeError(
                "posture diagnostics require policy actions [N, 29]"
            )

        roll, pitch, yaw = _quat_to_euler_wxyz(data.root_quat_w)
        mean_foot_xy = foot_pos_w[..., :2].mean(dim=1)
        root_minus_foot = data.root_pos_w[:, :2] - mean_foot_xy

        def _rms(values, indices):
            return torch.sqrt(
                torch.mean(values[:, indices] ** 2, dim=1)
            )

        base_posture = torch.stack(
            (
                roll,
                pitch,
                yaw,
                data.root_lin_vel_w[:, 0],
                data.root_lin_vel_w[:, 1],
                data.root_lin_vel_w[:, 2],
                data.root_ang_vel_w[:, 0],
                data.root_ang_vel_w[:, 1],
                data.root_ang_vel_w[:, 2],
                root_minus_foot[:, 0],
                root_minus_foot[:, 1],
                _rms(joint_error, ACTION_GROUP_INDICES["legs"]),
                _rms(joint_error, ACTION_GROUP_INDICES["arms"]),
                _rms(joint_error, ACTION_GROUP_INDICES["waist"]),
                _rms(actions, ACTION_GROUP_INDICES["legs"]),
                _rms(actions, ACTION_GROUP_INDICES["arms"]),
                _rms(actions, ACTION_GROUP_INDICES["waist"]),
            ),
            dim=1,
        )
        posture_matrix = torch.cat(
            (
                base_posture,
                joint_pos,
                joint_default,
                joint_error,
                joint_vel,
                actions,
            ),
            dim=1,
        )
        self._posture_distribution.update(posture_matrix)
        return result

    def consume_metrics(self) -> tuple[dict, dict[str, dict[str, float]]]:
        steps = max(self._steps, 1)
        snapshot = {
            "episode_return_mean": self._reward_total,
            "maximum_x_any_m": self._maximum_x,
            "maximum_base_height_any_m": self._maximum_height,
            "maximum_step_reached_any": self._maximum_step,
            "platform_reached_rate": (
                float(
                    self._platform_reached_envs.float().mean().item()
                )
                if self._platform_reached_envs is not None
                else 0.0
            ),
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

        diagnostic_snapshots = {
            **self._stair_distribution.snapshot(),
            **self._posture_distribution.snapshot(),
        }
        stair_diagnostics = {
            name: {
                "sample_count": values["sample_count"],
                "mean": values["mean"],
                "std": values["std"],
                "min": values["min"],
                "max": values["max"],
            }
            for name, values in diagnostic_snapshots.items()
        }

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
