"""Runtime validation of the physical step-up detector on an Isaac checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

import yaml

from simulation.isaac_lab.g1_stairs.step_up_isaac import (
    FOOT_BODY_NAMES,
    FORBIDDEN_CONTACT_BODY_NAMES,
    StepUpIsaacConfig,
    capture_step_up_observation,
)
from simulation.isaac_lab.g1_stairs.step_up_state import (
    StepUpDetectorConfig,
    StepUpState,
    StepUpStateDetector,
)


MINIMUM_CALIBRATION_SAMPLES = 5


def calibration_ready(
    loaded_force_samples: list[float],
    base_height_samples: list[float],
    *,
    minimum_samples: int = MINIMUM_CALIBRATION_SAMPLES,
) -> bool:
    """Return whether both measured calibration populations are sufficient."""
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    return (
        len(loaded_force_samples) >= minimum_samples
        and len(base_height_samples) >= minimum_samples
    )


@dataclass(frozen=True)
class AppliedReplayConfig:
    riser_height_m: float
    tread_depth_m: float
    step_count: int
    forward_velocity_mps: float
    first_riser_x_m: float = 1.0


def apply_resolved_replay_config(env_cfg, path: Path) -> AppliedReplayConfig:
    """Reproduce the recorded environment contract for replay/evaluation.

    Replay must use the same specialist robot, staging, observation, terrain,
    command, and reward contracts as training. Historical non-specialist runs
    continue through the generic terrain/command path below.
    """

    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))

    from simulation.isaac_lab.g1_stairs.specialist_runtime import (
        apply_specialist_runtime,
    )
    from simulation.isaac_lab.g1_stairs.reward_runtime import (
        apply_resolved_rewards,
    )

    specialist_summary = apply_specialist_runtime(
        env_cfg,
        resolved,
    )

    observations = resolved.get(
        "observations",
        {},
    )

    if observations.get(
        "actor_height_scan",
        True,
    ) is False:
        env_cfg.observations.policy.height_scan = None

    terrain = resolved.get("terrain", {})
    riser_height_m = float(terrain["riser_height_m"])
    tread_depth_m = float(terrain["tread_depth_m"])
    step_count = int(terrain["step_count"])

    if not 0.0 < riser_height_m <= 0.20:
        raise ValueError("invalid resolved riser height")
    if not 1 <= step_count <= 3:
        raise ValueError("invalid resolved step count")

    # The leaf experiment override is stored here.
    command = resolved.get("command", {})
    if "forward_velocity_mps" in command:
        forward_velocity_mps = float(command["forward_velocity_mps"])
    else:
        values = resolved["commands"]["forward_velocity_mps"]
        forward_velocity_mps = float(values[0])

    if not 0.0 <= forward_velocity_mps <= 0.50:
        raise ValueError("invalid resolved forward velocity")

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]

    stair_cfg.riser_height_range = (
        riser_height_m,
        riser_height_m,
    )
    stair_cfg.tread_depth = tread_depth_m
    stair_cfg.step_count = step_count

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (
        forward_velocity_mps,
        forward_velocity_mps,
    )
    env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (
        forward_velocity_mps,
        forward_velocity_mps,
    )

    stair_schedule_cfg = command.get(
        "stair_velocity_schedule",
        {},
    )

    # Match the exact reward-manager contract used by the recorded run.
    # For specialist experiments this removes all inherited legacy terms and
    # leaves only the YAML-declared active set.
    apply_resolved_rewards(env_cfg, resolved)

    if stair_schedule_cfg.get("enabled", False):
        from simulation.isaac_lab.g1_stairs import (
            terrain_observations,
        )

        schedule_params = {
            "approach_velocity_mps": float(
                stair_schedule_cfg[
                    "approach_velocity_mps"
                ]
            ),
            "stair_velocity_mps": float(
                stair_schedule_cfg[
                    "stair_velocity_mps"
                ]
            ),
            "slowdown_start_distance_m": float(
                stair_schedule_cfg[
                    "slowdown_start_distance_m"
                ]
            ),
            "slowdown_end_distance_m": float(
                stair_schedule_cfg[
                    "slowdown_end_distance_m"
                ]
            ),
            "platform_velocity_mps": float(
                stair_schedule_cfg[
                    "platform_velocity_mps"
                ]
            ),
            "platform_accel_distance_m": float(
                stair_schedule_cfg[
                    "platform_accel_distance_m"
                ]
            ),
        }

        for observation_group in (
            env_cfg.observations.policy,
            env_cfg.observations.critic,
        ):
            observation_group.velocity_commands.func = (
                terrain_observations.stair_velocity_command
            )
            observation_group.velocity_commands.params = dict(
                schedule_params
            )

    first_riser_x_m = 1.0
    if specialist_summary is not None:
        first_riser_x_m = float(
            specialist_summary["terrain_origin_distance_m"]
        )

    return AppliedReplayConfig(
        riser_height_m=riser_height_m,
        tread_depth_m=tread_depth_m,
        step_count=step_count,
        forward_velocity_mps=forward_velocity_mps,
        first_riser_x_m=first_riser_x_m,
    )


def apply_resolved_replay_agent_config(agent_cfg, resolved: dict) -> None:
    """Apply inference-critical PPO settings from a recorded config."""

    ppo = resolved.get("ppo", {})
    if not isinstance(ppo, dict):
        raise ValueError("ppo must be a mapping")

    if "action_clip" in ppo:
        action_clip = float(ppo["action_clip"])
        if action_clip <= 0.0:
            raise ValueError("ppo.action_clip must be > 0")
        agent_cfg.clip_actions = action_clip


class StepUpCheckpointValidator:
    """Calibrate on the approach, then classify physical step-up transitions."""

    def __init__(
        self,
        env,
        *,
        replay: AppliedReplayConfig,
        dt: float,
    ) -> None:
        import torch

        self.torch = torch
        self.env = env
        self.scene = env.scene
        self.robot = self.scene["robot"]
        self.sensor = self.scene.sensors["contact_forces"]
        self.replay = replay
        self.dt = float(dt)

        robot_names = tuple(self.robot.body_names)
        sensor_names = tuple(self.sensor.body_names)

        self.robot_foot_indices = {
            name: robot_names.index(name)
            for name in FOOT_BODY_NAMES
        }
        self.sensor_foot_indices = {
            name: sensor_names.index(name)
            for name in FOOT_BODY_NAMES
        }
        # Fixed bodies without collision shapes can be absent from the active
        # ContactSensor inventory. Follow the runtime sensor contract instead
        # of requiring every articulation body to be represented.
        self.sensor_forbidden_indices = {
            name: sensor_names.index(name)
            for name in FORBIDDEN_CONTACT_BODY_NAMES
            if name in sensor_names
        }

        self.loaded_force_samples: list[float] = []
        self.base_height_samples: list[float] = []

        self.max_left_fz = 0.0
        self.max_right_fz = 0.0
        self.forbidden_contact_steps = 0

        self.detector: StepUpStateDetector | None = None
        self.isaac_cfg: StepUpIsaacConfig | None = None

        self.first_support_events = 0
        self.base_above_events = 0
        self.both_feet_supported_events = 0
        self.completed_events = 0

        self.maximum_state = StepUpState.APPROACH

        self.both_support_start_x: float | None = None

        # Preserve the historical 0.55 m calibration boundary when the first
        # riser is at 1.0 m, while scaling correctly for specialist staging.
        self.calibration_end_x_m = 0.55 * replay.first_riser_x_m

    def _local_base(self) -> tuple[float, float]:
        origin = self.scene.env_origins[0]
        root = self.robot.data.root_pos_w[0]

        return (
            float(root[0].item() - origin[0].item()),
            float(root[2].item() - origin[2].item()),
        )

    def _upright(self) -> float:
        return float(
            -self.robot.data.projected_gravity_b[0, 2].item()
        )

    def _foot_fz(self, name: str) -> float:
        value = float(
            self.sensor.data.net_forces_w[
                0,
                self.sensor_foot_indices[name],
                2,
            ].item()
        )
        return max(0.0, value)

    def _forbidden_contact(self) -> bool:
        torch = self.torch

        for body_index in self.sensor_forbidden_indices.values():
            force = self.sensor.data.net_forces_w[0, body_index]

            if float(torch.linalg.vector_norm(force).item()) >= 5.0:
                return True

        return False

    def _calibration_sample(self) -> None:
        base_x, base_z = self._local_base()
        upright = self._upright()
        forbidden = self._forbidden_contact()

        left_fz = self._foot_fz("left_ankle_roll_link")
        right_fz = self._foot_fz("right_ankle_roll_link")

        self.max_left_fz = max(self.max_left_fz, left_fz)
        self.max_right_fz = max(self.max_right_fz, right_fz)

        if (
            base_x <= self.calibration_end_x_m
            and upright >= 0.80
            and not forbidden
        ):
            self.base_height_samples.append(base_z)

            # Exclude unloaded/swing feet from stance-force calibration.
            for force in (left_fz, right_fz):
                if force >= 5.0:
                    self.loaded_force_samples.append(force)

    def _finish_calibration(self) -> None:
        if self.detector is not None:
            return

        if len(self.loaded_force_samples) < MINIMUM_CALIBRATION_SAMPLES:
            raise RuntimeError(
                "insufficient loaded-foot samples for support calibration: "
                f"{len(self.loaded_force_samples)}"
            )

        if len(self.base_height_samples) < MINIMUM_CALIBRATION_SAMPLES:
            raise RuntimeError(
                "insufficient base-height samples for calibration: "
                f"{len(self.base_height_samples)}"
            )

        stance_force = float(median(self.loaded_force_samples))
        support_threshold = 0.25 * stance_force
        lower_base_height = float(median(self.base_height_samples))

        self.isaac_cfg = StepUpIsaacConfig(
            first_riser_x_m=self.replay.first_riser_x_m,
            tread_depth_m=self.replay.tread_depth_m,
            riser_height_m=self.replay.riser_height_m,
            nominal_ankle_height_m=0.10,
            foot_height_tolerance_m=0.025,
            tread_x_margin_m=0.02,
            maximum_supported_foot_speed_mps=0.15,
            minimum_upright_gravity_z=0.80,
            minimum_vertical_support_force_n=support_threshold,
            lower_level_base_height_m=lower_base_height,
            minimum_base_height_gain_m=0.025,
            base_past_riser_margin_m=0.02,
            forbidden_contact_force_threshold_n=5.0,
        )

        detector_cfg = StepUpDetectorConfig(
            minimum_support_duration_s=0.15,
            maximum_supported_foot_speed_mps=0.15,
            minimum_upright_gravity_z=0.80,
            minimum_vertical_support_force_n=support_threshold,
            completion_forward_distance_m=0.40,
        )

        self.detector = StepUpStateDetector(
            detector_cfg,
            dt=self.dt,
        )

        print(
            "[STEP-UP] calibrated "
            f"stance_force_median_n={stance_force:.3f} "
            f"support_threshold_n={support_threshold:.3f} "
            f"lower_base_height_m={lower_base_height:.4f}",
            flush=True,
        )

    def update(self) -> None:
        base_x, _ = self._local_base()

        left_fz = self._foot_fz("left_ankle_roll_link")
        right_fz = self._foot_fz("right_ankle_roll_link")

        self.max_left_fz = max(self.max_left_fz, left_fz)
        self.max_right_fz = max(self.max_right_fz, right_fz)

        if self._forbidden_contact():
            self.forbidden_contact_steps += 1

        if self.detector is None:
            self._calibration_sample()

            # Begin classification as soon as enough genuine approach data
            # exists. Waiting for the robot to cross a fixed X threshold can
            # leave failed policies uncalibrated when they stall or fall early.
            if calibration_ready(
                self.loaded_force_samples,
                self.base_height_samples,
            ):
                self._finish_calibration()

            elif base_x > self.calibration_end_x_m:
                # The calibration window has ended without sufficient valid
                # upright stance data. Fail immediately with the measured
                # sample counts instead of deferring the error to summary().
                self._finish_calibration()

            # Do not classify samples used to establish calibration.
            if self.detector is None:
                return

        assert self.isaac_cfg is not None
        assert self.detector is not None

        continuation = 0.0
        if self.both_support_start_x is not None:
            continuation = max(
                0.0,
                base_x - self.both_support_start_x,
            )

        observation = capture_step_up_observation(
            self.env,
            env_index=0,
            config=self.isaac_cfg,
            upper_forward_continuation_m=continuation,
        )

        result = self.detector.update(observation)

        self.maximum_state = max(
            self.maximum_state,
            result.state,
        )

        if result.entered_first_support:
            self.first_support_events += 1
            print(
                f"[STEP-UP] FIRST_SUPPORT x={base_x:.4f}",
                flush=True,
            )

        if result.entered_base_above:
            self.base_above_events += 1
            print(
                f"[STEP-UP] BASE_ABOVE x={base_x:.4f}",
                flush=True,
            )

        if result.entered_both_feet_supported:
            self.both_feet_supported_events += 1
            self.both_support_start_x = base_x
            print(
                f"[STEP-UP] BOTH_FEET_SUPPORTED x={base_x:.4f}",
                flush=True,
            )

        if result.entered_completed:
            self.completed_events += 1
            print(
                f"[STEP-UP] COMPLETED x={base_x:.4f}",
                flush=True,
            )

    def summary(self) -> dict:
        if self.detector is None:
            self._finish_calibration()

        assert self.isaac_cfg is not None

        stance_force = float(median(self.loaded_force_samples))

        passed = (
            self.both_feet_supported_events == 0
            and self.completed_events == 0
        )

        return {
            "stance_force_median_n": stance_force,
            "support_threshold_n": (
                self.isaac_cfg.minimum_vertical_support_force_n
            ),
            "lower_base_height_m": (
                self.isaac_cfg.lower_level_base_height_m
            ),
            "max_left_fz_n": self.max_left_fz,
            "max_right_fz_n": self.max_right_fz,
            "first_support_events": self.first_support_events,
            "base_above_events": self.base_above_events,
            "both_feet_supported_events": (
                self.both_feet_supported_events
            ),
            "completed_events": self.completed_events,
            "maximum_state": self.maximum_state.name,
            "forbidden_contact_steps": self.forbidden_contact_steps,
            "passed": passed,
        }
