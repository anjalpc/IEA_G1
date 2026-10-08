"""Reusable control state and integrated G1 MuJoCo backend."""

from __future__ import annotations

import math
import threading

import numpy as np

from g1_robot_interfaces.arm_motion import (
    resolve_arm_motion,
    validate_arm_motion,
)
from model_composition import (
    BODY_ACTUATOR_COUNT,
    build_combined_model,
    hand_actuator_names,
    validate_combined_model,
)
from runtime.model_runtime import ElasticHarness, initialize


# Median physical joint positions measured after Unitree action 27 reached
# its stable right-arm presentation pose on the project G1 29-DoF robot.
PRESENT_ARM_TARGETS = {
    "right_shoulder_pitch_joint": -0.108721,
    "right_shoulder_roll_joint": -0.063594,
    "right_shoulder_yaw_joint": 0.094819,
    "right_elbow_joint": 0.182400,
    "right_wrist_roll_joint": 0.159618,
    "right_wrist_pitch_joint": -0.339249,
    "right_wrist_yaw_joint": -0.061755,
}
SHAKE_FREQUENCY_HZ = 1.4
SHAKE_ELBOW_AMPLITUDE_RAD = 0.12
SHAKE_SHOULDER_AMPLITUDE_RAD = 0.05
SHAKE_WRIST_AMPLITUDE_RAD = 0.06


class SimulationControls:
    """Thread-safe actuator targets shared by the sequencer and simulator."""

    def __init__(self):
        self.lock = threading.Lock()
        self.arm_action = 99
        self.card_arm = None
        self.card_arm_generation = 0
        self.hand_positions = {
            "left": (0.0,) * 6,
            "right": (0.0,) * 6,
        }
        self.hand_ramps = {"left": 0.2, "right": 0.2}
        self.events: list[str] = []

    def set_arm(self, action_id: int, event: str) -> None:
        with self.lock:
            self.arm_action = action_id
            if action_id == 99:
                self.card_arm = None
            self.events.append(event)

    def card_arm_snapshot(self):
        with self.lock:
            return self.card_arm, self.card_arm_generation

    def set_arm_motion(self, motion):
        """Store an independent neutral two-phase motion request."""
        motion = validate_arm_motion(motion)
        with self.lock:
            self.arm_action = 99
            self.card_arm = motion
            self.card_arm_generation += 1
            self.events.append('arm:motion')

    def set_hand(self, hand: str, values: list[float], event: str,
                 ramp_seconds: float = 0.2) -> None:
        if len(values) != 6:
            raise ValueError("BrainCo target must contain six channels")
        with self.lock:
            self.hand_positions[hand] = tuple(float(value) for value in values)
            self.hand_ramps[hand] = float(ramp_seconds)
            self.events.append(event)

    def snapshot(self):
        with self.lock:
            return (self.arm_action, dict(self.hand_positions),
                    dict(self.hand_ramps))


class IntegratedG1Backend:
    """PD position backend for the real G1 joints and BrainCo hand channels."""

    def __init__(self, controls: SimulationControls,
                 body_control_enabled: bool = True):
        import mujoco

        self.mujoco = mujoco
        self.model = build_combined_model()
        validate_combined_model(self.model)
        self.data = mujoco.MjData(self.model)
        initialize(self.model, self.data)
        self.harness = ElasticHarness(self.model, self.data)

        self.neutral = np.zeros(BODY_ACTUATOR_COUNT)
        for actuator_id in range(BODY_ACTUATOR_COUNT):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            self.neutral[actuator_id] = self.data.qpos[
                self.model.jnt_qposadr[joint_id]]
        self.controls = controls
        self.body_control_enabled = body_control_enabled
        self.hand_commands = {side: np.zeros(6) for side in ("left", "right")}
        self.hand_starts = {side: np.zeros(6) for side in ("left", "right")}
        self.hand_targets = {side: np.zeros(6) for side in ("left", "right")}
        self.hand_transition_started = {
            side: self.data.time for side in ("left", "right")}
        self.hand_ids = {
            side: tuple(mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in hand_actuator_names(side))
            for side in ("left", "right")
        }
        self.card_generation = -1
        self.card_started = self.data.time
        self.card_initial_targets = {}

    def _body_torques(self, action_id: int, gripping: bool) -> None:
        if not self.body_control_enabled:
            self.data.ctrl[:BODY_ACTUATOR_COUNT] = 0.0
            return
        targets = self.neutral.copy()
        if action_id == 27:
            arm_targets = PRESENT_ARM_TARGETS.copy()
            if gripping:
                phase = 2.0 * math.pi * SHAKE_FREQUENCY_HZ * self.data.time
                wave = math.sin(phase)
                arm_targets["right_elbow_joint"] += (
                    SHAKE_ELBOW_AMPLITUDE_RAD * wave)
                arm_targets["right_shoulder_pitch_joint"] -= (
                    SHAKE_SHOULDER_AMPLITUDE_RAD * wave)
                arm_targets["right_wrist_pitch_joint"] += (
                    SHAKE_WRIST_AMPLITUDE_RAD * wave)
            for joint_name, target in arm_targets.items():
                joint_id = self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                actuator_ids = np.flatnonzero(
                    self.model.actuator_trnid[:BODY_ACTUATOR_COUNT, 0] == joint_id)
                if len(actuator_ids) != 1:
                    raise RuntimeError(f"no unique body actuator for {joint_name}")
                targets[int(actuator_ids[0])] = target

        card_arm, generation = self.controls.card_arm_snapshot()
        if isinstance(card_arm, dict):
            for joint_name, target in self._motion_targets(card_arm, generation).items():
                joint_id = self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                actuator_ids = np.flatnonzero(
                    self.model.actuator_trnid[:BODY_ACTUATOR_COUNT, 0] == joint_id)
                if len(actuator_ids) != 1:
                    raise RuntimeError(f'no unique body actuator for {joint_name}')
                targets[int(actuator_ids[0])] = target
        for actuator_id, target in enumerate(targets):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            qpos = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            qvel = self.data.qvel[self.model.jnt_dofadr[joint_id]]
            is_arm = "shoulder" in (
                self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "")
            kp = 35.0 if is_arm else 55.0
            kd = 3.0 if is_arm else 4.0
            torque = kp * (target - qpos) - kd * qvel
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(torque, low, high)

    def _motion_targets(self, motion, generation):
        """Resolve the restricted request from measured simulation state."""
        if generation != self.card_generation:
            self.card_generation = generation
            self.card_started = self.data.time
            self.card_initial_targets = {}
            for name in motion['preparation']:
                joint_id = self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
                self.card_initial_targets[name] = self.data.qpos[
                    self.model.jnt_qposadr[joint_id]]
        baseline = self.card_initial_targets
        preparation, target = resolve_arm_motion(motion, baseline)
        first, second = motion['phase_seconds']
        elapsed = self.data.time - self.card_started
        if elapsed < first:
            origin, destination, ratio = baseline, preparation, elapsed / first
        else:
            origin, destination, ratio = preparation, target, (elapsed - first) / second
        ratio = max(0.0, min(1.0, ratio))
        blend = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5
        return {name: origin[name] + blend * (destination[name] - origin[name])
                for name in baseline}

    def _hand_targets(self, hand: str,
                      normalized_positions: tuple[float, ...],
                      ramp_seconds: float) -> None:
        target = np.asarray(normalized_positions, dtype=float)
        if target.shape != (6,):
            raise ValueError("BrainCo target must contain six channels")
        if not np.array_equal(target, self.hand_targets[hand]):
            self.hand_starts[hand] = self.hand_commands[hand].copy()
            self.hand_targets[hand] = target.copy()
            self.hand_transition_started[hand] = self.data.time
        fraction = np.clip(
            (self.data.time - self.hand_transition_started[hand]) /
            ramp_seconds,
            0.0,
            1.0,
        )
        smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        self.hand_commands[hand] = (
            self.hand_starts[hand]
            + smooth_fraction * (
                self.hand_targets[hand] - self.hand_starts[hand])
        )
        for channel, actuator_id in enumerate(self.hand_ids[hand]):
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = (
                low + self.hand_commands[hand][channel] * (high - low))

    def step(self) -> None:
        action_id, hand_positions, hand_ramps = self.controls.snapshot()
        self._body_torques(
            action_id, gripping=any(
                value > 0.0 for value in hand_positions["right"]))
        for hand, positions in hand_positions.items():
            self._hand_targets(hand, positions, hand_ramps[hand])
        self.harness.apply(self.data)
        self.mujoco.mj_step(self.model, self.data)


