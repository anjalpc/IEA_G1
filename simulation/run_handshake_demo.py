#!/usr/bin/env python3
"""Run the handshake sequence on the integrated G1 and BrainCo MuJoCo model."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INTERFACES_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "g1_robot_interfaces"
HANDSHAKE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "g1_handshake"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "handshake_surrogate.xml"
sys.path.insert(0, str(INTERFACES_SOURCE))
sys.path.insert(0, str(HANDSHAKE_SOURCE))

from g1_handshake.handshake_sequence import AutomaticHandshake  # noqa: E402
from model_composition import (  # noqa: E402
    BODY_ACTUATOR_COUNT,
    build_combined_model,
    hand_actuator_names,
    validate_combined_model,
)
from run_g1_model import ElasticHarness, initialize  # noqa: E402


PRESENT_ARM_TARGETS = {
    "right_shoulder_pitch_joint": -0.55,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.05,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
SHAKE_FREQUENCY_HZ = 1.4
SHAKE_ELBOW_AMPLITUDE_RAD = 0.12
SHAKE_SHOULDER_AMPLITUDE_RAD = 0.05
SHAKE_WRIST_AMPLITUDE_RAD = 0.06
SIMULATED_PRESENTATION_SECONDS = 1.0


def configured_hand_profile() -> tuple[list[float], list[float], float]:
    """Read the shared hand targets and ramp used by both backends."""
    configuration = REPOSITORY_ROOT / "ros2_ws" / "src" / \
        "g1_handshake" / "config" / "handshake.yaml"
    data = yaml.safe_load(configuration.read_text())
    parameters = data["g1_hardware_bridge"]["ros__parameters"]
    return (list(parameters["hand_grip_target"]),
            list(parameters["hand_neutral_target"]),
            float(parameters["grip_ramp_s"]))


HAND_RAMP_SECONDS = configured_hand_profile()[2]


class SimulationControls:
    """Thread-safe actuator targets shared by the sequencer and simulator."""

    def __init__(self):
        self.lock = threading.Lock()
        self.arm_action = 99
        self.hand_positions = (0.0,) * 6
        self.events: list[str] = []

    def set_arm(self, action_id: int, event: str) -> None:
        with self.lock:
            self.arm_action = action_id
            self.events.append(event)

    def set_hand(self, values: list[float], event: str) -> None:
        if len(values) != 6:
            raise ValueError("BrainCo target must contain six channels")
        with self.lock:
            self.hand_positions = tuple(float(value) for value in values)
            self.events.append(event)

    def snapshot(self) -> tuple[int, tuple[float, ...]]:
        with self.lock:
            return self.arm_action, self.hand_positions


class SimulatedArm:
    def __init__(self, controls: SimulationControls, enabled: bool = True):
        self.controls = controls
        self.enabled = enabled
        self.cancelled = False

    def prepare_operation(self) -> None:
        self.cancelled = False

    def interrupt(self) -> None:
        self.cancelled = True

    def execute(self, action_id: int) -> None:
        if self.cancelled:
            raise RuntimeError("simulated arm action cancelled")
        if action_id not in (27, 99):
            raise ValueError(f"unsupported simulated arm action: {action_id}")
        simulated_action = action_id if self.enabled else 99
        event = (
            f"arm:{action_id}"
            if self.enabled or action_id == 99
            else f"arm:{action_id}:skipped"
        )
        self.controls.set_arm(simulated_action, event)
        if action_id == 27:
            # The hardware action service blocks while the arm reaches its
            # presentation pose. Preserve that behavior in the simulation so
            # gripping and shaking do not begin during the reach.
            time.sleep(SIMULATED_PRESENTATION_SECONDS)

    def release(self) -> None:
        self.interrupt()
        self.controls.set_arm(99, "arm:99")


class SimulatedHand:
    def __init__(self, controls: SimulationControls,
                 curl_target: list[float], neutral_target: list[float],
                 ramp_seconds: float):
        self.controls = controls
        self.curl_target = curl_target
        self.neutral_target = neutral_target
        self.ramp_seconds = ramp_seconds
        self.cancelled = False

    def prepare_operation(self) -> None:
        self.cancelled = False

    def interrupt(self) -> None:
        self.cancelled = True

    def grip(self) -> None:
        if self.cancelled:
            raise RuntimeError("simulated hand command cancelled")
        print(
            "HAND TARGET [thumb, thumb_aux, index, middle, ring, pinky]: "
            + " ".join(f"{value:.4f}" for value in self.curl_target),
            flush=True,
        )
        self.controls.set_hand(self.curl_target, "hand:grip")
        time.sleep(self.ramp_seconds)

    def release(self) -> None:
        self.interrupt()
        self.controls.set_hand(self.neutral_target, "hand:release")
        time.sleep(self.ramp_seconds)


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
        self.right_hand_command = np.zeros(6)
        self.right_hand_start = np.zeros(6)
        self.right_hand_target = np.zeros(6)
        self.hand_transition_started = self.data.time
        self.right_hand_ids = tuple(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in hand_actuator_names("right"))
        self.left_hand_ids = tuple(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in hand_actuator_names("left"))

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

    def _hand_targets(self, normalized_positions: tuple[float, ...]) -> None:
        target = np.asarray(normalized_positions, dtype=float)
        if target.shape != (6,):
            raise ValueError("BrainCo target must contain six channels")
        if not np.array_equal(target, self.right_hand_target):
            self.right_hand_start = self.right_hand_command.copy()
            self.right_hand_target = target.copy()
            self.hand_transition_started = self.data.time
        fraction = np.clip(
            (self.data.time - self.hand_transition_started) / HAND_RAMP_SECONDS,
            0.0,
            1.0,
        )
        smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        self.right_hand_command = (
            self.right_hand_start
            + smooth_fraction * (self.right_hand_target - self.right_hand_start)
        )
        for actuator_id in self.left_hand_ids:
            self.data.ctrl[actuator_id] = 0.0
        for channel, actuator_id in enumerate(self.right_hand_ids):
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = (
                low + self.right_hand_command[channel] * (high - low))

    def step(self) -> None:
        action_id, hand_positions = self.controls.snapshot()
        self._body_torques(
            action_id, gripping=any(value > 0.0 for value in hand_positions))
        self._hand_targets(hand_positions)
        self.harness.apply(self.data)
        self.mujoco.mj_step(self.model, self.data)


class SurrogateBackend:
    """Explicit opt-in fixture retained only for lightweight debugging."""

    def __init__(self, controls: SimulationControls):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.controls = controls

    def step(self) -> None:
        action_id, hand_positions = self.controls.snapshot()
        if action_id == 27:
            arm = (-0.55, -1.05)
        else:
            arm = (0.0, 0.0)
        fixture_hand = [hand_positions[0], *hand_positions[2:6]]
        self.data.ctrl[:] = [*arm, *fixture_hand]
        self.mujoco.mj_step(self.model, self.data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument(
        "--curl-target", type=float, nargs=6,
        metavar=("THUMB", "THUMB_AUX", "INDEX", "MIDDLE", "RING", "PINKY"),
        help="explicit six-channel simulation target; overrides curl-amount")
    parser.add_argument("--cancel-after", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--render-fps", type=float, default=30.0,
        help="interactive render rate; physics uses multiple substeps")
    parser.add_argument(
        "--surrogate", action="store_true",
        help="use the generic CI fixture instead of the integrated robot")
    parser.add_argument(
        "--hand-only", action="store_true",
        help="leave body actuators passive and isolate the BrainCo curl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration < 0.0:
        raise ValueError("duration must not be negative")
    if (args.curl_target is not None
            and any(value < 0.0 or value > 1.0
                    for value in args.curl_target)):
        raise ValueError("every curl-target value must be in the range 0..1")
    if args.cancel_after is not None and args.cancel_after < 0.0:
        raise ValueError("cancel-after must not be negative")
    if args.render_fps <= 0.0:
        raise ValueError("render-fps must be positive")

    import mujoco

    controls = SimulationControls()
    backend = (
        SurrogateBackend(controls)
        if args.surrogate else IntegratedG1Backend(
            controls, body_control_enabled=not args.hand_only)
    )
    model = backend.model
    data = backend.data
    if args.validate_only:
        if args.surrogate:
            print(f"VALID: explicit surrogate {MODEL_PATH.relative_to(REPOSITORY_ROOT)}")
        else:
            print("VALID: handshake uses the integrated G1 + dual BrainCo model")
        return 0

    arm = SimulatedArm(controls, enabled=not args.hand_only)
    configured_grip, neutral_target, hand_ramp = configured_hand_profile()
    curl_target = (
        list(args.curl_target)
        if args.curl_target is not None else configured_grip
    )
    hand = SimulatedHand(controls, curl_target, neutral_target, hand_ramp)
    cancelled = threading.Event()
    interrupted = threading.Event()
    states: list[str] = []
    failure: list[BaseException] = []

    def set_state(state: str) -> None:
        states.append(state)
        print(f"STATE: {state}", flush=True)

    sequence = AutomaticHandshake(
        arm, hand, args.duration, 0.0, cancelled, set_state,
        wait_for_grip=lambda: True)

    def run_sequence() -> None:
        try:
            sequence.prepare()
            sequence.run()
        except BaseException as exc:
            failure.append(exc)
            sequence.safe_state.release_once()

    worker = threading.Thread(target=run_sequence, daemon=True)
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame) -> None:
        if not interrupted.is_set():
            print("\nINTERRUPT: cancelling handshake safely", flush=True)
        interrupted.set()
        cancelled.set()

    signal.signal(signal.SIGINT, request_stop)
    worker.start()
    started = time.monotonic()

    def step() -> None:
        if (args.cancel_after is not None
                and time.monotonic() - started >= args.cancel_after):
            cancelled.set()
        backend.step()

    try:
        if args.headless:
            while worker.is_alive():
                step()
                time.sleep(model.opt.timestep)
        else:
            import mujoco.viewer

            frame_period = 1.0 / args.render_fps
            substeps = max(1, round(frame_period / model.opt.timestep))
            print(
                f"VIEWER: {args.render_fps:g} FPS, "
                f"{substeps} physics substeps/frame",
                flush=True,
            )
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and worker.is_alive():
                    frame_started = time.monotonic()
                    for _ in range(substeps):
                        step()
                    viewer.sync()
                    remaining = frame_period - (
                        time.monotonic() - frame_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
                if not viewer.is_running() and worker.is_alive():
                    cancelled.set()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    worker.join(timeout=(
        SIMULATED_PRESENTATION_SECONDS + 2.0 * hand_ramp + 1.0))
    if failure:
        raise failure[0]

    expected_terminal = (
        "CANCELLED"
        if args.cancel_after is not None or interrupted.is_set()
        else "COMPLETE"
    )
    if not states or states[-1] != expected_terminal:
        raise RuntimeError(
            f"expected terminal state {expected_terminal}, got {states[-1:]}")
    if controls.events[-2:] != ["hand:release", "arm:99"]:
        raise RuntimeError(f"unsafe cleanup order: {controls.events}")
    print("EVENTS: " + " -> ".join(controls.events))
    print("SIMULATION PASSED")
    return 130 if interrupted.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
