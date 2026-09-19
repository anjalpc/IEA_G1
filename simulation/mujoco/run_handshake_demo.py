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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INTERFACES_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "g1_robot_interfaces"
HANDSHAKE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "g1_handshake"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "handshake_surrogate.xml"
sys.path.insert(0, str(INTERFACES_SOURCE))
sys.path.insert(0, str(HANDSHAKE_SOURCE))

from g1_handshake.handshake_sequence import AutomaticHandshake  # noqa: E402
from g1_robot_interfaces.arm_motion import (  # noqa: E402
    resolve_arm_motion, validate_arm_motion)
from model_composition import (  # noqa: E402
    BODY_ACTUATOR_COUNT,
    build_combined_model,
    hand_actuator_names,
    validate_combined_model,
)
from run_g1_model import ElasticHarness, initialize  # noqa: E402


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
SIMULATED_PRESENTATION_SECONDS = 1.0


def configured_hand_profile() -> tuple[list[float], list[float], float]:
    """Read the handshake-owned target and simulation neutral posture."""
    configuration = REPOSITORY_ROOT / "ros2_ws" / "src" / \
        "g1_simulation_bridge" / "config" / "simulation.yaml"
    data = yaml.safe_load(configuration.read_text())
    simulation_parameters = data["g1_simulation_bridge"]["ros__parameters"]
    handshake_data = yaml.safe_load((REPOSITORY_ROOT / "ros2_ws" / "src" /
        "g1_handshake" / "config" / "handshake.yaml").read_text())
    parameters = handshake_data["event_handshake"]["ros__parameters"]
    return (list(parameters["grip_target"]),
            list(simulation_parameters["right_hand_neutral_target"]),
            float(parameters["grip_ramp_s"]))


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
        self.controls.set_hand(
            "right", self.curl_target, "hand:grip", self.ramp_seconds)
        time.sleep(self.ramp_seconds)

    def release(self) -> None:
        self.interrupt()
        self.controls.set_hand(
            "right", self.neutral_target, "hand:release", self.ramp_seconds)
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


class SurrogateBackend:
    """Explicit opt-in fixture retained only for lightweight debugging."""

    def __init__(self, controls: SimulationControls):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.controls = controls

    def step(self) -> None:
        action_id, hand_positions, _hand_ramps = self.controls.snapshot()
        if action_id == 27:
            arm = (-0.55, -1.05)
        else:
            arm = (0.0, 0.0)
        right_hand = hand_positions["right"]
        fixture_hand = [right_hand[0], *right_hand[2:6]]
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
        arm, hand, args.duration, 0.0, cancelled, set_state)

    def run_sequence() -> None:
        try:
            sequence.prepare()
            sequence.present()
            sequence.grip()
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
