"""Adapt acknowledged G1 ROS commands to the project MuJoCo backend."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from g1_robot_interfaces.bridge_server import BridgeServer
from g1_robot_interfaces.arm_motion import validate_arm_motion
from g1_robot_interfaces.validation import (
    validate_hand_name,
    validate_hand_ramp_seconds,
    validate_hand_target,
)


def resolve_simulation_root(explicit_path=None):
    """Find the checkout's simulation directory or validate an override."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get("IEA_G1_SIMULATION_ROOT"):
        candidates.append(Path(os.environ["IEA_G1_SIMULATION_ROOT"]))
    candidates.extend(parent / "simulation" for parent in [
        Path.cwd(), *Path.cwd().parents,
    ])
    candidates.extend(parent / "simulation" for parent in Path(__file__).parents)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "run_handshake_demo.py").is_file():
            return resolved
    raise FileNotFoundError(
        "simulation backend not found; run from the IEA_G1 checkout or set "
        "IEA_G1_SIMULATION_ROOT")


def load_backend(simulation_root):
    """Load the MuJoCo implementation behind the ROS adapter boundary."""
    sys.path.insert(0, str(simulation_root))
    from run_handshake_demo import IntegratedG1Backend, SimulationControls
    return SimulationControls, IntegratedG1Backend


class G1SimulationAdapter(Node):
    """Validate ROS requests and dispatch them to named simulation methods."""

    def __init__(self, controls):
        super().__init__("g1_simulation_bridge")
        self.controls = controls
        self.declare_parameter(
            "right_hand_neutral_target",
            [0.399, 0.399, 0.050, 0.050, 0.049, 0.050])
        self.declare_parameter(
            "left_hand_neutral_target",
            [0.400, 0.400, 0.049, 0.050, 0.049, 0.050])
        self.neutral_targets = {
            "right": list(self.get_parameter(
                "right_hand_neutral_target").value),
            "left": list(self.get_parameter(
                "left_hand_neutral_target").value),
        }
        for hand, target in self.neutral_targets.items():
            self.neutral_targets[hand] = validate_hand_target(
                target, f"{hand} hand neutral")
        self.server = BridgeServer(self, "simulation", {
            "arm_prepare": self._prepare,
            "arm_motion_prepare": self._prepare,
            "arm_execute": self._arm_execute,
            "arm_motion": self._arm_motion,
            "arm_motion_release": self._arm_motion_release,
            "arm_interrupt": self._arm_interrupt,
            "arm_release": self._arm_release,
            "hand_prepare": self._prepare,
            "hand_command": self._hand_command,
            "hand_interrupt": self._hand_interrupt,
            "hand_neutral": self._hand_neutral,
        })

    def _prepare(self, request):
        del request
        return None

    def _arm_execute(self, request):
        action_id = int(request["action_id"])
        if action_id != 27:
            raise ValueError("only arm action 27 is implemented")
        self.controls.set_arm(action_id, f"ros:arm:{action_id}")
        return {"code": 0}

    def _arm_interrupt(self, request):
        del request
        return None

    def _arm_motion(self, request):
        """Accept the restricted neutral arm motion contract."""
        motion = validate_arm_motion(request['motion'])
        execution_timeout_s = float(request['execution_timeout_s'])
        if not math.isfinite(execution_timeout_s) or execution_timeout_s <= 0.0:
            raise ValueError('execution_timeout_s must be finite and positive')
        self.controls.set_arm_motion(motion)
        return {'mode': 'simulation',
                'trajectory_duration': sum(motion['phase_seconds'])}

    def _arm_release(self, request):
        del request
        self.controls.set_arm(99, "ros:arm:99")
        return {"code": 0}

    def _arm_motion_release(self, request):
        del request
        self.controls.set_arm(99, "ros:card-arm:release")
        return {"mode": "simulation", "released": True}

    def _hand_command(self, request):
        hand = validate_hand_name(request["hand"])
        target = validate_hand_target(request["target"])
        ramp = validate_hand_ramp_seconds(request["ramp_s"])
        self.controls.set_hand(
            hand, target, f"ros:{hand}_hand:command", ramp)
        return None

    def _hand_interrupt(self, request):
        del request
        return None

    def _hand_neutral(self, request):
        hand = validate_hand_name(request["hand"])
        ramp = validate_hand_ramp_seconds(request["ramp_s"])
        self.controls.set_hand(
            hand, self.neutral_targets[hand], f"ros:{hand}_hand:neutral",
            ramp)
        return None

    def safe_reset(self):
        """Return application-controlled targets to passive defaults."""
        for hand, target in self.neutral_targets.items():
            self.controls.set_hand(
                hand, target, f"bridge:shutdown:{hand}_hand_neutral")
        self.controls.set_arm(99, "bridge:shutdown:arm_release")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--simulation-root", type=Path)
    return parser.parse_known_args(args)


def main(args=None):
    parsed, ros_args = parse_args(args)
    if parsed.render_fps <= 0.0:
        raise ValueError("render-fps must be positive")
    simulation_root = resolve_simulation_root(parsed.simulation_root)
    controls_type, backend_type = load_backend(simulation_root)
    controls = controls_type()
    backend = backend_type(controls)
    if parsed.validate_only:
        print("VALID: g1_simulation_bridge adapter loaded the MuJoCo backend")
        print("VALID: backend-neutral arm and hand operations")
        return 0

    rclpy.init(args=ros_args)
    node = G1SimulationAdapter(controls)
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous_sigint = signal.signal(signal.SIGINT, stop)
    frame_period = 1.0 / parsed.render_fps
    substeps = max(1, round(frame_period / backend.model.opt.timestep))

    def frame():
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
        for _ in range(substeps):
            backend.step()

    try:
        if parsed.headless:
            while not interrupted and rclpy.ok():
                started = time.monotonic()
                frame()
                remaining = frame_period - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(
                    backend.model, backend.data) as viewer:
                while (viewer.is_running() and not interrupted
                       and rclpy.ok()):
                    started = time.monotonic()
                    frame()
                    viewer.sync()
                    remaining = frame_period - (time.monotonic() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
    finally:
        node.safe_reset()
        for _ in range(substeps):
            backend.step()
        signal.signal(signal.SIGINT, previous_sigint)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    # SIGINT from ros2 launch is an expected supervised shutdown.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
