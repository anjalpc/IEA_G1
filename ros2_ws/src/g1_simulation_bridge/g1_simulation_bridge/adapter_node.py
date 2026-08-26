"""Adapt acknowledged G1 ROS commands to the project MuJoCo backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from g1_robot_bridge.simulation_client import (
    SIM_COMMAND_TOPIC,
    SIM_FEEDBACK_TOPIC,
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
        self.feedback = self.create_publisher(String, SIM_FEEDBACK_TOPIC, 10)
        self.create_subscription(
            String, SIM_COMMAND_TOPIC, self.command_callback, 10)
        self.handlers = {
            "arm_action": self._arm_action,
            "hand_position": self._hand_position,
        }

    def _arm_action(self, request):
        action_id = int(request["action_id"])
        if action_id not in (27, 99):
            raise ValueError("only arm actions 27 and 99 are implemented")
        self.controls.set_arm(action_id, f"ros:arm:{action_id}")

    def _hand_position(self, request):
        if request.get("hand") != "right":
            raise ValueError("only the right BrainCo hand is implemented")
        positions = [float(value) for value in request["positions"]]
        if len(positions) != 6:
            raise ValueError("hand command must contain six positions")
        if any(value < 0.0 or value > 1.0 for value in positions):
            raise ValueError("hand positions must be within 0..1")
        self.controls.set_hand(positions, "ros:right_hand")

    def _reply(self, command_id, status, error=None):
        if not rclpy.ok():
            return
        message = String()
        response = {"command_id": command_id, "status": status}
        if error is not None:
            response["error"] = error
        message.data = json.dumps(response, sort_keys=True)
        self.feedback.publish(message)

    def command_callback(self, message):
        command_id = None
        try:
            request = json.loads(message.data)
            command_id = int(request["command_id"])
            interface = str(request["interface"])
            handler = self.handlers.get(interface)
            if handler is None:
                raise ValueError(f"unsupported simulation interface: {interface}")
            handler(request)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._reply(command_id, "error", str(exc))
            return
        self._reply(command_id, "ok")

    def safe_reset(self):
        """Return application-controlled targets to passive defaults."""
        self.controls.set_hand([0.0] * 6, "bridge:shutdown:hand_open")
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
        print("VALID: interfaces arm_action and hand_position")
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
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
