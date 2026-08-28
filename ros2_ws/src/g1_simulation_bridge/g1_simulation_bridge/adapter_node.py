"""Adapt acknowledged G1 ROS commands to the project MuJoCo backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from g1_robot_interfaces.bridge_server import BridgeServer


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
            "hand_neutral_target", [0.399, 0.399, 0.050, 0.050, 0.049, 0.050])
        self.declare_parameter(
            "hand_grip_target", [0.48, 0.45, 0.15, 0.20, 0.25, 0.20])
        self.neutral_target = list(
            self.get_parameter("hand_neutral_target").value)
        self.grip_target = list(self.get_parameter("hand_grip_target").value)
        for target in (self.neutral_target, self.grip_target):
            if len(target) != 6 or any(value < 0.0 or value > 1.0
                                       for value in target):
                raise ValueError("configured hand targets must be six values in 0..1")
        self.server = BridgeServer(self, "simulation", {
            "arm_prepare": self._prepare,
            "arm_action": self._arm_action,
            "arm_interrupt": self._arm_interrupt,
            "arm_release": self._arm_release,
            "hand_prepare": self._prepare,
            "hand_grip": self._hand_grip,
            "hand_interrupt": self._hand_interrupt,
            "hand_release": self._hand_release,
        })

    def _prepare(self, request):
        del request
        return None

    def _arm_action(self, request):
        action_id = int(request["action_id"])
        if action_id != 27:
            raise ValueError("only arm action 27 is implemented")
        self.controls.set_arm(action_id, f"ros:arm:{action_id}")
        return {"code": 0}

    def _arm_interrupt(self, request):
        del request
        return None

    def _arm_release(self, request):
        del request
        self.controls.set_arm(99, "ros:arm:99")
        return {"code": 0}

    def _hand_grip(self, request):
        del request
        self.controls.set_hand(self.grip_target, "ros:right_hand:grip")
        return None

    def _hand_interrupt(self, request):
        del request
        return None

    def _hand_release(self, request):
        del request
        self.controls.set_hand(self.neutral_target, "ros:right_hand:release")
        return None

    def safe_reset(self):
        """Return application-controlled targets to passive defaults."""
        self.controls.set_hand(
            self.neutral_target, "bridge:shutdown:hand_neutral")
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
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
