"""ROS 2 wrapper for the verified G1 handshake and release actions."""

import json
import os
import subprocess
import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from g1_robot_bridge.handshake_sequence import HandshakeSequence, RELEASE_ARM
from g1_robot_bridge.simulation_client import (
    SimulationArmActionBackend,
    SimulationBrainCoBackend,
    SimulationTransport,
)


class DryRunBackend:
    def execute(self, action_id):
        print(f"DRY RUN: Unitree arm action {action_id}", flush=True)
        return 0


class DryRunHandBackend:
    def grip(self):
        print("DRY RUN: BrainCo right hand grip", flush=True)

    def release(self):
        print("DRY RUN: BrainCo right hand release", flush=True)


class BrainCoBackend:
    """Run native BrainCo DDS commands outside the ROS-domain process."""

    def __init__(self, interface, timeout, grip_amount):
        self.interface = interface
        self.timeout = timeout
        self.grip_amount = grip_amount
        self.lock = threading.Lock()

    def _execute(self, command, amount):
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = "0"
        result = subprocess.run(
            [sys.executable, "-m", "g1_robot_bridge.brainco_hand_helper",
             "command-position", "--hand", "right",
             "--interface", self.interface, "--timeout", str(self.timeout),
             "--amount", str(amount), "--ramp-seconds", "1.5",
             "--confirm", "HANDSHAKE_OPERATOR_ENABLED"],
            env=env, capture_output=True, text=True,
            timeout=self.timeout + 5.0, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"BrainCo {command} failed: {result.stderr.strip()}")

    def grip(self):
        with self.lock:
            self._execute("grip", self.grip_amount)

    def release(self):
        with self.lock:
            self._execute("release", 0.0)


class UnitreeBackend:
    def __init__(self, interface, timeout):
        self.interface = interface
        self.timeout = timeout

    def execute(self, action_id):
        # ROS 2 runs on domain 42 to avoid native Unitree DDS type collisions.
        # Execute the Unitree domain-0 RPC in a separate process so the two DDS
        # participants never share CycloneDDS global state.
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = "0"
        result = subprocess.run(
            [sys.executable, "-m", "g1_robot_bridge.unitree_action_helper",
             self.interface, str(self.timeout), str(action_id)],
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout + 3.0,
            check=False,
        )
        marker = "ACTION_RESULT="
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(marker):
                return int(line[len(marker):])
        raise RuntimeError(
            "Unitree action helper failed: "
            f"rc={result.returncode} stderr={result.stderr.strip()}")


class HandshakeNode(Node):
    def __init__(self):
        super().__init__("event_handshake")
        self.declare_parameter("hardware_enabled", False)
        self.declare_parameter("simulation_enabled", False)
        self.declare_parameter("network_interface", "eth0")
        self.declare_parameter("action_duration_s", 2.0)
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("hand_control_enabled", False)
        self.declare_parameter("hand_grip_amount", 0.12)
        self.hardware = bool(self.get_parameter("hardware_enabled").value)
        self.simulation = bool(self.get_parameter("simulation_enabled").value)
        if self.hardware and self.simulation:
            raise ValueError(
                "hardware_enabled and simulation_enabled are mutually exclusive")
        interface = str(self.get_parameter("network_interface").value)
        self.duration = float(self.get_parameter("action_duration_s").value)
        timeout = float(self.get_parameter("action_timeout_s").value)
        self.hand_control = bool(
            self.get_parameter("hand_control_enabled").value)
        self.hand_control_active = (
            self.simulation or (self.hardware and self.hand_control))
        grip_amount = float(self.get_parameter("hand_grip_amount").value)
        if not 0.0 < grip_amount <= 0.15:
            raise ValueError("hand_grip_amount must be > 0 and <= 0.15")
        self.sim_transport = (
            SimulationTransport(self, timeout) if self.simulation else None)
        if self.simulation:
            self.backend = SimulationArmActionBackend(self.sim_transport)
            self.hand_backend = SimulationBrainCoBackend(
                self.sim_transport, grip_amount)
        else:
            self.backend = (UnitreeBackend(interface, timeout)
                            if self.hardware else DryRunBackend())
            self.hand_backend = (
                BrainCoBackend(interface, timeout, grip_amount)
                if self.hardware and self.hand_control else DryRunHandBackend())
        self.enabled = False
        self.state = "IDLE"
        self.cancelled = threading.Event()
        self.action_lock = threading.Lock()
        self.worker = None
        self.publisher = self.create_publisher(
            String, "/event/handshake/state", 10)
        self.create_service(SetBool, "/event/handshake/operator_enable",
                            self.enable_callback)
        self.create_service(Trigger, "/event/handshake/start",
                            self.start_callback)
        self.create_service(Trigger, "/event/handshake/cancel",
                            self.cancel_callback)
        self.create_timer(0.5, self.publish_state)
        mode = ("HARDWARE" if self.hardware else
                "SIMULATION" if self.simulation else "DRY RUN")
        self.get_logger().warning(
            f"Handshake node ready in {mode} mode; BrainCo grip is "
            f"{'enabled' if self.hand_control_active else 'dry-run'}")

    def publish_state(self):
        msg = String()
        msg.data = json.dumps({
            "state": self.state,
            "operator_enabled": self.enabled,
            "hardware_enabled": self.hardware,
            "simulation_enabled": self.simulation,
            "hand_control_enabled": self.hand_control_active,
        }, sort_keys=True)
        self.publisher.publish(msg)

    def set_state(self, state):
        self.state = state
        self.publish_state()

    def execute(self, action_id):
        with self.action_lock:
            code = self.backend.execute(action_id)
        if code != 0:
            raise RuntimeError(f"action {action_id} returned {code}")

    def release(self):
        try:
            self.hand_backend.release()
        except Exception as exc:
            self.get_logger().error(f"release_hand failed: {exc}")
        try:
            self.execute(RELEASE_ARM)
        except Exception as exc:
            self.get_logger().error(f"release_arm failed: {exc}")

    def enable_callback(self, request, response):
        self.enabled = bool(request.data)
        if not self.enabled:
            self.cancelled.set()
            self.release()
            self.set_state("IDLE")
        response.success = True
        response.message = "enabled" if self.enabled else "disabled; arm released"
        return response

    def start_callback(self, request, response):
        del request
        if not self.enabled:
            response.success = False
            response.message = "operator enable is required"
            return response
        if self.worker is not None and self.worker.is_alive():
            response.success = False
            response.message = "handshake already running"
            return response
        self.cancelled.clear()
        self.worker = threading.Thread(target=self.run_handshake, daemon=True)
        self.worker.start()
        response.success = True
        response.message = "handshake accepted"
        return response

    def cancel_callback(self, request, response):
        del request
        self.cancelled.set()
        self.release()
        self.set_state("CANCELLED")
        response.success = True
        response.message = "cancel accepted; arm released"
        return response

    def run_handshake(self):
        try:
            HandshakeSequence(
                self.backend, self.hand_backend, self.duration,
                self.cancelled, lambda: self.enabled, self.set_state).run()
        except Exception as exc:
            self.get_logger().error(f"handshake failed: {exc}")
            self.release()
            self.set_state("ERROR")

    def shutdown(self):
        self.cancelled.set()
        self.release()


def main(args=None):
    # Keep ROS from invalidating the context before guarded release commands
    # are published. Python's normal SIGINT handling raises KeyboardInterrupt.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HandshakeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.shutdown()
        executor.shutdown(timeout_sec=3.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
