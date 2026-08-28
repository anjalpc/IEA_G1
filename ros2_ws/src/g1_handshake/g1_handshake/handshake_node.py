"""ROS 2 wrapper for the verified G1 handshake and release actions."""

import json
import signal
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    # Available in newer ROS 2 releases (including Humble), but not Foxy.
    from rclpy.signals import SignalHandlerOptions
except ImportError:  # pragma: no cover - exercised on the robot's Foxy runtime
    SignalHandlerOptions = None

from g1_handshake.handshake_sequence import AutomaticHandshake
from g1_robot_interfaces.bridge_client import (
    BridgeArmBackend,
    BridgeHandBackend,
    BridgeTransport,
)


class HandshakeNode(Node):
    def __init__(self):
        super().__init__("event_handshake")
        self.declare_parameter("bridge_name", "unconfigured")
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("presentation_settle_s", 2.0)
        self.declare_parameter("grip_command_timeout_s", 15.0)
        self.declare_parameter("grip_hold_s", 5.0)
        self.bridge_name = str(self.get_parameter("bridge_name").value)
        timeout = float(self.get_parameter("action_timeout_s").value)
        if timeout <= 0.0:
            raise ValueError("action_timeout_s must be positive")
        settle = float(self.get_parameter("presentation_settle_s").value)
        if not 0.0 <= settle <= 10.0:
            raise ValueError("presentation_settle_s must be between 0 and 10")
        grip_command_timeout = float(
            self.get_parameter("grip_command_timeout_s").value)
        grip_hold = float(self.get_parameter("grip_hold_s").value)
        if not 5.0 <= grip_hold <= 8.0:
            raise ValueError("grip_hold_s must be between 5 and 8 seconds")
        if not 1.0 <= grip_command_timeout <= 60.0:
            raise ValueError(
                "grip_command_timeout_s must be between 1 and 60 seconds")
        self.transport = BridgeTransport(self, timeout, self.bridge_name)
        self.backend = BridgeArmBackend(self.transport)
        self.hand_backend = BridgeHandBackend(self.transport)
        self.state = "IDLE"
        self.cancelled = threading.Event()
        self.operation_lock = threading.Lock()
        self.worker = None
        self.grip_requested = threading.Event()
        self.grip_command_timeout = grip_command_timeout
        self.publisher = self.create_publisher(
            String, "/event/handshake/state", 10)
        self.create_service(Trigger, "/event/handshake/start",
                            self.start_callback)
        self.create_service(Trigger, "/event/handshake/cancel",
                            self.cancel_callback)
        self.create_service(Trigger, "/event/handshake/grip",
                            self.grip_callback)
        self.create_timer(0.5, self.publish_state)
        self.interaction = AutomaticHandshake(
            self.backend, self.hand_backend, grip_hold, settle,
            self.cancelled, self.set_state,
            wait_for_grip=self.wait_for_grip_command)
        self.get_logger().warning(
            f"Handshake application ready for bridge '{self.bridge_name}'")

    def publish_state(self):
        msg = String()
        msg.data = json.dumps({
            "state": self.state,
            "bridge": self.bridge_name,
        }, sort_keys=True)
        self.publisher.publish(msg)

    def grip_callback(self, request, response):
        """Accept grip only while a presented arm is waiting for it."""
        del request
        with self.operation_lock:
            if (self.worker is None or not self.worker.is_alive()
                    or self.state != "WAITING_FOR_GRIP"):
                response.success = False
                response.message = "handshake is not waiting for a grip command"
                return response
            if self.grip_requested.is_set():
                response.success = False
                response.message = "grip command already accepted"
                return response
            self.grip_requested.set()
        response.success = True
        response.message = "grip accepted"
        return response

    def wait_for_grip_command(self):
        """Wait for GRIP, cancellation, or the safety timeout."""
        deadline = time.monotonic() + self.grip_command_timeout
        while not self.grip_requested.wait(0.02):
            if self.cancelled.is_set():
                return False
            if time.monotonic() >= deadline:
                return False
        return not self.cancelled.is_set()

    def set_state(self, state):
        self.state = state
        self.publish_state()

    def start_callback(self, request, response):
        del request
        with self.operation_lock:
            if self.worker is not None and self.worker.is_alive():
                response.success = False
                response.message = "handshake already running"
                return response
            try:
                self.grip_requested.clear()
                self.interaction.prepare()
            except Exception as exc:
                response.success = False
                response.message = f"could not prepare handshake: {exc}"
                return response
            self.worker = threading.Thread(
                target=self.run_interaction_command,
                args=(self.interaction.run,), name="handshake-worker",
                daemon=True)
            self.worker.start()
        response.success = True
        response.message = "handshake start accepted; send grip when ready"
        return response

    def cancel_callback(self, request, response):
        del request
        with self.operation_lock:
            try:
                self.interaction.cancel()
            except Exception as exc:
                self.set_state("ERROR")
                response.success = False
                response.message = (
                    f"cancel requested, but safe release failed: {exc}")
                return response
            self.set_state("CANCELLED")
        response.success = True
        response.message = "cancelled; hand opened and arm released"
        return response

    def run_interaction_command(self, command):
        try:
            message = command()
            self.get_logger().info(message)
        except Exception as exc:
            if self.cancelled.is_set():
                try:
                    self.interaction.safe_state.release_once()
                except Exception as cleanup_exc:
                    self.get_logger().error(
                        f"safe release after cancellation failed: {cleanup_exc}")
                    self.set_state("ERROR")
                    return
                self.get_logger().info(f"handshake stopped: {exc}")
                self.set_state("CANCELLED")
                return
            self.get_logger().error(f"handshake failed: {exc}")
            try:
                self.interaction.safe_state.release_once()
            except Exception as cleanup_exc:
                self.get_logger().error(f"safe release failed: {cleanup_exc}")
            self.set_state("ERROR")

    def shutdown(self):
        # During a whole-launch shutdown the hardware bridge receives the same
        # signal and performs direct safe release. Do not wait for ROS replies
        # from a bridge that is shutting down concurrently.
        self.cancelled.set()
        self.grip_requested.set()
        worker = self.worker
        if (worker is not None and worker.is_alive()
                and worker is not threading.current_thread()):
            worker.join(timeout=1.0)
            if worker.is_alive():
                self.get_logger().warning(
                    "handshake worker is yielding to hardware bridge shutdown")


def main(args=None):
    # Keep ROS from invalidating the context before guarded release commands
    # are published. Foxy has no SignalHandlerOptions; replacing its installed
    # handlers immediately after init provides the same guarded shutdown path.
    if SignalHandlerOptions is None:
        rclpy.init(args=args)
    else:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HandshakeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.shutdown()
        executor.shutdown(timeout_sec=3.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
