"""ROS 2 wrapper for the verified G1 handshake and release actions."""

import json
import signal
import threading

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
from g1_handshake.handshake_backends import HandshakeHand
from g1_robot_interfaces.bridge_client import (
    ArmClient,
    HandClient,
    BridgeTransport,
)
from g1_robot_interfaces.validation import (
    validate_hand_name,
    validate_hand_ramp_seconds,
    validate_hand_target,
)


class HandshakeNode(Node):
    def __init__(self):
        super().__init__("event_handshake")
        self.declare_parameter("bridge_name", "unconfigured")
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("presentation_settle_s", 2.0)
        self.declare_parameter("presentation_timeout_s", 15.0)
        self.declare_parameter("grip_hold_s", 2.0)
        self.declare_parameter("presentation_action_id", 27)
        self.declare_parameter("hand", "right")
        self.declare_parameter(
            "grip_target", [0.48, 0.45, 0.15, 0.20, 0.25, 0.20])
        self.declare_parameter("grip_ramp_s", 0.2)
        self.declare_parameter("release_ramp_s", 0.2)
        self.bridge_name = str(self.get_parameter("bridge_name").value)
        timeout = float(self.get_parameter("action_timeout_s").value)
        if timeout <= 0.0:
            raise ValueError("action_timeout_s must be positive")
        settle = float(self.get_parameter("presentation_settle_s").value)
        if not 0.0 <= settle <= 10.0:
            raise ValueError("presentation_settle_s must be between 0 and 10")
        presentation_timeout = float(
            self.get_parameter("presentation_timeout_s").value)
        grip_hold = float(self.get_parameter("grip_hold_s").value)
        presentation_action = int(
            self.get_parameter("presentation_action_id").value)
        hand = validate_hand_name(self.get_parameter("hand").value)
        grip_target = validate_hand_target(
            self.get_parameter("grip_target").value, "grip_target")
        grip_ramp = validate_hand_ramp_seconds(
            self.get_parameter("grip_ramp_s").value, "grip_ramp_s")
        release_ramp = validate_hand_ramp_seconds(
            self.get_parameter("release_ramp_s").value, "release_ramp_s")
        if not 1.0 <= grip_hold <= 8.0:
            raise ValueError("grip_hold_s must be between 1.0 and 8 seconds")
        if not 5.0 <= presentation_timeout <= 60.0:
            raise ValueError(
                "presentation_timeout_s must be between 5 and 60 seconds")
        self.transport = BridgeTransport(self, timeout, self.bridge_name)
        self.backend = ArmClient(self.transport)
        self.hand_backend = HandshakeHand(
            HandClient(self.transport), hand, grip_target,
            grip_ramp, release_ramp)
        self.state = "IDLE"
        self.arm_ownership = (
            "IDLE" if self.bridge_name != "hardware" else "UNAVAILABLE")
        self.arm_action_id = None
        self.arm_action_name = ""
        self.cancelled = threading.Event()
        self.yield_requested = threading.Event()
        self.operation_lock = threading.Lock()
        self.worker = None
        self.presentation_timeout = presentation_timeout
        self.presentation_timer = None
        self.shutting_down = threading.Event()
        self.publisher = self.create_publisher(
            String, "/event/handshake/state", 10)
        self.create_subscription(
            String, "/g1/bridge/hardware/arm_ownership",
            self.arm_ownership_callback, 10)
        self.create_service(Trigger, "/event/handshake/start",
                            self.start_callback)
        self.create_service(Trigger, "/event/handshake/cancel",
                            self.cancel_callback)
        self.create_service(Trigger, "/event/handshake/grip",
                            self.grip_callback)
        self.state_timer = self.create_timer(
            0.5, self.publish_state)
        self.interaction = AutomaticHandshake(
            self.backend, self.hand_backend, grip_hold, settle,
            self.cancelled, self.set_state,
            yield_requested=self.yield_requested,
            presentation_action=presentation_action)
        self.get_logger().warning(
            f"Handshake application ready for bridge '{self.bridge_name}'")

    def publish_state(self):
        if self.shutting_down.is_set():
            return
        msg = String()
        msg.data = json.dumps({
            "state": self.state,
            "bridge": self.bridge_name,
            "arm_ownership": self.arm_ownership,
            "arm_action_id": self.arm_action_id,
            "arm_action_name": self.arm_action_name,
        }, sort_keys=True)
        self.publisher.publish(msg)

    def arm_ownership_callback(self, message):
        """Consume bridge ownership and yield safely to external actions."""
        if self.bridge_name != "hardware":
            return
        try:
            state = json.loads(message.data)
            status = str(state["status"])
            action_id = state.get("id")
            action_name = str(state.get("name", ""))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"invalid arm ownership state: {exc}")
            status = "UNAVAILABLE"
            action_id = None
            action_name = ""

        self.arm_ownership = status
        self.arm_action_id = action_id
        self.arm_action_name = action_name

        if status == "IDLE":
            with self.operation_lock:
                worker_running = (
                    self.worker is not None and self.worker.is_alive())
                if (self.state == "EXTERNAL_ARM_CONTROL"
                        and not worker_running):
                    self.set_state("IDLE")
            return

        if status not in ("EXTERNAL", "UNAVAILABLE"):
            return
        if self.state not in {
                "PRESENTING", "ARM_PRESENTED", "GRIPPING", "HOLDING"}:
            return

        self.cancel_presentation_timer()
        try:
            self.interaction.request_external_yield()
        except Exception as exc:
            self.get_logger().warning(
                f"could not interrupt active hand command: {exc}")
        with self.operation_lock:
            if self.worker is not None and self.worker.is_alive():
                return
            self.worker = threading.Thread(
                target=self.run_interaction_command,
                args=(self.interaction.yield_arm_control,),
                name="handshake-external-yield-worker",
                daemon=True)
            self.worker.start()

    def arm_command_allowed(self, allowed):
        """Return an operator-facing reason when arm ownership blocks work."""
        if self.bridge_name != "hardware":
            return None
        if self.arm_ownership in allowed:
            return None
        if self.arm_ownership == "EXTERNAL":
            return (
                "external Unitree arm action is active: "
                f"{self.arm_action_id} {self.arm_action_name}".rstrip())
        return "Unitree arm state is unavailable or not stably idle"

    def cancel_presentation_timer(self):
        """Cancel the pending dashboard presentation timeout."""
        timer = self.presentation_timer
        self.presentation_timer = None
        if timer is not None:
            timer.cancel()

    def schedule_presentation_timeout(self):
        """Schedule safe release of an unused dashboard presentation."""
        self.cancel_presentation_timer()
        timer = threading.Timer(
            self.presentation_timeout,
            self.presentation_timeout_callback)
        timer.daemon = True
        self.presentation_timer = timer
        timer.start()

    def presentation_timeout_callback(self):
        """Release an arm left presented by the dashboard."""
        current = threading.current_thread()
        with self.operation_lock:
            self.presentation_timer = None
            if self.state != "ARM_PRESENTED":
                return
            if self.worker is not None and self.worker.is_alive():
                return
            self.worker = current

        try:
            self.run_interaction_command(
                self.interaction.expire_presentation)
        finally:
            with self.operation_lock:
                if self.worker is current:
                    self.worker = None

    def run_present_command(self):
        """Raise the arm and start its automatic-release timer."""
        self.run_interaction_command(self.interaction.present)
        with self.operation_lock:
            if (self.state == "ARM_PRESENTED"
                    and not self.cancelled.is_set()):
                self.schedule_presentation_timeout()

    def run_release_command(self):
        """Lower an already-presented arm without starting a new operation."""
        self.run_interaction_command(self.interaction.release_presentation)

    def grip_callback(self, request, response):
        """Start a grip independently of how the arm was raised."""
        del request
        with self.operation_lock:
            blocked = self.arm_command_allowed({"IDLE", "OWNED"})
            if blocked is not None:
                response.success = False
                response.message = blocked
                return response
            if self.worker is not None and self.worker.is_alive():
                response.success = False
                response.message = "another handshake operation is running"
                return response

            self.cancel_presentation_timer()
            try:
                self.interaction.prepare()
            except Exception as exc:
                response.success = False
                response.message = f"could not prepare grip: {exc}"
                return response

            self.worker = threading.Thread(
                target=self.run_interaction_command,
                args=(self.interaction.grip,),
                name="handshake-grip-worker",
                daemon=True)
            self.worker.start()

        response.success = True
        response.message = "grip accepted"
        return response

    def set_state(self, state):
        self.state = state
        self.publish_state()

    def start_callback(self, request, response):
        """Toggle the dashboard-controlled arm between presented and safe."""
        del request
        with self.operation_lock:
            if self.worker is not None and self.worker.is_alive():
                response.success = False
                response.message = "another handshake operation is running"
                return response
            if self.state == "ARM_PRESENTED":
                blocked = self.arm_command_allowed({"IDLE", "OWNED"})
                if blocked is not None:
                    response.success = False
                    response.message = blocked
                    return response
                self.cancel_presentation_timer()
                self.worker = threading.Thread(
                    target=self.run_release_command,
                    name="handshake-release-worker",
                    daemon=True)
                self.worker.start()
                response.success = True
                response.message = "arm release accepted"
                return response

            blocked = self.arm_command_allowed({"IDLE"})
            if blocked is not None:
                response.success = False
                response.message = blocked
                return response

            self.cancel_presentation_timer()
            try:
                self.interaction.prepare()
            except Exception as exc:
                response.success = False
                response.message = (
                    f"could not prepare arm presentation: {exc}")
                return response

            self.worker = threading.Thread(
                target=self.run_present_command,
                name="handshake-present-worker",
                daemon=True)
            self.worker.start()

        response.success = True
        response.message = "arm presentation accepted"
        return response

    def cancel_callback(self, request, response):
        del request
        self.cancel_presentation_timer()
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
            if self.yield_requested.is_set():
                try:
                    message = self.interaction.yield_arm_control()
                except Exception as cleanup_exc:
                    self.get_logger().error(
                        "hand-only release after external takeover failed: "
                        f"{cleanup_exc}")
                    self.set_state("ERROR")
                    return
                self.get_logger().warning(message)
                return
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
        self.shutting_down.set()
        self.state_timer.cancel()

        # During a whole-launch shutdown the hardware bridge receives the same
        # signal and performs direct safe release. Do not wait for ROS replies
        # from a bridge that is shutting down concurrently.
        self.cancel_presentation_timer()
        self.cancelled.set()
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
