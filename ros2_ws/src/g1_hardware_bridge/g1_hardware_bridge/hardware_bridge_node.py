"""Guarded ROS bridge that exclusively owns physical G1 commands."""

import signal
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

try:
    # Humble supports disabling rclpy's automatic signal handlers directly.
    from rclpy.signals import SignalHandlerOptions
except ImportError:  # Foxy on the robot
    SignalHandlerOptions = None

from g1_hardware_bridge.brainco_hand import BrainCoHand
from g1_hardware_bridge.unitree_arm import UnitreeArm
from g1_robot_interfaces.bridge_server import BridgeServer
from g1_robot_interfaces.constants import PRESENT_HAND_ACTION


class G1HardwareBridge(Node):
    """Validate and dispatch commands to the physical robot only."""

    def __init__(self):
        super().__init__("g1_hardware_bridge")
        self.declare_parameter("network_interface", "eth0")
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("grip_ramp_s", 0.75)
        self.declare_parameter("release_ramp_s", 0.75)
        self.declare_parameter(
            "hand_neutral_target", [0.399, 0.399, 0.050, 0.050, 0.049, 0.050])
        self.declare_parameter(
            "hand_grip_target", [0.48, 0.45, 0.15, 0.20, 0.25, 0.20])
        interface = str(self.get_parameter("network_interface").value)
        timeout = float(self.get_parameter("action_timeout_s").value)
        neutral = list(self.get_parameter("hand_neutral_target").value)
        grip = list(self.get_parameter("hand_grip_target").value)
        grip_ramp = float(self.get_parameter("grip_ramp_s").value)
        release_ramp = float(self.get_parameter("release_ramp_s").value)
        if timeout <= 0.0:
            raise ValueError("action_timeout_s must be positive")
        if not 0.5 <= grip_ramp <= 5.0:
            raise ValueError("grip_ramp_s must be between 0.5 and 5.0")
        if not 0.5 <= release_ramp <= 5.0:
            raise ValueError("release_ramp_s must be between 0.5 and 5.0")
        if not interface.strip():
            raise ValueError("network_interface must not be empty")
        self._validate_targets(neutral, grip)
        self.arm = UnitreeArm(interface, timeout)
        self.hand = BrainCoHand(
            interface, timeout, neutral, grip, grip_ramp, release_ramp)
        self.server = BridgeServer(self, "hardware", {
            "arm_prepare": self._arm_prepare,
            "arm_action": self._arm_action,
            "arm_interrupt": self._arm_interrupt,
            "arm_release": self._arm_release,
            "hand_prepare": self._hand_prepare,
            "hand_grip": self._hand_grip,
            "hand_interrupt": self._hand_interrupt,
            "hand_release": self._hand_release,
        })
        self.get_logger().warning("Physical arm and hand bridge ready")

    @staticmethod
    def _validate_targets(neutral, grip):
        if len(neutral) != 6 or len(grip) != 6:
            raise ValueError("hand targets must contain six positions")
        if any(value < 0.0 or value > 1.0 for value in neutral):
            raise ValueError("hand neutral target must be between 0 and 1")
        if any(value < 0.0 or value > 1.0 for value in grip):
            raise ValueError("hand grip target must be between 0 and 1")

    def _arm_prepare(self, request):
        del request
        prepare = getattr(self.arm, "prepare_operation", None)
        if prepare is not None:
            prepare()

    def _arm_action(self, request):
        action_id = int(request["action_id"])
        if action_id != PRESENT_HAND_ACTION:
            raise ValueError("only reviewed arm action 27 is accepted")
        return {"code": self.arm.execute(action_id)}

    def _arm_interrupt(self, request):
        del request
        self.arm.interrupt()

    def _arm_release(self, request):
        del request
        return {"code": self.arm.release()}

    def _hand_prepare(self, request):
        del request
        prepare = getattr(self.hand, "prepare_operation", None)
        if prepare is not None:
            prepare()

    def _hand_grip(self, request):
        del request
        return self.hand.grip()

    def _hand_interrupt(self, request):
        del request
        self.hand.interrupt()

    def _hand_release(self, request):
        del request
        self.hand.release()

    def safe_shutdown(self):
        """Return the physical hand and arm to their guarded safe state."""
        started = time.monotonic()
        errors = []

        for name, command in (
            ("hand", self.hand.release),
            ("arm", self.arm.release),
        ):
            try:
                command()
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        elapsed = time.monotonic() - started
        if errors:
            self.get_logger().error(
                "physical bridge safe shutdown incomplete after "
                f"{elapsed:.3f}s: " + "; ".join(errors)
            )
        else:
            self.get_logger().info(
                "physical bridge safe shutdown complete in "
                f"{elapsed:.3f}s"
            )


def main(args=None):
    """Run with explicit signals so guarded hardware shutdown always executes."""
    if SignalHandlerOptions is None:
        rclpy.init(args=args)
    else:
        rclpy.init(
            args=args,
            signal_handler_options=SignalHandlerOptions.NO,
        )

    node = G1HardwareBridge()
    executor = MultiThreadedExecutor(num_threads=4)
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
        node.safe_shutdown()

        if not executor.shutdown(timeout_sec=10.0):
            node.get_logger().warning(
                "hardware bridge executor did not stop within 10 seconds"
            )

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
