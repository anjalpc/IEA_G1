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
from g1_hardware_bridge.settings import (
    DEFAULT_SAFE_HAND_RAMP_S,
)
from g1_hardware_bridge.unitree_arm import UnitreeArm
from g1_robot_interfaces.bridge_server import BridgeServer
from g1_robot_interfaces.validation import (
    validate_hand_name,
    validate_hand_ramp_seconds,
    validate_hand_target,
)


class G1HardwareBridge(Node):
    """Validate and dispatch commands to the physical robot only."""

    def __init__(self):
        super().__init__("g1_hardware_bridge")
        self.declare_parameter("network_interface", "eth0")
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("safe_hand_ramp_s", DEFAULT_SAFE_HAND_RAMP_S)
        self.declare_parameter("allowed_arm_actions", [27])
        self.declare_parameter("card_presentation_mode", "disabled")
        self.declare_parameter(
            "right_hand_neutral_target",
            [0.399, 0.399, 0.050, 0.050, 0.049, 0.050])
        self.declare_parameter(
            "left_hand_neutral_target",
            [0.400, 0.400, 0.049, 0.050, 0.049, 0.050])
        interface = str(self.get_parameter("network_interface").value)
        timeout = float(self.get_parameter("action_timeout_s").value)
        self.safe_hand_ramp = validate_hand_ramp_seconds(
            self.get_parameter("safe_hand_ramp_s").value,
            "safe_hand_ramp_s")
        self.allowed_arm_actions = {
            int(value) for value in
            self.get_parameter("allowed_arm_actions").value}
        self.neutral_targets = {
            "right": list(self.get_parameter(
                "right_hand_neutral_target").value),
            "left": list(self.get_parameter(
                "left_hand_neutral_target").value),
        }
        if timeout <= 0.0:
            raise ValueError("action_timeout_s must be positive")
        if not interface.strip():
            raise ValueError("network_interface must not be empty")
        if not self.allowed_arm_actions:
            raise ValueError("allowed_arm_actions must not be empty")
        for hand, target in self.neutral_targets.items():
            self.neutral_targets[hand] = validate_hand_target(
                target, f"{hand} hand neutral")
        self.card_presentation_mode = str(
            self.get_parameter("card_presentation_mode").value)
        self.arm = UnitreeArm(
            interface, timeout,
            self.card_presentation_mode)
        self.hand = BrainCoHand(interface, timeout)
        self._owned_hands = set()
        self.server = BridgeServer(self, "hardware", {
            "arm_prepare": self._arm_prepare,
            "arm_prepare_presentation": self._arm_prepare_presentation,
            "arm_execute": self._arm_execute,
            "arm_present": self._arm_present,
            "arm_present_release": self._arm_present_release,
            "arm_interrupt": self._arm_interrupt,
            "arm_release": self._arm_release,
            "hand_prepare": self._hand_prepare,
            "hand_command": self._hand_command,
            "hand_interrupt": self._hand_interrupt,
            "hand_neutral": self._hand_neutral,
        })
        self.get_logger().warning("Physical arm and hand bridge ready")

    @staticmethod
    def _request_hand(request):
        return validate_hand_name(request["hand"])

    def _arm_prepare(self, request):
        del request
        prepare = getattr(self.arm, "prepare_operation", None)
        if prepare is not None:
            prepare()

    def _arm_prepare_presentation(self, request):
        del request
        self.arm.prepare_presentation()

    def _arm_execute(self, request):
        action_id = int(request["action_id"])
        if action_id not in self.allowed_arm_actions:
            raise ValueError(f"arm action {action_id} is not allowed")
        return {"code": self.arm.execute(action_id)}

    def _arm_present(self, request):
        return self.arm.present(
            str(request["pose"]), float(request["fraction"]),
            float(request["move_s"]))

    def _arm_present_release(self, request):
        del request
        return self.arm.release_presentation()

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

    def _hand_command(self, request):
        hand = self._request_hand(request)
        target = validate_hand_target(request["target"])
        ramp = validate_hand_ramp_seconds(request["ramp_s"], "ramp_s")
        if (getattr(self, "card_presentation_mode", "disabled") ==
                "offline_shadow"
                and hand == "left"):
            return {"mode": "offline_shadow", "published": False}
        self._owned_hands.add(hand)
        result = self.hand.command_position(hand, target, ramp)
        return result

    def _hand_interrupt(self, request):
        del request
        self.hand.interrupt()

    def _hand_neutral(self, request):
        hand = self._request_hand(request)
        ramp = validate_hand_ramp_seconds(request["ramp_s"], "ramp_s")
        if (getattr(self, "card_presentation_mode", "disabled") ==
                "offline_shadow"
                and hand == "left"):
            return {"mode": "offline_shadow", "published": False}
        result = self.hand.neutral(
            hand, self.neutral_targets[hand], ramp)
        self._owned_hands.discard(hand)
        return result

    def safe_shutdown(self):
        """Return the physical hand and arm to their guarded safe state."""
        started = time.monotonic()
        errors = []

        commands = [
            (f"{hand} hand", lambda selected=hand: self.hand.neutral(
                selected, self.neutral_targets[selected],
                self.safe_hand_ramp))
            for hand in sorted(self._owned_hands)
        ]
        commands.append(("arm", self.arm.release))
        for name, command in commands:
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
