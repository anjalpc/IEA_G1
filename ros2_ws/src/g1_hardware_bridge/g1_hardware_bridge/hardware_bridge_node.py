"""Guarded ROS bridge that exclusively owns physical G1 commands."""

import json
import math
import signal
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

try:
    # Humble supports disabling rclpy's automatic signal handlers directly.
    from rclpy.signals import SignalHandlerOptions
except ImportError:  # Foxy on the robot
    SignalHandlerOptions = None

from g1_hardware_bridge.arm.unitree_arm import UnitreeArm
from g1_hardware_bridge.arm.arm_state_monitor import UnitreeArmMonitor
from g1_hardware_bridge.arm.ownership import ArmOwnership
from g1_hardware_bridge.hand.backend import BrainCoHand
from g1_hardware_bridge.native_runtime import (
    native_dds_environment,
    network_interface,
)
from g1_hardware_bridge.settings import (
    DEFAULT_SAFE_HAND_RAMP_S,
)
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
        self.declare_parameter("network_interface", network_interface())
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("safe_hand_ramp_s", DEFAULT_SAFE_HAND_RAMP_S)
        self.declare_parameter("arm_state_stale_timeout_s", 2.0)
        self.declare_parameter("arm_idle_stable_s", 0.5)
        self.declare_parameter("allowed_arm_actions", [27])
        self.declare_parameter("arm_motion_mode", "disabled")
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
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("action_timeout_s must be positive")
        if not interface.strip():
            raise ValueError("network_interface must not be empty")
        if not self.allowed_arm_actions:
            raise ValueError("allowed_arm_actions must not be empty")
        for hand, target in self.neutral_targets.items():
            self.neutral_targets[hand] = validate_hand_target(
                target, f"{hand} hand neutral")
        self.arm_motion_mode = str(
            self.get_parameter("arm_motion_mode").value)
        self.arm = UnitreeArm(
            interface, timeout,
            self.arm_motion_mode)
        self.hand = BrainCoHand(interface, timeout)
        self._arm_lease_lock = threading.Lock()
        self._arm_lease_client = None
        self._owned_hands = set()
        stale_timeout = float(
            self.get_parameter("arm_state_stale_timeout_s").value)
        idle_stable = float(
            self.get_parameter("arm_idle_stable_s").value)
        self.arm_ownership = ArmOwnership(stale_timeout, idle_stable)
        self.arm_ownership_publisher = self.create_publisher(
            String, "/g1/bridge/hardware/arm_ownership", 10)
        self.arm_monitor = UnitreeArmMonitor(
            self.arm_ownership,
            native_dds_environment(),
            state_callback=self._publish_arm_ownership)
        self.arm_monitor.start()
        self.arm_ownership_timer = self.create_timer(
            0.2, self._publish_arm_ownership)
        self.server = BridgeServer(self, "hardware", {
            "arm_prepare": self._arm_prepare,
            "arm_motion_prepare": self._arm_prepare_motion,
            "arm_execute": self._arm_execute,
            "arm_motion": self._arm_motion,
            "arm_motion_release": self._arm_motion_release,
            "arm_interrupt": self._arm_interrupt,
            "arm_release": self._arm_release,
            "hand_prepare": self._hand_prepare,
            "hand_command": self._hand_command,
            "hand_interrupt": self._hand_interrupt,
            "hand_neutral": self._hand_neutral,
        })
        self.get_logger().warning("Physical arm and hand bridge ready")

    def _publish_arm_ownership(self, state=None):
        """Publish native arm ownership on the application ROS domain."""
        if state is None:
            state = self.arm_ownership.snapshot()
        message = String()
        message.data = json.dumps(state, sort_keys=True)
        self.arm_ownership_publisher.publish(message)

    def stop_arm_monitor(self):
        """Stop and reap the isolated native arm-state reader."""
        timer = getattr(self, "arm_ownership_timer", None)
        if timer is not None:
            timer.cancel()
        monitor = getattr(self, "arm_monitor", None)
        if monitor is not None:
            monitor.stop()

    @staticmethod
    def _request_hand(request):
        return validate_hand_name(request["hand"])

    def _acquire_arm_lease(self, request, allow_owned=False):
        """Atomically admit one bridge client for a complete arm operation."""
        client_id = str(request["client_id"])
        with self._arm_lease_lock:
            if self._arm_lease_client not in (None, client_id):
                raise RuntimeError("arm is leased by another bridge client")
            state = self.arm_ownership.snapshot()
            accepted = (ArmOwnership.IDLE,)
            if allow_owned and self._arm_lease_client == client_id:
                accepted = (ArmOwnership.IDLE, ArmOwnership.OWNED)
            if state["status"] not in accepted:
                raise RuntimeError(ArmOwnership.block_reason(state))
            self._arm_lease_client = client_id
        return client_id

    def _require_arm_lease(self, request):
        client_id = str(request["client_id"])
        with self._arm_lease_lock:
            if self._arm_lease_client != client_id:
                state = self.arm_ownership.snapshot()
                if state["status"] not in (
                        ArmOwnership.IDLE, ArmOwnership.OWNED):
                    raise RuntimeError(ArmOwnership.block_reason(state))
                raise RuntimeError("arm operation requires this client's lease")
            state = self.arm_ownership.snapshot()
            if state["status"] not in (
                    ArmOwnership.IDLE, ArmOwnership.OWNED) and not (
                        state["status"] == ArmOwnership.UNAVAILABLE
                        and self.arm.has_motion()
                    ):
                raise RuntimeError(ArmOwnership.block_reason(state))
        return client_id

    def _release_arm_lease(self, client_id=None):
        with self._arm_lease_lock:
            if client_id is None or self._arm_lease_client == client_id:
                self._arm_lease_client = None

    def _arm_prepare(self, request):
        client_id = self._acquire_arm_lease(request, allow_owned=True)
        prepare = getattr(self.arm, "prepare_operation", None)
        try:
            if prepare is not None:
                prepare()
        except Exception:
            self._release_arm_lease(client_id)
            raise

    def _arm_prepare_motion(self, request):
        client_id = self._acquire_arm_lease(request)
        try:
            self.arm.prepare_motion()
        except Exception:
            self._release_arm_lease(client_id)
            raise

    def _arm_execute(self, request):
        self._require_arm_lease(request)
        action_id = int(request["action_id"])
        if action_id not in self.allowed_arm_actions:
            raise ValueError(f"arm action {action_id} is not allowed")
        self.arm_ownership.expect(action_id)
        try:
            return {"code": self.arm.execute(action_id)}
        except Exception:
            self.arm_ownership.clear_expectation(action_id)
            raise

    def _arm_motion(self, request):
        self._require_arm_lease(request)
        return self.arm.execute_motion(
            request["motion"], float(request["execution_timeout_s"]))

    def _arm_motion_release(self, request):
        client_id = self._require_arm_lease(request)
        try:
            return self.arm.release_motion()
        finally:
            self._release_arm_lease(client_id)

    def _arm_interrupt(self, request):
        self._require_arm_lease(request)
        self.arm.interrupt()

    def _arm_release(self, request):
        client_id = self._require_arm_lease(request)
        self.arm_ownership.expect(UnitreeArm.RELEASE_ACTION)
        try:
            return {"code": self.arm.release()}
        except Exception:
            self.arm_ownership.clear_expectation(UnitreeArm.RELEASE_ACTION)
            raise
        finally:
            self._release_arm_lease(client_id)

    def _hand_prepare(self, request):
        del request
        prepare = getattr(self.hand, "prepare_operation", None)
        if prepare is not None:
            prepare()

    def _hand_command(self, request):
        hand = self._request_hand(request)
        target = validate_hand_target(request["target"])
        ramp = validate_hand_ramp_seconds(request["ramp_s"], "ramp_s")
        self._owned_hands.add(hand)
        result = self.hand.command_position(hand, target, ramp)
        return result

    def _hand_interrupt(self, request):
        del request
        self.hand.interrupt()

    def _hand_neutral(self, request):
        hand = self._request_hand(request)
        ramp = validate_hand_ramp_seconds(request["ramp_s"], "ramp_s")
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
        ownership = getattr(self, "arm_ownership", None)
        if self.arm.has_motion():
            commands.append(("arm motion", self.arm.release_motion))
        elif ownership is None or ownership.snapshot()["status"] in (
                ArmOwnership.IDLE, ArmOwnership.OWNED):
            def release_arm():
                code = self.arm.release()
                if code != 0:
                    raise RuntimeError(f"release action returned code {code}")
            commands.append(("arm", release_arm))
        else:
            self.get_logger().warning(
                "safe shutdown yielded arm release to external or "
                "unavailable Unitree control")
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
        self._release_arm_lease()


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
        node.stop_arm_monitor()

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
