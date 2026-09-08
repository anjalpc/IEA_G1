"""ROS node for independent card-hand and card-arm control."""

import json
import signal
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:  # Foxy
    SignalHandlerOptions = None

from g1_card_presentation.card_backends import CardHand
from g1_card_presentation.card_sequence import CardPresentation
from g1_robot_interfaces.bridge_client import (
    ArmClient,
    BridgeTransport,
    HandClient,
)
from g1_robot_interfaces.validation import (
    validate_hand_ramp_seconds,
    validate_hand_target,
)


ACTIVE_ARM_STATES = {
    "PREPARING",
    "MOVING",
    "PRESENTED",
}


class CardPresentationNode(Node):
    """Expose independent card hand and arm operations."""

    def __init__(self):
        super().__init__("event_card_presentation")

        self.declare_parameter("bridge_name", "simulation")
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("pose", "left_card_presentation")
        self.declare_parameter("motion_fraction", 1.0)
        self.declare_parameter("move_seconds", 1.2)

        self.declare_parameter(
            "presentation_timeout_s",
            60.0,
        )

        self.declare_parameter(
            "released_timeout_s",
            20.0,
        )

        self.declare_parameter(
            "release_dwell_s",
            0.5,
        )

        self.declare_parameter(
            "grip_target",
            [0.78, 1.0, 0.4, 1.0, 1.0, 1.0],
        )

        self.declare_parameter(
            "grip_ramp_s",
            1.5,
        )

        self.declare_parameter(
            "release_ramp_s",
            1.5,
        )

        self.bridge_name = str(
            self.get_parameter("bridge_name").value
        )

        timeout = float(
            self.get_parameter(
                "action_timeout_s"
            ).value
        )

        pose = str(
            self.get_parameter("pose").value
        )

        fraction = float(
            self.get_parameter(
                "motion_fraction"
            ).value
        )

        move_seconds = float(
            self.get_parameter(
                "move_seconds"
            ).value
        )

        self.presentation_timeout = float(
            self.get_parameter(
                "presentation_timeout_s"
            ).value
        )

        self.released_timeout = float(
            self.get_parameter(
                "released_timeout_s"
            ).value
        )

        release_dwell = float(
            self.get_parameter(
                "release_dwell_s"
            ).value
        )

        if timeout <= 0.0:
            raise ValueError(
                "action_timeout_s must be positive"
            )

        if pose != "left_card_presentation":
            raise ValueError(
                "unsupported card presentation pose"
            )

        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "motion_fraction must be between 0 and 1"
            )

        if not 1.2 <= move_seconds <= 10.0:
            raise ValueError(
                "move_seconds must be between 1.2 and 10"
            )

        if not 5.0 <= self.presentation_timeout <= 300.0:
            raise ValueError(
                "presentation_timeout_s must be "
                "between 5 and 300"
            )

        if not 5.0 <= self.released_timeout <= 60.0:
            raise ValueError(
                "released_timeout_s must be "
                "between 5 and 60"
            )

        if not 0.0 <= release_dwell <= 3.0:
            raise ValueError(
                "release_dwell_s must be between 0 and 3"
            )

        transport = BridgeTransport(
            self,
            timeout,
            self.bridge_name,
        )

        arm = ArmClient(transport)

        hand = CardHand(
            HandClient(transport),
            validate_hand_target(
                self.get_parameter(
                    "grip_target"
                ).value
            ),
            validate_hand_ramp_seconds(
                self.get_parameter(
                    "grip_ramp_s"
                ).value
            ),
            validate_hand_ramp_seconds(
                self.get_parameter(
                    "release_ramp_s"
                ).value
            ),
        )

        # Independent stable subsystem states.
        self.hand_state = "OPEN"
        self.arm_state = "DOWN"

        # Last physical transition/event.
        self.state = "IDLE"

        self.arm_ownership = (
            "IDLE"
            if self.bridge_name != "hardware"
            else "UNAVAILABLE"
        )

        self.arm_action_id = None
        self.arm_action_name = ""

        self.operation_lock = threading.Lock()

        # Normal commands remain serialized deliberately.
        self.operation_worker = None

        self.cleanup_worker = None

        self.presentation_timer = None
        self.shutting_down = threading.Event()

        self.sequence = CardPresentation(
            arm,
            hand,
            pose,
            fraction,
            move_seconds,
            set_state=self.set_state,
            release_dwell_s=release_dwell,
        )

        self.publisher = self.create_publisher(
            String,
            "/event/card/state",
            10,
        )

        self.create_subscription(
            String,
            "/g1/bridge/hardware/arm_ownership",
            self.arm_ownership_callback,
            10,
        )

        self.create_service(
            Trigger,
            "/event/card/grip",
            self.grip_callback,
        )

        self.create_service(
            Trigger,
            "/event/card/release",
            self.release_callback,
        )

        self.create_service(
            Trigger,
            "/event/card/present",
            self.present_callback,
        )

        self.create_service(
            Trigger,
            "/event/card/retract",
            self.retract_callback,
        )

        self.create_service(
            Trigger,
            "/event/card/cancel",
            self.cancel_callback,
        )

        self.state_timer = self.create_timer(
            0.5,
            self.publish_state,
        )

        self.get_logger().warning(
            "Card application ready for bridge "
            f"'{self.bridge_name}'"
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def publish_state(self):
        """Publish independent hand and arm state."""

        if self.shutting_down.is_set():
            return

        message = String()

        message.data = json.dumps(
            {
                "state": self.state,
                "hand_state": self.hand_state,
                "arm_state": self.arm_state,
                "bridge": self.bridge_name,
                "arm_ownership": self.arm_ownership,
                "arm_action_id": self.arm_action_id,
                "arm_action_name": self.arm_action_name,
            },
            sort_keys=True,
        )

        self.publisher.publish(message)

    def set_state(self, state):
        """Update event state and relevant subsystem state."""

        state = str(state)
        self.state = state

        # Hand transitions.
        if state == "GRIPPING_CARD":
            self.hand_state = "MOVING"

        elif state == "CARD_GRIPPED":
            self.hand_state = "GRIPPED"

        elif state == "RELEASING_CARD":
            self.hand_state = "MOVING"

        elif state == "CARD_RELEASED":
            self.hand_state = "OPEN"

        # Arm transitions.
        elif state == "PREPARING":
            self.arm_state = "PREPARING"

        elif state == "PRESENTING":
            self.arm_state = "MOVING"

        elif state == "CARD_PRESENTED":
            self.arm_state = "PRESENTED"

        elif state == "RETRACTING":
            self.arm_state = "MOVING"

        elif state == "ARM_DOWN":
            self.arm_state = "DOWN"

        elif state == "CANCELLED":
            self.hand_state = "OPEN"
            self.arm_state = "DOWN"

        self.publish_state()

    # ------------------------------------------------------------------
    # Arm ownership
    # ------------------------------------------------------------------

    def arm_ownership_callback(self, message):
        """Track Unitree native arm ownership."""

        if self.bridge_name != "hardware":
            return

        try:
            state = json.loads(message.data)

            status = str(state["status"])
            action_id = state.get("id")
            action_name = str(
                state.get("name", "")
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self.get_logger().error(
                f"invalid arm ownership state: {exc}"
            )

            status = "UNAVAILABLE"
            action_id = None
            action_name = ""

        self.arm_ownership = status
        self.arm_action_id = action_id
        self.arm_action_name = action_name

        # Hand-only activity does not care about native arm ownership.
        # Cleanup only when our arm is actually active.
        if (
            status in ("EXTERNAL", "UNAVAILABLE")
            and self.arm_state in ACTIVE_ARM_STATES
        ):
            self.get_logger().warning(
                "arm ownership changed while card arm "
                "was active; requesting safe cancellation"
            )

            self._start_cleanup()

    def arm_start_block_reason(self):
        """Return why a new physical arm motion cannot start."""

        if self.bridge_name != "hardware":
            return None

        if self.arm_ownership == "IDLE":
            return None

        if self.arm_ownership == "EXTERNAL":
            return (
                "external Unitree arm action is active: "
                f"{self.arm_action_id} "
                f"{self.arm_action_name}"
            ).rstrip()

        return (
            "Unitree arm state is unavailable "
            "or not stably idle"
        )

    # ------------------------------------------------------------------
    # Hand
    # ------------------------------------------------------------------

    def grip_callback(self, request, response):
        """Close the hand regardless of arm position."""

        del request

        if self.hand_state != "OPEN":
            response.success = False
            response.message = (
                f"cannot grip while hand state is "
                f"{self.hand_state}"
            )
            return response

        if not self._start_operation(
            self.sequence.grip,
            "grip",
        ):
            response.success = False
            response.message = (
                "another card operation is running"
            )
            return response

        response.success = True
        response.message = "card grip accepted"
        return response

    def release_callback(self, request, response):
        """Open the hand regardless of arm position."""

        del request

        if self.hand_state != "GRIPPED":
            response.success = False
            response.message = (
                f"cannot release while hand state is "
                f"{self.hand_state}"
            )
            return response

        if not self._start_operation(
            self.sequence.release_card,
            "release",
        ):
            response.success = False
            response.message = (
                "another card operation is running"
            )
            return response

        response.success = True
        response.message = "card release accepted"
        return response

    # ------------------------------------------------------------------
    # Arm
    # ------------------------------------------------------------------

    def present_callback(self, request, response):
        """Present the arm regardless of hand state."""

        del request

        if self.arm_state != "DOWN":
            response.success = False
            response.message = (
                f"cannot present while arm state is "
                f"{self.arm_state}"
            )
            return response

        blocked = self.arm_start_block_reason()

        if blocked is not None:
            response.success = False
            response.message = blocked
            return response

        self.cancel_presentation_timer()

        if not self._start_operation(
            self.sequence.present,
            "present",
        ):
            response.success = False
            response.message = (
                "another card operation is running"
            )
            return response

        response.success = True
        response.message = "card presentation accepted"
        return response

    def retract_callback(self, request, response):
        """Retract the arm regardless of hand state."""

        del request

        if self.arm_state != "PRESENTED":
            response.success = False
            response.message = (
                f"cannot retract while arm state is "
                f"{self.arm_state}"
            )
            return response

        self.cancel_presentation_timer()

        if not self._start_operation(
            self.sequence.retract,
            "retract",
        ):
            response.success = False
            response.message = (
                "another card operation is running"
            )
            return response

        response.success = True
        response.message = "card retraction accepted"
        return response

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_callback(self, request, response):
        """Safely release all currently-active card subsystems."""

        del request

        if (
            self.hand_state == "OPEN"
            and self.arm_state == "DOWN"
        ):
            response.success = True
            response.message = (
                "card system is already safe"
            )
            return response

        if not self._start_cleanup():
            response.success = True
            response.message = (
                "card cleanup is already running"
            )
            return response

        response.success = True
        response.message = (
            "card cancellation accepted"
        )
        return response

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _start_operation(self, command, name):
        """Start one serialized normal operation."""

        with self.operation_lock:
            if self._worker_running(
                self.operation_worker
            ):
                return False

            if self._worker_running(
                self.cleanup_worker
            ):
                return False

            self.operation_worker = threading.Thread(
                target=self._run_operation,
                args=(command, name),
                name=f"card-{name}-worker",
                daemon=True,
            )

            self.operation_worker.start()

            return True

    def _run_operation(self, command, name):
        """Run one normal card primitive."""

        try:
            result = command()

            self.get_logger().info(
                f"card {name} complete: {result}"
            )

            if (
                name == "present"
                and self.arm_state == "PRESENTED"
                and not self.sequence.stop_requested.is_set()
            ):
                self.schedule_presentation_timeout()

            # If the hand is released while the arm is presented,
            # replace the long presentation timeout with 20 seconds.
            if (
                name == "release"
                and self.arm_state == "PRESENTED"
                and not self.sequence.stop_requested.is_set()
            ):
                self.schedule_released_timeout()

        except Exception as exc:
            if self.sequence.stop_requested.is_set():
                self.get_logger().warning(
                    f"card {name} interrupted "
                    "during cleanup"
                )
                return

            self.get_logger().error(
                f"card {name} failed: {exc}"
            )

            cleanup_succeeded = True

            try:
                self.sequence.ensure_safe()
            except Exception as cleanup_exc:
                cleanup_succeeded = False
                self.get_logger().error(
                    "card safe release failed: "
                    f"{cleanup_exc}"
                )

            if cleanup_succeeded:
                self.hand_state = "OPEN"
                self.arm_state = "DOWN"
            else:
                self.hand_state = "UNKNOWN"
                self.arm_state = "UNKNOWN"

            self.set_state("ERROR")

    # ------------------------------------------------------------------
    # Safety cleanup
    # ------------------------------------------------------------------

    def _start_cleanup(self):
        """Start coordinated cleanup."""

        with self.operation_lock:
            if self._worker_running(
                self.cleanup_worker
            ):
                return False

            self.cancel_presentation_timer()

            operation_worker = self.operation_worker
            self.sequence.interrupt_active()

            self.cleanup_worker = threading.Thread(
                target=self._run_cleanup,
                args=(
                    self.sequence.cancel,
                    operation_worker,
                ),
                name="card-cleanup-worker",
                daemon=True,
            )

            self.cleanup_worker.start()

            return True

    def _run_cleanup(
        self,
        command,
        operation_worker,
    ):
        try:
            if (
                self._worker_running(operation_worker)
                and operation_worker
                is not threading.current_thread()
            ):
                operation_worker.join()

            message = command()
            self.get_logger().info(message)

        except Exception as exc:
            self.get_logger().error(
                f"card cleanup failed: {exc}"
            )
            self.hand_state = "UNKNOWN"
            self.arm_state = "UNKNOWN"
            self.set_state("ERROR")

    # ------------------------------------------------------------------
    # Timeouts
    # ------------------------------------------------------------------

    def schedule_presentation_timeout(self):
        """Start long timeout while arm is presented."""

        self.cancel_presentation_timer()

        timer = threading.Timer(
            self.presentation_timeout,
            self.presentation_timeout_callback,
        )

        timer.daemon = True
        self.presentation_timer = timer
        timer.start()

    def schedule_released_timeout(self):
        """Start short timeout after releasing at presented pose."""

        self.cancel_presentation_timer()

        timer = threading.Timer(
            self.released_timeout,
            self.presentation_timeout_callback,
        )

        timer.daemon = True
        self.presentation_timer = timer
        timer.start()

    def cancel_presentation_timer(self):
        timer = self.presentation_timer
        self.presentation_timer = None

        if timer is not None:
            timer.cancel()

    def presentation_timeout_callback(self):
        """Retract safely when arm presentation expires."""

        self.presentation_timer = None

        if self.arm_state != "PRESENTED":
            return

        self.get_logger().warning(
            "card arm presentation timed out; "
            "starting safe cleanup"
        )

        self._start_cleanup()

    # ------------------------------------------------------------------
    # Utilities / shutdown
    # ------------------------------------------------------------------

    @staticmethod
    def _worker_running(worker):
        return (
            worker is not None
            and worker.is_alive()
        )

    def shutdown(self):
        self.shutting_down.set()

        self.state_timer.cancel()
        self.cancel_presentation_timer()

        self.sequence.interrupt_active()

        for worker in (
            self.cleanup_worker,
            self.operation_worker,
        ):
            if (
                self._worker_running(worker)
                and worker is not threading.current_thread()
            ):
                worker.join(timeout=1.0)


def main(args=None):
    """Run with explicit signals and a multithreaded executor."""

    if SignalHandlerOptions is None:
        rclpy.init(args=args)
    else:
        rclpy.init(
            args=args,
            signal_handler_options=(
                SignalHandlerOptions.NO
            ),
        )

    node = CardPresentationNode()

    executor = MultiThreadedExecutor(
        num_threads=3
    )

    executor.add_node(node)

    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    previous_sigint = signal.signal(
        signal.SIGINT,
        request_stop,
    )

    previous_sigterm = signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    try:
        while (
            not stop_requested.is_set()
            and rclpy.ok()
        ):
            executor.spin_once(
                timeout_sec=0.2
            )

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.shutdown()

        executor.shutdown(
            timeout_sec=3.0
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        signal.signal(
            signal.SIGINT,
            previous_sigint,
        )

        signal.signal(
            signal.SIGTERM,
            previous_sigterm,
        )


if __name__ == "__main__":
    main()
