"""ROS node for the separate left-hand card-presentation behavior."""

import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from g1_card_presentation.card_backends import CardHand
from g1_card_presentation.card_sequence import CardPresentation
from g1_robot_interfaces.bridge_client import ArmClient, BridgeTransport, HandClient
from g1_robot_interfaces.validation import (
    validate_hand_ramp_seconds,
    validate_hand_target,
)


class CardPresentationNode(Node):
    """Expose left-card presentation without modifying the handshake app."""

    def __init__(self):
        super().__init__("event_card_presentation")
        self.declare_parameter("bridge_name", "simulation")
        self.declare_parameter("action_timeout_s", 5.0)
        self.declare_parameter("pose", "left_card_presentation")
        self.declare_parameter("motion_fraction", 1.0)
        self.declare_parameter("move_seconds", 5.0)
        self.declare_parameter(
            "grip_target", [0.78, 1.0, 0.4, 1.0, 1.0, 1.0])
        self.declare_parameter("grip_ramp_s", 1.5)
        self.declare_parameter("release_ramp_s", 1.5)

        timeout = float(self.get_parameter("action_timeout_s").value)
        fraction = float(self.get_parameter("motion_fraction").value)
        move_seconds = float(self.get_parameter("move_seconds").value)
        pose = str(self.get_parameter("pose").value)
        if timeout <= 0.0:
            raise ValueError("action_timeout_s must be positive")
        if pose != "left_card_presentation":
            raise ValueError("unsupported card presentation pose")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("motion_fraction must be between 0 and 1")
        if not 2.0 <= move_seconds <= 10.0:
            raise ValueError("move_seconds must be between 2 and 10")

        transport = BridgeTransport(
            self, timeout, str(self.get_parameter("bridge_name").value))
        hand = CardHand(
            HandClient(transport),
            validate_hand_target(self.get_parameter("grip_target").value),
            validate_hand_ramp_seconds(
                self.get_parameter("grip_ramp_s").value),
            validate_hand_ramp_seconds(
                self.get_parameter("release_ramp_s").value),
        )
        self.sequence = CardPresentation(
            ArmClient(transport), hand, pose, fraction, move_seconds)
        self.lock = threading.Lock()
        self.create_service(
            Trigger, "/event/card/present", self._present)
        self.create_service(
            Trigger, "/event/card/release", self._release)

    def _present(self, request, response):
        del request
        with self.lock:
            try:
                self.sequence.prepare()
                result = self.sequence.present()
                response.success = True
                response.message = f"left-card presentation complete: {result}"
            except Exception as exc:
                response.success = False
                response.message = str(exc)
        return response

    def _release(self, request, response):
        del request
        with self.lock:
            try:
                self.sequence.release()
                response.success = True
                response.message = "left hand opened and arm released"
            except Exception as exc:
                response.success = False
                response.message = str(exc)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CardPresentationNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.sequence.release()
        except Exception as exc:
            node.get_logger().error(f"card safe release failed: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
