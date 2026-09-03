"""Stream normalized Unitree arm-action state from native DDS domain 0."""

import json
import signal
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


OUTPUT_PREFIX = "ARM_STATE="


def normalize_state(payload):
    """Validate one `/arm/action/state` JSON payload."""
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("arm action state must be a JSON object")
    return {
        "id": int(decoded["id"]),
        "name": str(decoded.get("name", "")),
        "holding": bool(decoded.get("holding", False)),
    }


class ArmStateReader(Node):
    """Print valid native arm state samples for the parent bridge process."""

    def __init__(self):
        super().__init__("g1_arm_state_reader")
        self.create_subscription(
            String, "/arm/action/state", self._state_callback, 10)

    def _state_callback(self, message):
        try:
            state = normalize_state(message.data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"ARM_STATE_ERROR={exc}", file=sys.stderr, flush=True)
            return
        print(
            OUTPUT_PREFIX + json.dumps(state, sort_keys=True),
            flush=True,
        )


def main(args=None):
    """Run until the parent supervisor requests termination."""
    rclpy.init(args=args)
    node = ArmStateReader()
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        while rclpy.ok() and not stop_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
