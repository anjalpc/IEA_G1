"""Acknowledged backend-neutral command client for G1 applications."""

import json
import threading
import time
import uuid

from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String


RESPONSE_MARGIN_S = 2.0


def bridge_topics(bridge_name):
    """Return isolated command and feedback topics for one named bridge."""
    name = str(bridge_name).strip()
    if not name or name == "unconfigured" or "/" in name:
        raise ValueError("bridge_name must be a configured ROS name component")
    return f"/g1/bridge/{name}/command", f"/g1/bridge/{name}/feedback"


class BridgeTransport:
    """Send correlated commands to whichever G1 bridge is running."""

    def __init__(self, node, timeout, bridge_name):
        if timeout <= 0.0:
            raise ValueError("bridge timeout must be positive")
        self.timeout = timeout
        self.bridge_name = str(bridge_name)
        self.client_id = uuid.uuid4().hex
        command_topic, feedback_topic = bridge_topics(self.bridge_name)
        self.publisher = node.create_publisher(String, command_topic, 10)
        self.condition = threading.Condition()
        self.responses = {}
        self.pending = set()
        self.next_id = 1
        callback_group = ReentrantCallbackGroup()
        node.create_subscription(
            String, feedback_topic, self._feedback, 10,
            callback_group=callback_group)

    def _feedback(self, message):
        try:
            response = json.loads(message.data)
            if response["client_id"] != self.client_id:
                return
            command_id = int(response["command_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self.condition:
            if command_id not in self.pending:
                return
            self.responses[command_id] = response
            self.condition.notify_all()

    def send(self, operation, response_timeout=None, **payload):
        discovery_deadline = time.monotonic() + self.timeout
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() >= discovery_deadline:
                raise TimeoutError("G1 bridge is not available")
            time.sleep(0.02)
        with self.condition:
            command_id = self.next_id
            self.next_id += 1
            self.pending.add(command_id)
        message = String()
        message.data = json.dumps({
            "client_id": self.client_id,
            "command_id": command_id,
            "operation": operation,
            **payload,
        }, sort_keys=True)
        try:
            self.publisher.publish(message)
            timeout = (self.timeout if response_timeout is None
                       else response_timeout)
            deadline = time.monotonic() + timeout
            with self.condition:
                while command_id not in self.responses:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(
                            f"G1 bridge did not acknowledge {operation}")
                    self.condition.wait(remaining)
                response = self.responses.pop(command_id)
        finally:
            with self.condition:
                self.pending.discard(command_id)
                self.responses.pop(command_id, None)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"G1 bridge rejected {operation}: "
                f"{response.get('error', 'unknown error')}")
        return response.get("result")


class ArmClient:
    """Generic acknowledged arm-command client."""

    def __init__(self, transport):
        self.transport = transport
        self.cancelled = True

    def prepare_operation(self):
        self.transport.send("arm_prepare")
        self.cancelled = False

    def prepare_presentation(self):
        """Require backend approval before a card workflow can move its hand."""
        self.transport.send("arm_prepare_presentation")
        self.cancelled = False

    def interrupt(self):
        self.cancelled = True
        self.transport.send("arm_interrupt")

    def execute(self, action_id):
        if self.cancelled:
            raise RuntimeError("arm action cancelled")
        result = self.transport.send(
            "arm_execute", response_timeout=(self.transport.timeout + 3.0
                                             + RESPONSE_MARGIN_S),
            action_id=int(action_id))
        try:
            return int(result["code"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G1 bridge returned an invalid arm result") from exc

    def present(self, pose, fraction, move_seconds):
        """Request a backend-owned named presentation pose."""
        if self.cancelled:
            raise RuntimeError("arm action cancelled")
        return self.transport.send(
            "arm_present",
            response_timeout=(self.transport.timeout + float(move_seconds)
                              + RESPONSE_MARGIN_S),
            pose=str(pose), fraction=float(fraction),
            move_s=float(move_seconds))

    def release_presentation(self):
        """Release a named presentation without invoking a firmware action."""
        self.cancelled = True
        return self.transport.send("arm_present_release")

    def release(self):
        self.interrupt()
        result = self.transport.send(
            "arm_release", response_timeout=(self.transport.timeout + 3.0
                                              + RESPONSE_MARGIN_S))
        try:
            code = int(result["code"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G1 bridge returned an invalid release result") from exc
        if code != 0:
            raise RuntimeError(f"arm release returned {code}")
        return code


class HandClient:
    """Generic acknowledged hand-position client."""

    def __init__(self, transport):
        self.transport = transport
        self.cancelled = True

    def prepare_operation(self):
        self.transport.send("hand_prepare")
        self.cancelled = False

    def interrupt(self):
        self.cancelled = True
        self.transport.send("hand_interrupt")

    def command_position(self, hand, target, ramp_seconds):
        if self.cancelled:
            raise RuntimeError("hand command cancelled")
        return self.transport.send(
            "hand_command",
            response_timeout=(self.transport.timeout + float(ramp_seconds)
                              + 5.0 + RESPONSE_MARGIN_S),
            hand=str(hand), target=list(target),
            ramp_s=float(ramp_seconds))

    def neutral(self, hand, ramp_seconds):
        self.interrupt()
        return self.transport.send(
            "hand_neutral",
            response_timeout=(self.transport.timeout + float(ramp_seconds)
                              + 5.0 + RESPONSE_MARGIN_S),
            hand=str(hand), ramp_s=float(ramp_seconds))
