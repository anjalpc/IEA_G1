"""Reusable acknowledged ROS client for the G1 MuJoCo bridge."""

import json
import threading
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

from g1_robot_bridge.brainco_hand_helper import plan_curl


SIM_COMMAND_TOPIC = "/simulation/g1/command"
SIM_FEEDBACK_TOPIC = "/simulation/g1/feedback"


class SimulationTransport:
    """Send acknowledged JSON commands to the generic simulation bridge."""

    def __init__(self, node, timeout):
        self.timeout = timeout
        self.publisher = node.create_publisher(String, SIM_COMMAND_TOPIC, 10)
        self.condition = threading.Condition()
        self.responses = {}
        self.next_id = 1
        callback_group = ReentrantCallbackGroup()
        node.create_subscription(
            String, SIM_FEEDBACK_TOPIC, self._feedback, 10,
            callback_group=callback_group)

    def _feedback(self, message):
        try:
            response = json.loads(message.data)
            command_id = int(response["command_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self.condition:
            self.responses[command_id] = response
            self.condition.notify_all()

    def send(self, interface, **payload):
        with self.condition:
            command_id = self.next_id
            self.next_id += 1
        message = String()
        message.data = json.dumps({
            "command_id": command_id,
            "interface": interface,
            **payload,
        }, sort_keys=True)
        self.publisher.publish(message)
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while command_id not in self.responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"MuJoCo bridge did not acknowledge {interface}")
                self.condition.wait(remaining)
            response = self.responses.pop(command_id)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"MuJoCo bridge rejected {interface}: "
                f"{response.get('error', 'unknown error')}")


class SimulationArmActionBackend:
    """Unitree-style arm action adapter using the generic bridge protocol."""

    def __init__(self, transport):
        self.transport = transport

    def execute(self, action_id):
        self.transport.send("arm_action", action_id=int(action_id))
        if action_id == 27:
            time.sleep(1.0)
        return 0


class SimulationBrainCoBackend:
    """BrainCo right-hand adapter using the generic bridge protocol."""

    def __init__(self, transport, grip_amount):
        self.transport = transport
        self.grip_target = plan_curl(grip_amount)

    def grip(self):
        self.transport.send(
            "hand_position", hand="right", positions=self.grip_target)
        time.sleep(1.5)

    def release(self):
        self.transport.send(
            "hand_position", hand="right", positions=[0.0] * 6)
        time.sleep(1.5)
