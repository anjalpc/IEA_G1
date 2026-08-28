"""Shared validated request/reply plumbing for G1 bridge nodes."""

import json

from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

from g1_robot_interfaces.bridge_client import bridge_topics


class BridgeServer:
    """Dispatch correlated commands to an explicit operation allowlist."""

    def __init__(self, node, bridge_name, handlers):
        self.node = node
        self.handlers = handlers
        command_topic, feedback_topic = bridge_topics(bridge_name)
        self.feedback = node.create_publisher(String, feedback_topic, 10)
        node.create_subscription(
            String, command_topic, self.command_callback, 10,
            callback_group=ReentrantCallbackGroup())

    def _reply(self, client_id, command_id, status, result=None, error=None):
        message = String()
        response = {
            "client_id": client_id,
            "command_id": command_id,
            "status": status,
        }
        if result is not None:
            response["result"] = result
        if error is not None:
            response["error"] = error
        message.data = json.dumps(response, sort_keys=True)
        self.feedback.publish(message)

    def command_callback(self, message):
        client_id = None
        command_id = None
        try:
            request = json.loads(message.data)
            client_id = str(request["client_id"])
            command_id = int(request["command_id"])
            operation = str(request["operation"])
            handler = self.handlers.get(operation)
            if handler is None:
                raise ValueError(f"unsupported G1 operation: {operation}")
            result = handler(request)
        except Exception as exc:
            self._reply(client_id, command_id, "error", error=str(exc))
            return
        self._reply(client_id, command_id, "ok", result=result)
