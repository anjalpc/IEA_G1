import json
import threading
from types import SimpleNamespace

import pytest

from g1_robot_interfaces.bridge_client import (
    BridgeArmBackend, BridgeHandBackend, BridgeTransport, bridge_topics)


class TransportRecorder:
    def __init__(self):
        self.timeout = 5.0
        self.commands = []

    def send(self, operation, **payload):
        self.commands.append((operation, payload))
        if operation in ("arm_action", "arm_release"):
            return {"code": 0}
        return None


def test_arm_uses_only_backend_neutral_operations():
    transport = TransportRecorder()
    backend = BridgeArmBackend(transport)
    backend.prepare_operation()
    assert backend.execute(27) == 0
    assert backend.release() == 0
    assert [command[0] for command in transport.commands] == [
        "arm_prepare", "arm_action", "arm_interrupt", "arm_release"]


def test_hand_uses_only_backend_neutral_operations():
    transport = TransportRecorder()
    backend = BridgeHandBackend(transport)
    backend.prepare_operation()
    backend.grip()
    backend.release()
    assert [command[0] for command in transport.commands] == [
        "hand_prepare", "hand_grip",
        "hand_interrupt", "hand_release"]


def test_arm_preserves_nonzero_action_result():
    transport = TransportRecorder()
    backend = BridgeArmBackend(transport)
    backend.prepare_operation()
    original_send = transport.send

    def send(operation, **payload):
        if operation == "arm_action":
            return {"code": 7402}
        return original_send(operation, **payload)

    transport.send = send
    assert backend.execute(27) == 7402


def test_arm_release_rejects_nonzero_result():
    transport = TransportRecorder()
    backend = BridgeArmBackend(transport)
    backend.prepare_operation()
    original_send = transport.send

    def send(operation, **payload):
        if operation == "arm_release":
            return {"code": 7403}
        return original_send(operation, **payload)

    transport.send = send
    with pytest.raises(RuntimeError, match="arm release returned 7403"):
        backend.release()


def test_bridge_topics_are_isolated_and_require_configuration():
    assert bridge_topics("hardware") == (
        "/g1/bridge/hardware/command", "/g1/bridge/hardware/feedback")
    assert bridge_topics("simulation") != bridge_topics("hardware")
    with pytest.raises(ValueError):
        bridge_topics("unconfigured")


def test_late_feedback_for_expired_command_is_discarded():
    transport = BridgeTransport.__new__(BridgeTransport)
    transport.client_id = "client"
    transport.condition = threading.Condition()
    transport.responses = {}
    transport.pending = set()
    message = SimpleNamespace(data=json.dumps({
        "client_id": "client", "command_id": 7, "status": "ok"}))
    transport._feedback(message)
    assert transport.responses == {}
