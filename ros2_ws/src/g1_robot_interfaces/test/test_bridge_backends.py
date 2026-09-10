import json
import threading
from types import SimpleNamespace

import pytest

from g1_robot_interfaces.bridge_client import (
    ArmClient, HandClient, BridgeTransport, bridge_topics)
from g1_robot_interfaces.validation import (
    validate_hand_name,
    validate_hand_ramp_seconds,
    validate_hand_target,
)


class TransportRecorder:
    def __init__(self):
        self.timeout = 5.0
        self.commands = []

    def send(self, operation, **payload):
        self.commands.append((operation, payload))
        if operation in ("arm_execute", "arm_release"):
            return {"code": 0}
        return None


def test_arm_uses_only_backend_neutral_operations():
    transport = TransportRecorder()
    backend = ArmClient(transport)
    backend.prepare_operation()
    assert backend.execute(27) == 0
    assert backend.release() == 0
    assert [command[0] for command in transport.commands] == [
        "arm_prepare", "arm_execute", "arm_interrupt", "arm_release"]


def test_hand_uses_only_backend_neutral_operations():
    transport = TransportRecorder()
    backend = HandClient(transport)
    backend.prepare_operation()
    backend.command_position("right", [0.2] * 6, 0.2)
    backend.neutral("right", 0.2)
    assert [command[0] for command in transport.commands] == [
        "hand_prepare", "hand_command",
        "hand_interrupt", "hand_neutral"]
    assert transport.commands[1] == (
        "hand_command",
        {
            "response_timeout": 12.2,
            "hand": "right",
            "target": [0.2] * 6,
            "ramp_s": 0.2,
        },
    )
    assert transport.commands[3] == (
        "hand_neutral",
        {
            "response_timeout": 12.2,
            "hand": "right",
            "ramp_s": 0.2,
        },
    )


def test_restricted_motion_keeps_legacy_watchdog_deadlines():
    from g1_robot_interfaces.arm_motion import ARM_JOINT_NAMES
    motion = {
        'version': 1, 'preparation': dict.fromkeys(ARM_JOINT_NAMES, 0.0),
        'target_offsets': dict.fromkeys(ARM_JOINT_NAMES[:7], 0.0),
        'phase_seconds': [0.3, 0.9], 'recovery': ['preparation', 'baseline'],
        'fraction': 0.25,
    }
    transport = TransportRecorder()
    arm = ArmClient(transport)
    arm.prepare_motion()
    arm.execute_motion(motion, 5.0)
    arm.release_motion()
    assert transport.commands[1] == ('arm_motion', {
        'motion': motion, 'execution_timeout_s': 5.0,
        'response_timeout': 32.0})
    assert transport.commands[2] == (
        'arm_motion_release', {'response_timeout': 17.0})


@pytest.mark.parametrize(
    'execution_timeout_s',
    [0.0, -1.0, float('nan'), float('inf'), float('-inf')],
)
def test_restricted_motion_rejects_invalid_compatibility_timeout(
        execution_timeout_s):
    """Reject invalid watchdog arithmetic before publishing a request."""
    from g1_robot_interfaces.arm_motion import ARM_JOINT_NAMES
    motion = {
        'version': 1,
        'preparation': dict.fromkeys(ARM_JOINT_NAMES, 0.0),
        'target_offsets': dict.fromkeys(ARM_JOINT_NAMES[:7], 0.0),
        'phase_seconds': [0.3, 0.9],
        'recovery': ['preparation', 'baseline'],
        'fraction': 0.25,
    }
    transport = TransportRecorder()
    arm = ArmClient(transport)
    arm.prepare_motion()
    with pytest.raises(ValueError, match='finite and positive'):
        arm.execute_motion(motion, execution_timeout_s)
    assert transport.commands == [('arm_motion_prepare', {})]


def test_arm_preserves_nonzero_action_result():
    transport = TransportRecorder()
    backend = ArmClient(transport)
    backend.prepare_operation()
    original_send = transport.send

    def send(operation, **payload):
        if operation == "arm_execute":
            return {"code": 7402}
        return original_send(operation, **payload)

    transport.send = send
    assert backend.execute(27) == 7402


def test_arm_release_rejects_nonzero_result():
    transport = TransportRecorder()
    backend = ArmClient(transport)
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


@pytest.mark.parametrize("hand", ["left", "right"])
def test_validates_supported_hands(hand):
    assert validate_hand_name(hand) == hand


@pytest.mark.parametrize("hand", ["", "both", None])
def test_rejects_unsupported_hands(hand):
    with pytest.raises(ValueError):
        validate_hand_name(hand)


def test_validates_hand_target_and_ramp_limits():
    assert validate_hand_target([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) == [
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert validate_hand_ramp_seconds(0.2) == 0.2
    with pytest.raises(ValueError):
        validate_hand_target([0.2] * 5)
    with pytest.raises(ValueError):
        validate_hand_target([0.2] * 5 + [1.1])
    with pytest.raises(ValueError):
        validate_hand_ramp_seconds(0.19)


def test_rejects_nonfinite_hand_values():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            validate_hand_target([0.2] * 5 + [value])
        with pytest.raises(ValueError):
            validate_hand_ramp_seconds(value)
