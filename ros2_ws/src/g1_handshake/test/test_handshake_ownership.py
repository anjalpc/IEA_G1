"""Tests for handshake response to physical arm ownership changes."""

import json
import threading

from g1_handshake.handshake_node import HandshakeNode


class Message:
    """Minimal std_msgs/String-compatible message."""

    def __init__(self, data):
        self.data = data


class Worker:
    """Report a controlled worker liveness state."""

    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


class Interaction:
    """Record external-yield requests."""

    def __init__(self):
        self.yields = 0

    def request_external_yield(self):
        self.yields += 1


class Logger:
    """Record log messages needed by the callback contract."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)


def node(state="IDLE", worker=None):
    """Construct callback state without initializing ROS middleware."""
    instance = HandshakeNode.__new__(HandshakeNode)
    instance.bridge_name = "hardware"
    instance.arm_ownership = "IDLE"
    instance.arm_action_id = 0
    instance.arm_action_name = ""
    instance.state = state
    instance.worker = worker
    instance.operation_lock = threading.Lock()
    instance.interaction = Interaction()
    instance.timer_cancelled = False
    instance.cancel_presentation_timer = lambda: setattr(
        instance, "timer_cancelled", True)
    instance.states = []
    instance.set_state = lambda value: (
        setattr(instance, "state", value), instance.states.append(value))
    logger = Logger()
    instance.get_logger = lambda: logger
    return instance, logger


def ownership_message(status, action_id, name=""):
    """Build one bridge ownership sample."""
    return Message(json.dumps({
        "status": status,
        "id": action_id,
        "name": name,
    }))


def test_external_action_interrupts_active_sequence_without_local_release():
    instance, _ = node("HOLDING", Worker(True))

    instance.arm_ownership_callback(
        ownership_message("EXTERNAL", 552, "vq_shake_hand"))

    assert instance.interaction.yields == 1
    assert instance.timer_cancelled is True
    assert instance.arm_ownership == "EXTERNAL"
    assert instance.arm_action_id == 552
    assert instance.states == []


def test_idle_feedback_recovers_external_state_after_worker_stops():
    instance, _ = node("EXTERNAL_ARM_CONTROL", Worker(False))

    instance.arm_ownership_callback(ownership_message("IDLE", 0))

    assert instance.state == "IDLE"
    assert instance.states == ["IDLE"]


def test_external_action_blocks_start_and_grip_with_action_details():
    instance, _ = node()
    instance.arm_ownership = "EXTERNAL"
    instance.arm_action_id = 552
    instance.arm_action_name = "vq_shake_hand"

    reason = instance.arm_command_allowed({"IDLE", "OWNED"})

    assert reason == (
        "external Unitree arm action is active: 552 vq_shake_hand")


def test_missing_feedback_fails_closed():
    instance, _ = node()
    instance.arm_ownership = "UNAVAILABLE"

    assert instance.arm_command_allowed({"IDLE"}) == (
        "Unitree arm state is unavailable or not stably idle")


def test_simulation_does_not_require_physical_ownership_feedback():
    instance, _ = node()
    instance.bridge_name = "simulation"
    instance.arm_ownership = "UNAVAILABLE"

    assert instance.arm_command_allowed({"IDLE"}) is None
