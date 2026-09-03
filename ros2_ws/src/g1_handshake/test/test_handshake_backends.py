"""Tests for adapting generic hand commands to handshake behavior."""

from pathlib import Path

import yaml

from g1_handshake.handshake_backends import HandshakeHand


class HandRecorder:
    """Record generic hand-client calls."""

    def __init__(self):
        self.calls = []

    def prepare_operation(self):
        self.calls.append(("prepare",))

    def interrupt(self):
        self.calls.append(("interrupt",))

    def command_position(self, hand, target, ramp):
        self.calls.append(("command", hand, target, ramp))

    def neutral(self, hand, ramp):
        self.calls.append(("neutral", hand, ramp))


def test_maps_handshake_operations_to_configured_generic_commands():
    client = HandRecorder()
    target = [0.78, 1.0, 0.4, 1.0, 1.0, 1.0]
    hand = HandshakeHand(client, "left", target, 0.2, 0.3)

    hand.prepare_operation()
    hand.grip()
    hand.release()
    hand.interrupt()

    assert client.calls == [
        ("prepare",),
        ("command", "left", target, 0.2),
        ("neutral", "left", 0.3),
        ("interrupt",),
    ]


def test_deployment_profile_commands_verified_right_hand_and_coordinates():
    config_path = Path(__file__).parents[1] / "config" / "handshake.yaml"
    parameters = yaml.safe_load(config_path.read_text())[
        "event_handshake"]["ros__parameters"]
    client = HandRecorder()
    hand = HandshakeHand(
        client,
        parameters["hand"],
        parameters["grip_target"],
        parameters["grip_ramp_s"],
        parameters["release_ramp_s"],
    )

    hand.prepare_operation()
    hand.grip()
    hand.release()

    assert parameters["hand"] == "right"
    assert parameters["grip_target"] == [
        0.48, 0.45, 0.15, 0.20, 0.25, 0.20]
    assert client.calls == [
        ("prepare",),
        ("command", "right", parameters["grip_target"], 0.2),
        ("neutral", "right", 0.2),
    ]
