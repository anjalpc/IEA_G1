"""Tests for the fixed left-hand card profile adapter."""

from g1_card_presentation.card_backends import CardHand


class HandRecorder:
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


def test_card_hand_always_uses_verified_left_profile():
    client = HandRecorder()
    target = [0.78, 1.0, 0.4, 1.0, 1.0, 1.0]
    hand = CardHand(client, target, 1.5, 1.5)
    hand.prepare_operation()
    hand.grip()
    hand.release()
    assert client.calls == [
        ("prepare",),
        ("command", "left", target, 1.5),
        ("neutral", "left", 1.5),
    ]
