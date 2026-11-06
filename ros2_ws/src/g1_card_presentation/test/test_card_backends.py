"""Tests for the side-configurable card profile adapter."""

import pytest

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


@pytest.mark.parametrize("side", ["left", "right"])
def test_card_hand_uses_configured_profile(side):
    client = HandRecorder()
    target = [0.78, 1.0, 0.4, 1.0, 1.0, 1.0]
    hand = CardHand(client, side, target, 1.5, 1.5)
    hand.prepare_operation()
    hand.grip()
    hand.release()
    assert client.calls == [
        ("prepare",),
        ("command", side, target, 1.5),
        ("neutral", side, 1.5),
    ]


def test_card_hand_rejects_unknown_side():
    with pytest.raises(ValueError, match="left.*right"):
        CardHand(HandRecorder(), "centre", [0.1] * 6, 1.0, 1.0)
