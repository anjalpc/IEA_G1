import threading

import pytest

from g1_robot_bridge.handshake_sequence import HandshakeSequence


class Recorder:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def execute(self, action):
        self.events.append((self.name, action))

    def grip(self):
        self.events.append((self.name, "grip"))

    def release(self):
        self.events.append((self.name, "release"))


def make_sequence(duration=0.0, enabled=lambda: True):
    events = []
    states = []
    clock = iter((0.0, 0.0, 1.0))
    sequence = HandshakeSequence(
        Recorder(events, "arm"), Recorder(events, "hand"), duration,
        threading.Event(), enabled, states.append,
        sleep=lambda _: None, monotonic=lambda: next(clock),
    )
    return sequence, events, states


def test_success_orders_grip_and_release_around_arm_actions():
    sequence, events, states = make_sequence()
    sequence.run()
    assert events == [
        ("arm", 27), ("hand", "grip"),
        ("hand", "release"), ("arm", 99),
    ]
    assert states == ["PRESENTING", "GRIPPING", "RELEASING", "COMPLETE"]


def test_cancel_after_presenting_releases_without_gripping():
    sequence, events, states = make_sequence(enabled=lambda: False)
    sequence.run()
    assert events == [("arm", 27), ("hand", "release"), ("arm", 99)]
    assert states[-1] == "CANCELLED"


def test_arm_release_is_attempted_when_hand_release_fails():
    sequence, events, _ = make_sequence()

    def fail_release():
        events.append(("hand", "release"))
        raise RuntimeError("hand fault")

    sequence.hand.release = fail_release
    with pytest.raises(RuntimeError, match="hand fault"):
        sequence.run()
    assert ("arm", 99) in events
