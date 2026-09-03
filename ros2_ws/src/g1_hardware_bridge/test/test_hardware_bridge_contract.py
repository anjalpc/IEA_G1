"""Contract tests for physical bridge routing and safe shutdown."""

from g1_hardware_bridge.hardware_bridge_node import G1HardwareBridge


class HandRecorder:
    """Record physical hand backend calls."""

    def __init__(self, events):
        self.events = events

    def command_position(self, hand, target, ramp):
        self.events.append(("hand_command", hand, target, ramp))
        return {"commanded": True}

    def neutral(self, hand, target, ramp):
        self.events.append(("hand_neutral", hand, target, ramp))
        return {"commanded": True}

    def interrupt(self):
        self.events.append(("hand_interrupt",))


class ArmRecorder:
    """Record physical arm backend calls."""

    def __init__(self, events):
        self.events = events

    def release(self):
        self.events.append(("arm_release",))
        return 0


class Logger:
    """Minimal logger accepted by safe_shutdown."""

    def info(self, _message):
        pass

    def error(self, _message):
        pass


def bridge():
    """Construct the routing portion without initializing ROS or hardware."""
    events = []
    instance = G1HardwareBridge.__new__(G1HardwareBridge)
    instance.hand = HandRecorder(events)
    instance.arm = ArmRecorder(events)
    instance.safe_hand_ramp = 0.2
    instance.neutral_targets = {
        "right": [0.399, 0.399, 0.050, 0.050, 0.049, 0.050],
        "left": [0.400, 0.400, 0.049, 0.050, 0.049, 0.050],
    }
    instance._owned_hands = set()
    instance.get_logger = lambda: Logger()
    return instance, events


def test_routes_exact_hand_target_and_tracks_selected_hand():
    instance, events = bridge()
    target = [0.48, 0.45, 0.15, 0.20, 0.25, 0.20]

    result = instance._hand_command({
        "hand": "right", "target": target, "ramp_s": 0.2})

    assert result == {"commanded": True}
    assert events == [("hand_command", "right", target, 0.2)]
    assert instance._owned_hands == {"right"}


def test_rejects_invalid_hand_before_hardware_dispatch():
    instance, events = bridge()

    try:
        instance._hand_command({
            "hand": "both", "target": [0.2] * 6, "ramp_s": 0.2})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid hand was accepted")

    assert events == []
    assert instance._owned_hands == set()


def test_shutdown_neutralizes_owned_hand_before_releasing_arm():
    instance, events = bridge()
    instance._owned_hands.add("right")

    instance.safe_shutdown()

    assert events == [
        ("hand_neutral", "right", instance.neutral_targets["right"], 0.2),
        ("arm_release",),
    ]


def test_shutdown_neutralizes_both_owned_hands_in_deterministic_order():
    instance, events = bridge()
    instance._owned_hands.update(("right", "left"))

    instance.safe_shutdown()

    assert events == [
        ("hand_neutral", "left", instance.neutral_targets["left"], 0.2),
        ("hand_neutral", "right", instance.neutral_targets["right"], 0.2),
        ("arm_release",),
    ]
