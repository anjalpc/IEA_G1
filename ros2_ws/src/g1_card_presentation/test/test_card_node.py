"""Tests for the dashboard-facing card toggle callbacks."""

from types import SimpleNamespace

import pytest

from g1_card_presentation.card_node import CardPresentationNode
from g1_card_presentation.handover_detector import depth_statistics


class Response:
    """Minimal Trigger response double."""

    def __init__(self):
        self.success = False
        self.message = ""


def test_depth_statistics_reports_median_and_near25():
    import numpy as np

    depth = np.array([
        [0.0, 400.0, 500.0],
        [100.0, 600.0, 700.0],
    ])

    median, near25 = depth_statistics(depth, (1, 0, 2, 2))

    assert median == 550.0
    assert near25 == 475.0


class Sequence:
    """Expose distinguishable commands for callback routing."""

    def grip(self):
        return "grip"

    def release_card(self):
        return "release"

    def present(self):
        return "present"

    def retract(self):
        return "retract"

    def handover(self):
        return "handover"


def node(hand_state="OPEN", arm_state="DOWN", blocked=None):
    """Build a callback-only node double."""
    calls = []
    instance = SimpleNamespace(
        hand_state=hand_state,
        arm_state=arm_state,
        sequence=Sequence(),
        cancel_presentation_timer=lambda: calls.append(
            ("timer", "cancel")
        ),
        arm_start_block_reason=lambda: blocked,
    )

    def start(command, name):
        calls.append((command.__name__, name))
        return True

    instance._start_operation = start
    return instance, calls


@pytest.mark.parametrize(
    "hand_state, expected",
    [
        ("OPEN", ("grip", "hold")),
        ("GRIPPED", ("release_card", "release")),
    ],
)
def test_hold_toggles_hand(hand_state, expected):
    instance, calls = node(hand_state=hand_state)
    response = Response()

    CardPresentationNode.hold_callback(
        instance,
        None,
        response,
    )

    assert response.success
    assert expected in calls


@pytest.mark.parametrize(
    "arm_state, expected",
    [
        ("DOWN", ("present", "present")),
        ("PRESENTED", ("retract", "lower")),
    ],
)
def test_present_toggles_arm(arm_state, expected):
    instance, calls = node(
        hand_state="GRIPPED",
        arm_state=arm_state,
    )
    response = Response()

    CardPresentationNode.present_callback(
        instance,
        None,
        response,
    )

    assert response.success
    assert expected in calls
    assert instance.hand_state == "GRIPPED"


@pytest.mark.parametrize(
    "callback, hand_state, arm_state",
    [
        ("hold", "MOVING", "DOWN"),
        ("present", "OPEN", "MOVING"),
        ("present", "OPEN", "PREPARING"),
        ("present", "OPEN", "UNKNOWN"),
    ],
)
def test_toggle_rejects_transitional_states(
        callback,
        hand_state,
        arm_state):
    instance, calls = node(
        hand_state=hand_state,
        arm_state=arm_state,
    )
    response = Response()

    method = getattr(
        CardPresentationNode,
        f"{callback}_callback",
    )
    method(instance, None, response)

    assert not response.success
    assert not calls


def test_present_respects_arm_ownership_guard():
    instance, calls = node(
        arm_state="DOWN",
        blocked="external arm action active",
    )
    response = Response()

    CardPresentationNode.present_callback(
        instance,
        None,
        response,
    )

    assert not response.success
    assert response.message == "external arm action active"
    assert not calls


def test_presentation_timeout_lowers_without_releasing_hand():
    instance, calls = node(
        hand_state="GRIPPED",
        arm_state="PRESENTED",
    )
    instance.presentation_timer = object()
    instance.get_logger = lambda: SimpleNamespace(
        warning=lambda message: calls.append(("warning", message))
    )

    CardPresentationNode.presentation_timeout_callback(instance)

    assert instance.presentation_timer is None
    assert ("retract", "timeout-lower") in calls
    assert all(call[0] != "release_card" for call in calls)
    assert instance.hand_state == "GRIPPED"


def test_handover_accepts_only_gripped_presented_card():
    instance, calls = node(
        hand_state="GRIPPED",
        arm_state="PRESENTED",
    )
    response = Response()

    CardPresentationNode.handover_callback(
        instance,
        None,
        response,
    )

    assert response.success
    assert calls == [
        ("timer", "cancel"),
        ("handover", "handover"),
    ]


@pytest.mark.parametrize(
    "hand_state, arm_state",
    [
        ("OPEN", "PRESENTED"),
        ("GRIPPED", "DOWN"),
        ("MOVING", "PRESENTED"),
    ],
)
def test_handover_rejects_other_states(hand_state, arm_state):
    instance, calls = node(
        hand_state=hand_state,
        arm_state=arm_state,
    )
    response = Response()

    CardPresentationNode.handover_callback(
        instance,
        None,
        response,
    )

    assert not response.success
    assert not calls
