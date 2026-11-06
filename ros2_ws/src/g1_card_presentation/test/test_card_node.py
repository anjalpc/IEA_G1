"""Tests for the dashboard-facing card toggle callbacks."""

from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from g1_card_presentation.card_node import CardPresentationNode
from g1_card_presentation.handover_detector import (
    CardHandoverDetector,
    depth_statistics,
    hand_in_roi,
    roi_profile,
)


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


def test_card_handover_requires_semantic_hand_inside_rgb_region():
    roi = (100, 200, 300, 400)

    assert hand_in_roi(
        {"hands": [{"center_x": 250, "center_y": 300}]},
        roi,
    )
    assert not hand_in_roi({"hands": []}, roi)
    assert not hand_in_roi(
        {"hands": [{"center_x": 50, "center_y": 300}]},
        roi,
    )
    assert hand_in_roi(
        {
            "hands": [{
                "center_x": 250,
                "center_y": 150,
                "bbox": [200, 100, 300, 220],
            }],
        },
        roi,
    )


def test_right_profile_rejects_hands_in_left_half_of_image():
    config = (
        Path(__file__).parents[1]
        / "config"
        / "card_presentation.yaml"
    )
    parameters = yaml.safe_load(
        config.read_text(encoding="utf-8")
    )["card_handover_detector"]["ros__parameters"]
    roi = tuple(
        parameters[f"right_hand_roi_{part}"]
        for part in ("x", "y", "width", "height")
    )

    assert not hand_in_roi(
        {"hands": [{"center_x": 100, "center_y": 240}]},
        roi,
    )
    assert hand_in_roi(
        {"hands": [{"center_x": 500, "center_y": 200}]},
        roi,
    )


def test_card_side_selects_complete_matching_roi_profile():
    values = {
        f"{side}_{kind}_roi_{part}": index
        for index, (side, kind, part) in enumerate(
            (
                (side, kind, part)
                for side in ("left", "right")
                for kind in ("rgb", "hand", "depth")
                for part in ("x", "y", "width", "height")
            ),
            start=1,
        )
    }

    left = roi_profile(values.__getitem__, "left")
    right = roi_profile(values.__getitem__, "right")

    assert left == {
        "rgb": (1, 2, 3, 4),
        "hand": (5, 6, 7, 8),
        "depth": (9, 10, 11, 12),
    }
    assert right == {
        "rgb": (13, 14, 15, 16),
        "hand": (17, 18, 19, 20),
        "depth": (21, 22, 23, 24),
    }


def test_card_state_automatically_switches_detector_roi_profile():
    profiles = {
        "left": {
            "rgb": (1, 2, 3, 4),
            "hand": (5, 6, 7, 8),
            "depth": (9, 10, 11, 12),
        },
        "right": {
            "rgb": (13, 14, 15, 16),
            "hand": (17, 18, 19, 20),
            "depth": (21, 22, 23, 24),
        },
    }
    messages = []
    instance = SimpleNamespace(
        roi_profiles=profiles,
        hand_side=None,
        rgb_roi=None,
        hand_roi=None,
        depth_roi=None,
        armed=False,
        fired=False,
        pending=False,
        hits=deque(),
        waiting_hand_hits=deque(),
        last_rgb_hit_at=None,
        last_depth_hit_at=None,
        last_hand_hit_at=None,
        get_logger=lambda: SimpleNamespace(
            info=messages.append,
            error=messages.append,
        ),
    )

    for side in ("left", "right"):
        message = SimpleNamespace(data=json.dumps({
            "hand": side,
            "hand_state": "OPEN",
            "arm_state": "DOWN",
        }))
        CardHandoverDetector.state_callback(instance, message)
        assert instance.hand_side == side
        assert instance.rgb_roi == profiles[side]["rgb"]
        assert instance.hand_roi == profiles[side]["hand"]
        assert instance.depth_roi == profiles[side]["depth"]


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
