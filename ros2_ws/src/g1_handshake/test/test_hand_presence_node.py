"""Tests for the ROS-facing hand-presence node.

These tests deliberately avoid real ROS initialization, MediaPipe, cameras,
shared memory, Unitree SDK2, and robot actuation.

They test only HandPresenceNode's orchestration logic.
"""

import json

import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String

from g1_handshake.hand_presence_node import (
    ARMED_STATE,
    HandPresenceNode,
)
from g1_handshake.presence.hand_detector import (
    DetectedHand,
)
from g1_handshake.presence.target_region import TargetRegion


class FakeLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []

    def info(self, message):
        self.info_messages.append(str(message))

    def warning(self, message):
        self.warning_messages.append(str(message))

    def error(self, message):
        self.error_messages.append(str(message))


class FakeDetector:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.observations = []
        self.reset_count = 0

    def observe(self, hit):
        self.observations.append(bool(hit))

        if self.results:
            return self.results.pop(0)

        return False

    def reset(self):
        self.reset_count += 1


class FakeHandDetector:
    def __init__(self, hands=None):
        self.hands = list(hands or [])

    def detect(self, _frame):
        if self.hands:
            return self.hands.pop(0)

        return None


class FakeRgbClient:
    def __init__(self, readings=None):
        self.readings = list(readings or [])

    def read_latest(self):
        if self.readings:
            return self.readings.pop(0)

        return None, None, None


class FakeDepthCamera:
    def read_frame_mm(self, timeout_ms=100):
        del timeout_ms
        return np.full((480, 640), 500.0, dtype=np.float32)


def fresh_frame(sequence=1):
    return (
        np.zeros(
            (1080, 1920, 3),
            dtype=np.uint8,
        ),
        0.01,
        sequence,
    )


def make_node(
    detector=None,
    hand_detector=None,
    rgb_client=None,
    enable_grip_trigger=False,
):
    """Construct HandPresenceNode without invoking rclpy Node.__init__."""

    instance = HandPresenceNode.__new__(
        HandPresenceNode
    )

    logger = FakeLogger()

    instance.get_logger = lambda: logger

    instance.detector = (
        detector
        if detector is not None
        else FakeDetector()
    )

    instance.hand_detector = (
        hand_detector
        if hand_detector is not None
        else FakeHandDetector()
    )

    instance.rgb_client = (
        rgb_client
        if rgb_client is not None
        else FakeRgbClient()
    )
    instance.depth_camera = FakeDepthCamera()
    instance._read_rgb = instance.rgb_client.read_latest
    instance._read_depth = instance.depth_camera.read_frame_mm
    instance.depth_roi = (381, 170, 193, 181)
    instance.depth_near25_below_mm = 800.0
    instance.rgb_recent_s = 1.0

    instance.rgb_stale_after_s = 0.25
    instance.enable_grip_trigger = bool(
        enable_grip_trigger
    )
    instance.publish_debug_image = False
    instance.target_region = TargetRegion(0.0, 0.0, 1.0, 1.0, 0.0)

    instance._armed = False
    instance._fired_this_presentation = False
    instance._rgb_state = None
    instance._grip_pending = False
    instance._last_rgb_hit_at = None
    instance._depth_median_mm = None
    instance._depth_near25_mm = None
    instance._rgb_hit_state = None
    instance._depth_hit_state = None
    instance._latest_rgb = None
    instance._latest_depth = None
    instance._rgb_received_at = None
    instance._depth_received_at = None

    grip_calls = []

    def fake_call_grip():
        grip_calls.append(True)
        return True

    instance._call_grip = fake_call_grip

    return instance, logger, grip_calls


def state_message(state):
    message = String()
    message.data = json.dumps(
        {"state": state}
    )
    return message


def test_poll_does_nothing_when_not_armed():
    detector = FakeDetector()

    instance, _logger, _grip_calls = (
        make_node(
            detector=detector,
            rgb_client=FakeRgbClient(
                [fresh_frame()]
            ),
        )
    )

    instance._poll()

    assert detector.observations == []


def test_no_hand_detected_is_not_a_hit():
    detector = FakeDetector()

    instance, _logger, _grip_calls = (
        make_node(
            detector=detector,
            hand_detector=FakeHandDetector(
                hands=[None]
            ),
            rgb_client=FakeRgbClient(
                [fresh_frame()]
            ),
        )
    )

    instance._armed = True

    instance._poll()

    assert detector.observations == [False]
    assert instance._fired_this_presentation is False


def test_depth_callback_accepts_shared_16uc1_frame():
    instance, _logger, _grip_calls = make_node()
    message = Image()
    message.height = 2
    message.width = 3
    message.encoding = "16UC1"
    message.data = np.array(
        [[400, 500, 600], [700, 800, 900]], dtype=np.uint16
    ).tobytes()

    instance._depth_callback(message)

    assert instance._latest_depth.tolist() == [
        [400.0, 500.0, 600.0],
        [700.0, 800.0, 900.0],
    ]
    assert instance._depth_received_at is not None


def test_detected_hand_is_a_hit():
    detector = FakeDetector()

    hand = DetectedHand(
        center_x=500,
        center_y=400,
        confidence=0.95,
        bbox=(450, 350, 550, 450),
    )

    instance, _logger, _grip_calls = (
        make_node(
            detector=detector,
            hand_detector=FakeHandDetector(
                hands=[hand]
            ),
            rgb_client=FakeRgbClient(
                [fresh_frame()]
            ),
        )
    )

    instance._armed = True

    instance._poll()

    assert detector.observations == [True]


def test_gate_states_log_only_on_transition():
    instance, logger, _grip_calls = make_node()
    instance._armed = True
    instance._depth_near25_mm = 425.0

    instance._log_gate_transitions(True, True)
    instance._log_gate_transitions(True, True)
    instance._log_gate_transitions(False, False)

    gate_messages = [
        message
        for message in logger.info_messages
        if "gate" in message
    ]
    assert gate_messages == [
        "RGB gate HIT",
        "depth gate HIT (near25=425mm)",
        "RGB gate CLEAR",
        "depth gate CLEAR (near25=425mm)",
    ]


def test_arming_resets_gate_transition_state():
    instance, _logger, _grip_calls = make_node()
    instance._rgb_hit_state = True
    instance._depth_hit_state = True

    instance._state_callback(state_message(ARMED_STATE))

    assert instance._rgb_hit_state is None
    assert instance._depth_hit_state is None


def test_unavailable_rgb_resets_stability():
    detector = FakeDetector()

    instance, logger, _grip_calls = (
        make_node(
            detector=detector,
            rgb_client=FakeRgbClient(
                [(None, None, None)]
            ),
        )
    )

    instance._armed = True

    instance._poll()

    assert detector.reset_count == 1
    assert detector.observations == []

    assert any(
        "UNAVAILABLE" in message
        for message in logger.warning_messages
    )


def test_stale_rgb_resets_stability():
    detector = FakeDetector()

    stale = (
        np.zeros(
            (1080, 1920, 3),
            dtype=np.uint8,
        ),
        0.50,
        10,
    )

    instance, logger, _grip_calls = (
        make_node(
            detector=detector,
            rgb_client=FakeRgbClient(
                [stale]
            ),
        )
    )

    instance._armed = True

    instance._poll()

    assert detector.reset_count == 1
    assert detector.observations == []

    assert any(
        "STALE" in message
        for message in logger.warning_messages
    )


def test_fresh_rgb_transitions_to_ready():
    instance, logger, _grip_calls = (
        make_node(
            rgb_client=FakeRgbClient(
                [fresh_frame()]
            ),
            hand_detector=FakeHandDetector(
                hands=[None]
            ),
        )
    )

    instance._armed = True

    instance._poll()

    assert instance._rgb_state == "READY"

    assert any(
        "READY" in message
        for message in logger.info_messages
    )


def test_rgb_state_warning_is_not_repeated_every_poll():
    detector = FakeDetector()

    instance, logger, _grip_calls = (
        make_node(
            detector=detector,
            rgb_client=FakeRgbClient(
                [
                    (None, None, None),
                    (None, None, None),
                ]
            ),
        )
    )

    instance._armed = True

    instance._poll()
    instance._poll()

    unavailable_warnings = [
        message
        for message in logger.warning_messages
        if "UNAVAILABLE" in message
    ]

    assert len(unavailable_warnings) == 1


def test_stable_hand_latches_once_when_grip_disabled():
    detector = FakeDetector(
        results=[True]
    )

    hand = DetectedHand(
        center_x=500,
        center_y=400,
        confidence=0.95,
        bbox=(450, 350, 550, 450),
    )

    instance, logger, grip_calls = (
        make_node(
            detector=detector,
            hand_detector=FakeHandDetector(
                hands=[hand]
            ),
            rgb_client=FakeRgbClient(
                [fresh_frame()]
            ),
            enable_grip_trigger=False,
        )
    )

    instance._armed = True

    instance._poll()

    assert (
        instance._fired_this_presentation
        is True
    )

    assert grip_calls == []

    assert any(
        "DISABLED" in message
        for message in logger.warning_messages
    )


def test_stable_hand_calls_grip_when_enabled():
    detector = FakeDetector(
        results=[True]
    )

    hand = DetectedHand(
        center_x=500,
        center_y=400,
        confidence=0.95,
        bbox=(450, 350, 550, 450),
    )

    instance, _logger, grip_calls = (
        make_node(
            detector=detector,
            hand_detector=FakeHandDetector(
                hands=[hand]
            ),
            rgb_client=FakeRgbClient(
                [fresh_frame()]
            ),
            enable_grip_trigger=True,
        )
    )

    instance._armed = True

    instance._poll()

    assert (
        instance._grip_pending
        is True
    )

    assert len(grip_calls) == 1


def test_fired_presentation_does_not_fire_again():
    detector = FakeDetector(
        results=[True, True]
    )

    hand = DetectedHand(
        center_x=500,
        center_y=400,
        confidence=0.95,
        bbox=(450, 350, 550, 450),
    )

    instance, _logger, grip_calls = (
        make_node(
            detector=detector,
            hand_detector=FakeHandDetector(
                hands=[hand, hand]
            ),
            rgb_client=FakeRgbClient(
                [
                    fresh_frame(1),
                    fresh_frame(2),
                ]
            ),
            enable_grip_trigger=True,
        )
    )

    instance._armed = True

    instance._poll()
    instance._poll()

    assert len(grip_calls) == 1


def test_hand_detection_failure_resets_stability():
    class FailingHandDetector:
        def detect(self, _frame):
            raise RuntimeError(
                "model unavailable"
            )

    detector = FakeDetector()

    instance, logger, _grip_calls = (
        make_node(
            detector=detector,
            hand_detector=FailingHandDetector(),
            rgb_client=FakeRgbClient(
                [fresh_frame()]
            ),
        )
    )

    instance._armed = True

    instance._poll()

    assert detector.reset_count == 1

    assert any(
        "hand detection failed" in message
        for message in logger.error_messages
    )


def test_entering_arm_presented_resets_and_arms():
    detector = FakeDetector()

    instance, _logger, _grip_calls = (
        make_node(
            detector=detector
        )
    )

    instance._fired_this_presentation = True

    instance._state_callback(
        state_message(
            ARMED_STATE
        )
    )

    assert instance._armed is True
    assert (
        instance._fired_this_presentation
        is False
    )
    assert detector.reset_count == 1


def test_leaving_arm_presented_resets_detector():
    detector = FakeDetector()

    instance, _logger, _grip_calls = (
        make_node(
            detector=detector
        )
    )

    instance._armed = True

    instance._state_callback(
        state_message(
            "IDLE"
        )
    )

    assert instance._armed is False
    assert detector.reset_count == 1


def test_invalid_state_message_does_not_crash():
    detector = FakeDetector()

    instance, logger, _grip_calls = (
        make_node(
            detector=detector
        )
    )

    message = String()
    message.data = "not-json"

    instance._state_callback(
        message
    )

    assert instance._armed is False

    assert any(
        "invalid handshake state" in message
        for message in logger.error_messages
    )
