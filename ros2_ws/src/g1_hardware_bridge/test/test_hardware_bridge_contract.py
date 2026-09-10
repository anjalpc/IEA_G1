"""Contract tests for physical bridge routing and safe shutdown."""

from g1_hardware_bridge.arm.ownership import ArmOwnership
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

    def __init__(self, events, release_code=0):
        self.events = events
        self.release_code = release_code

    def has_motion(self):
        return False

    def prepare_operation(self):
        self.events.append(("arm_prepare",))

    def prepare_motion(self):
        self.events.append(("arm_prepare",))

    def execute(self, action_id):
        self.events.append(("arm_execute", action_id))
        return 0

    def execute_motion(self, motion, execution_timeout_s):
        self.events.append(("arm_motion", motion, execution_timeout_s))
        return {"mode": "live"}

    def release(self):
        self.events.append(("arm_release",))
        return self.release_code


class Logger:
    """Minimal logger accepted by safe_shutdown."""

    def info(self, _message):
        pass

    def error(self, _message):
        pass

    def warning(self, _message):
        pass


class OwnershipRecorder:
    """Provide controllable ownership results for routing tests."""

    def __init__(self, status=ArmOwnership.IDLE):
        self.status = status
        self.expected = []

    def snapshot(self):
        return {
            "status": self.status,
            "id": 552 if self.status == ArmOwnership.EXTERNAL else 0,
            "name": "vq_shake_hand" if self.status == ArmOwnership.EXTERNAL
            else "",
            "holding": self.status == ArmOwnership.EXTERNAL,
            "expected_id": None,
            "idle_stable": self.status == ArmOwnership.IDLE,
        }

    def require_available(self):
        state = self.snapshot()
        if self.status != ArmOwnership.IDLE:
            raise RuntimeError(ArmOwnership.block_reason(state))
        return state

    def expect(self, action_id):
        self.expected.append(int(action_id))

    def clear_expectation(self, action_id=None):
        del action_id


class TimerRecorder:
    """Record timer cancellation."""

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class MonitorRecorder:
    """Record monitor shutdown."""

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def bridge():
    """Construct the routing portion without initializing ROS or hardware."""
    events = []
    instance = G1HardwareBridge.__new__(G1HardwareBridge)
    instance.hand = HandRecorder(events)
    instance.arm = ArmRecorder(events)
    instance._arm_lease_lock = __import__("threading").Lock()
    instance._arm_lease_client = None
    instance.safe_hand_ramp = 0.2
    instance.neutral_targets = {
        "right": [0.399, 0.399, 0.050, 0.050, 0.049, 0.050],
        "left": [0.400, 0.400, 0.049, 0.050, 0.049, 0.050],
    }
    instance._owned_hands = set()
    instance.allowed_arm_actions = {27}
    instance.arm_ownership = OwnershipRecorder()
    instance._test_logger = Logger()
    instance.get_logger = lambda: instance._test_logger
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


def test_arm_monitor_shutdown_cancels_timer_and_stops_process():
    instance = G1HardwareBridge.__new__(G1HardwareBridge)
    instance.arm_ownership_timer = TimerRecorder()
    instance.arm_monitor = MonitorRecorder()

    instance.stop_arm_monitor()

    assert instance.arm_ownership_timer.cancelled is True
    assert instance.arm_monitor.stopped is True


def test_arm_execute_requires_idle_then_records_expected_action():
    instance, events = bridge()
    instance._arm_prepare({"client_id": "client-a"})

    assert instance._arm_execute({
        "client_id": "client-a", "action_id": 27}) == {"code": 0}

    assert events == [("arm_prepare",), ("arm_execute", 27)]
    assert instance.arm_ownership.expected == [27]


def test_arm_motion_routes_neutral_request_unchanged():
    """Route neutral data without introducing behavior names in the bridge."""
    instance, events = bridge()
    motion = {
        "version": 1,
        "preparation": {"left_elbow_joint": 0.978},
        "target_offsets": {"left_elbow_joint": -0.619944},
        "phase_seconds": [0.3, 0.9],
        "recovery": ["preparation", "baseline"],
    }

    instance._arm_prepare_motion({"client_id": "client-a"})
    result = instance._arm_motion({
        "client_id": "client-a",
        "motion": motion,
        "execution_timeout_s": 8.0,
    })

    assert result == {"mode": "live"}
    assert events == [("arm_prepare",), ("arm_motion", motion, 8.0)]


def test_external_action_blocks_prepare_and_execute():
    instance, events = bridge()
    instance.arm_ownership.status = ArmOwnership.EXTERNAL

    for command in (
            lambda: instance._arm_prepare({"client_id": "client-a"}),
            lambda: instance._arm_execute({
                "client_id": "client-a", "action_id": 27})):
        try:
            command()
        except RuntimeError as exc:
            assert "external Unitree arm action" in str(exc)
        else:
            raise AssertionError("external action was not blocked")

    assert events == []


def test_owned_action_allows_prepare_for_existing_lease():
    instance, events = bridge()
    instance._arm_lease_client = "client-a"
    instance.arm_ownership.status = ArmOwnership.OWNED

    instance._arm_prepare({"client_id": "client-a"})

    assert events == [("arm_prepare",)]


def test_arm_lease_rejects_a_second_client_until_release():
    import pytest

    instance, events = bridge()
    instance._arm_prepare({"client_id": "client-a"})
    with pytest.raises(RuntimeError, match="leased by another"):
        instance._arm_prepare({"client_id": "client-b"})
    instance._arm_release({"client_id": "client-a"})
    instance._arm_prepare({"client_id": "client-b"})

    assert events == [
        ("arm_prepare",), ("arm_release",), ("arm_prepare",)]


def test_external_action_blocks_release_action_99():
    instance, events = bridge()
    instance.arm_ownership.status = ArmOwnership.EXTERNAL

    try:
        instance._arm_release({"client_id": "client-a"})
    except RuntimeError as exc:
        assert "external Unitree arm action" in str(exc)
    else:
        raise AssertionError("external action did not block release")

    assert events == []


def test_shutdown_does_not_release_externally_owned_arm():
    instance, events = bridge()
    instance.arm_ownership.status = ArmOwnership.EXTERNAL

    instance.safe_shutdown()

    assert events == []


def test_shutdown_reports_nonzero_arm_release_as_incomplete():
    instance, events = bridge()
    errors = []
    instance.arm.release_code = 7400
    instance.get_logger().error = errors.append

    instance.safe_shutdown()

    assert events == [("arm_release",)]
    assert len(errors) == 1
    assert "release action returned code 7400" in errors[0]


def test_card_prepare_and_present_reject_external_owner():
    import pytest

    instance, events = bridge()
    instance.arm_ownership.status = ArmOwnership.EXTERNAL
    with pytest.raises(RuntimeError, match="external Unitree arm action"):
        instance._arm_prepare_motion({"client_id": "client-a"})
    with pytest.raises(RuntimeError, match="external Unitree arm action"):
        instance._arm_motion({"client_id": "client-a"})
    assert events == []


def test_shutdown_releases_local_card_even_if_native_monitor_unavailable():
    instance, events = bridge()
    instance.arm_ownership.status = ArmOwnership.UNAVAILABLE
    instance.arm.has_motion = lambda: True
    instance.arm.release_motion = lambda: events.append(("motion_release",))
    instance.safe_shutdown()
    assert events == [("motion_release",)]
