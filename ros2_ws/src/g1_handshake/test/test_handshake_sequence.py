import threading

import pytest

from g1_handshake.handshake_sequence import (
    AutomaticHandshake, SafeStateCoordinator, SafeStateError)


class ArmRecorder:
    def __init__(self, events): self.events = events
    def prepare_operation(self): self.events.append(("arm", "prepare"))
    def execute(self, action): self.events.append(("arm", action))
    def release(self): self.events.append(("arm", "release"))


class HandRecorder:
    def __init__(self, events): self.events = events
    def prepare_operation(self): self.events.append(("hand", "prepare"))
    def grip(self): self.events.append(("hand", "grip"))
    def release(self): self.events.append(("hand", "release"))


def make_sequence(hold=0.0, settle=0.0, wait_for_grip=lambda: True):
    events, states, clock = [], [], [0.0]

    def monotonic():
        clock[0] += 1.0
        return clock[0]

    sequence = AutomaticHandshake(
        ArmRecorder(events), HandRecorder(events), hold, settle,
        threading.Event(), states.append, sleep=lambda _: None,
        monotonic=monotonic, wait_for_grip=wait_for_grip)
    return sequence, events, states


def test_grip_command_completes_hold_then_safe_release():
    sequence, events, states = make_sequence(hold=1.0)
    assert sequence.run() == "handshake complete"
    assert events == [
        ("arm", 27), ("hand", "grip"),
        ("hand", "release"), ("arm", "release")]
    assert states == [
        "PRESENTING", "WAITING_FOR_GRIP", "GRIPPING", "HOLDING",
        "RELEASING", "COMPLETE"]


def test_grip_timeout_releases_without_gripping():
    sequence, events, states = make_sequence(
        hold=5.0, wait_for_grip=lambda: False)
    assert sequence.run() == (
        "grip command timed out; hand opened and arm released")
    assert ("hand", "grip") not in events
    assert events[-2:] == [("hand", "release"), ("arm", "release")]
    assert states == [
        "PRESENTING", "WAITING_FOR_GRIP", "GRIP_TIMEOUT", "RELEASING",
        "COMPLETE"]


def test_cancel_while_waiting_releases_without_gripping():
    sequence, events, states = make_sequence()

    def cancel_wait():
        sequence.cancelled.set()
        return False

    sequence.wait_for_grip = cancel_wait
    assert sequence.run() == "handshake cancelled"
    assert ("hand", "grip") not in events
    assert events[-2:] == [("hand", "release"), ("arm", "release")]
    assert states[-1] == "CANCELLED"


def test_arm_release_is_attempted_when_hand_release_fails():
    sequence, events, _ = make_sequence()

    def fail_release():
        events.append(("hand", "release"))
        raise RuntimeError("hand fault")

    sequence.hand.release = fail_release
    with pytest.raises(RuntimeError, match="hand fault"):
        sequence.run()
    assert ("arm", "release") in events


def test_failed_present_action_never_grips():
    sequence, events, states = make_sequence()
    sequence.arm.execute = lambda action: 7402
    with pytest.raises(RuntimeError, match="action 27 returned 7402"):
        sequence.run()
    assert events == [] and states == ["PRESENTING"]


def test_run_preserves_cancel_requested_before_worker_starts():
    sequence, events, states = make_sequence()
    sequence.cancelled.set()
    assert sequence.run() == "handshake cancelled"
    assert ("hand", "grip") not in events
    assert events[-2:] == [("hand", "release"), ("arm", "release")]
    assert states[-1] == "CANCELLED"


def test_safe_state_cleanup_runs_once():
    events = []
    coordinator = SafeStateCoordinator(ArmRecorder(events), HandRecorder(events))
    coordinator.release_once()
    coordinator.release_once()
    assert events == [("hand", "release"), ("arm", "release")]


def test_safe_state_attempts_arm_release_when_hand_release_fails():
    events = []
    arm, hand = ArmRecorder(events), HandRecorder(events)
    hand.release = lambda: (_ for _ in ()).throw(RuntimeError("hand fault"))
    coordinator = SafeStateCoordinator(arm, hand)
    with pytest.raises(SafeStateError, match="hand release failed"):
        coordinator.release_once()
    assert ("arm", "release") in events


def test_prepare_failure_returns_both_devices_to_safe_state():
    sequence, events, _ = make_sequence()

    def fail_prepare():
        events.append(("hand", "prepare"))
        raise RuntimeError("hand prepare fault")

    sequence.hand.prepare_operation = fail_prepare
    with pytest.raises(RuntimeError, match="hand prepare fault"):
        sequence.prepare()
    assert sequence.cancelled.is_set()
    assert events == [
        ("arm", "prepare"), ("hand", "prepare"),
        ("hand", "release"), ("arm", "release")]


def test_prepare_and_release_failure_reports_both_errors():
    sequence, events, _ = make_sequence()
    sequence.hand.prepare_operation = lambda: (_ for _ in ()).throw(
        RuntimeError("prepare fault"))
    sequence.hand.release = lambda: (_ for _ in ()).throw(
        RuntimeError("release fault"))
    with pytest.raises(SafeStateError, match="safe release also failed"):
        sequence.prepare()
    assert ("arm", "release") in events
