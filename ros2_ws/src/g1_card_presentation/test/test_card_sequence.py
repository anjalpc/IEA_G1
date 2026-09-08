"""Tests for independent, interruptible card operations."""

import threading

import pytest

from g1_card_presentation.card_sequence import (
    CardPresentation,
    CardSafeStateCoordinator,
    CardSafeStateError,
)


class Recorder:
    """Record arm or hand backend operations."""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def prepare_operation(self):
        self.calls.append((self.name, "prepare"))

    def prepare_presentation(self):
        self.calls.append((self.name, "prepare"))

    def grip(self):
        self.calls.append((self.name, "grip"))
        return "gripped"

    def present(self, pose, fraction, move_seconds):
        self.calls.append((
            self.name,
            "present",
            pose,
            fraction,
            move_seconds,
        ))
        return {"mode": "simulation"}

    def release_presentation(self):
        self.calls.append((self.name, "release"))
        return "arm released"

    def release(self):
        self.calls.append((self.name, "release"))
        return "card released"

    def interrupt(self):
        self.calls.append((self.name, "interrupt"))


class BlockingRecorder(Recorder):
    """Block preparation until interrupt is received."""

    def __init__(self, name, calls):
        super().__init__(name, calls)
        self.prepare_entered = threading.Event()
        self.interrupted = threading.Event()

    def _block_prepare(self):
        self.calls.append((self.name, "prepare"))
        self.prepare_entered.set()

        if not self.interrupted.wait(1.0):
            raise RuntimeError("test preparation was not interrupted")

    def prepare_operation(self):
        self._block_prepare()

    def prepare_presentation(self):
        self._block_prepare()

    def interrupt(self):
        super().interrupt()
        self.interrupted.set()


class FailingReleaseRecorder(Recorder):
    """Raise while attempting safety release."""

    def release(self):
        super().release()
        raise RuntimeError("release rejected")


def sequence(calls, states, arm=None, hand=None):
    """Build a zero-dwell card controller."""
    return CardPresentation(
        arm or Recorder("arm", calls),
        hand or Recorder("hand", calls),
        "left_card_presentation",
        0.25,
        1.2,
        set_state=states.append,
        release_dwell_s=0.0,
    )


def test_independent_full_card_cycle():
    calls = []
    states = []
    interaction = sequence(calls, states)

    assert interaction.grip() == "gripped"
    assert interaction.present() == {"mode": "simulation"}
    assert interaction.release_card() == "card released"
    assert interaction.retract() == "arm released"

    assert calls == [
        ("hand", "prepare"),
        ("hand", "grip"),
        ("arm", "prepare"),
        (
            "arm",
            "present",
            "left_card_presentation",
            0.25,
            1.2,
        ),
        ("hand", "release"),
        ("arm", "release"),
    ]

    assert states == [
        "GRIPPING_CARD",
        "CARD_GRIPPED",
        "PREPARING",
        "PRESENTING",
        "CARD_PRESENTED",
        "RELEASING_CARD",
        "CARD_RELEASED",
        "RETRACTING",
        "ARM_DOWN",
    ]

    assert not interaction.interaction_active
    assert not interaction.hand_prepared
    assert not interaction.arm_prepared


def test_cancel_releases_hand_before_arm():
    calls = []
    states = []
    interaction = sequence(calls, states)

    interaction.grip()
    interaction.present()
    assert interaction.cancel() == "card presentation cancelled"

    hand_release = calls.index(("hand", "release"))
    arm_release = calls.index(("arm", "release"))

    assert hand_release < arm_release
    assert states[-3:] == [
        "RELEASING_CARD",
        "RETRACTING",
        "CANCELLED",
    ]


def test_partial_hand_cleanup_does_not_suppress_arm_cleanup():
    calls = []
    states = []
    arm = Recorder("arm", calls)
    hand = Recorder("hand", calls)
    safe_state = CardSafeStateCoordinator(
        arm,
        hand,
        states.append,
        release_dwell_s=0.0,
    )

    safe_state.reset()
    safe_state.release_once(
        release_hand=True,
        release_arm=False,
    )
    safe_state.release_once(
        release_hand=False,
        release_arm=True,
    )

    assert calls == [
        ("hand", "release"),
        ("arm", "release"),
    ]
    assert safe_state.complete


def test_repeated_cleanup_releases_each_subsystem_once():
    calls = []
    states = []
    safe_state = CardSafeStateCoordinator(
        Recorder("arm", calls),
        Recorder("hand", calls),
        states.append,
        release_dwell_s=0.0,
    )

    safe_state.reset()

    workers = [
        threading.Thread(
            target=safe_state.release_once
        )
        for _ in range(3)
    ]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join(1.0)
        assert not worker.is_alive()

    assert calls.count(("hand", "release")) == 1
    assert calls.count(("arm", "release")) == 1


def test_cleanup_failure_remains_visible():
    calls = []
    states = []
    safe_state = CardSafeStateCoordinator(
        Recorder("arm", calls),
        FailingReleaseRecorder("hand", calls),
        states.append,
        release_dwell_s=0.0,
    )

    safe_state.reset()

    with pytest.raises(
        CardSafeStateError,
        match="left hand release failed",
    ):
        safe_state.release_once(
            release_hand=True,
            release_arm=False,
        )

    with pytest.raises(
        CardSafeStateError,
        match="left hand release failed",
    ):
        safe_state.release_once(
            release_hand=True,
            release_arm=False,
        )

    assert calls.count(("hand", "release")) == 1


@pytest.mark.parametrize("subsystem", ["hand", "arm"])
def test_interrupt_during_preparation_is_cleaned_after_worker_stops(
        subsystem):
    calls = []
    states = []

    arm = (
        BlockingRecorder("arm", calls)
        if subsystem == "arm"
        else Recorder("arm", calls)
    )
    hand = (
        BlockingRecorder("hand", calls)
        if subsystem == "hand"
        else Recorder("hand", calls)
    )

    interaction = sequence(
        calls,
        states,
        arm=arm,
        hand=hand,
    )

    operation = (
        interaction.present
        if subsystem == "arm"
        else interaction.grip
    )

    worker = threading.Thread(target=operation)
    worker.start()

    blocking_backend = (
        arm if subsystem == "arm" else hand
    )
    assert blocking_backend.prepare_entered.wait(1.0)

    interaction.interrupt_active()

    worker.join(1.0)
    assert not worker.is_alive()

    interaction.cancel()

    assert (
        subsystem,
        "release",
    ) in calls

    release_index = calls.index((
        subsystem,
        "release",
    ))
    prepare_index = calls.index((
        subsystem,
        "prepare",
    ))

    assert release_index > prepare_index


def test_present_and_grip_are_independent():
    calls = []
    states = []
    interaction = sequence(calls, states)

    interaction.present()

    assert ("hand", "prepare") not in calls
    assert ("hand", "grip") not in calls
    assert interaction.arm_presented
    assert not interaction.hand_gripped


def test_idle_interrupt_does_not_contact_hardware():
    calls = []
    states = []
    interaction = sequence(calls, states)

    interaction.interrupt_active()

    assert calls == []
    assert interaction.stop_requested.is_set()
