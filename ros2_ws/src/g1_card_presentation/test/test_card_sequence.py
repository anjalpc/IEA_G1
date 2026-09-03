"""Tests for the independent left-card behavior."""

from g1_card_presentation.card_sequence import CardPresentation


class Recorder:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def prepare_operation(self):
        self.calls.append((self.name, "prepare"))

    def prepare_presentation(self):
        self.calls.append((self.name, "prepare"))

    def grip(self):
        self.calls.append((self.name, "grip"))

    def present(self, pose, fraction, move_seconds):
        self.calls.append((self.name, "present", pose, fraction, move_seconds))
        return {"mode": "simulation"}

    def release_presentation(self):
        self.calls.append((self.name, "release"))

    def release(self):
        self.calls.append((self.name, "release"))

    def interrupt(self):
        self.calls.append((self.name, "interrupt"))


def test_card_sequence_is_separate_left_behavior():
    calls = []
    sequence = CardPresentation(
        Recorder("arm", calls), Recorder("hand", calls),
        "left_card_presentation", 0.25, 5.0)
    sequence.prepare()
    assert sequence.present() == {"mode": "simulation"}
    sequence.release()
    assert calls == [
        ("arm", "prepare"),
        ("hand", "prepare"),
        ("hand", "grip"),
        ("arm", "present", "left_card_presentation", 0.25, 5.0),
        ("hand", "release"),
        ("arm", "release"),
    ]
