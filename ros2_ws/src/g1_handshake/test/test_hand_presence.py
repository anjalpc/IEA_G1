import pytest

from g1_handshake.presence.hand_presence import HandPresenceDetector


def test_rejects_invalid_construction():
    with pytest.raises(ValueError, match="required_stable_frames"):
        HandPresenceDetector(0)


def test_single_hit_does_not_fire():
    d = HandPresenceDetector(3)
    assert d.observe(True) is False
    assert d.observe(True) is False


def test_fires_after_required_consecutive_hits():
    d = HandPresenceDetector(3)
    assert d.observe(True) is False
    assert d.observe(True) is False
    assert d.observe(True) is True


def test_losing_the_hit_resets_the_counter():
    d = HandPresenceDetector(3)
    assert d.observe(True) is False
    assert d.observe(True) is False
    assert d.observe(False) is False
    assert d.observe(True) is False
    assert d.observe(True) is False
    assert d.observe(True) is True


def test_does_not_repeat_fire_on_continued_presence():
    d = HandPresenceDetector(2)
    assert d.observe(True) is False
    assert d.observe(True) is True
    # still present, but the counter was reset by firing
    assert d.observe(True) is False
    assert d.observe(True) is True


def test_reset_clears_accumulated_stability():
    d = HandPresenceDetector(2)
    assert d.observe(True) is False
    d.reset()
    assert d.observe(True) is False
    assert d.observe(True) is True


def test_single_frame_detector_fires_immediately():
    d = HandPresenceDetector(1)
    assert d.observe(True) is True
    assert d.observe(False) is False
    assert d.observe(True) is True
