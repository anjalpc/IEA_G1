"""Tests for the automatic-grip image target region."""

import pytest

from g1_handshake.presence.hand_detector import DetectedHand
from g1_handshake.presence.target_region import TargetRegion


def hand(center=(500, 500), bbox=(400, 400, 600, 600)):
    """Construct a detected hand for a 1000 by 1000 frame."""
    return DetectedHand(center[0], center[1], 0.9, bbox=bbox)


def test_accepts_large_hand_centered_in_region():
    region = TargetRegion(0.25, 0.25, 0.75, 0.75, 0.01)
    assert region.accepts(hand(), 1000, 1000)


def test_rejects_hand_outside_region():
    region = TargetRegion(0.25, 0.25, 0.75, 0.75, 0.01)
    assert not region.accepts(hand(center=(900, 500)), 1000, 1000)


def test_rejects_hand_that_is_too_small():
    region = TargetRegion(0.25, 0.25, 0.75, 0.75, 0.01)
    assert not region.accepts(hand(bbox=(490, 490, 510, 510)), 1000, 1000)


def test_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="x bounds"):
        TargetRegion(0.8, 0.0, 0.2, 1.0, 0.01)
