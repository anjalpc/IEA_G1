"""Tests for the hardware-free recording locomotion backend."""

import pytest

from g1_locomotion_control.backend import RecordingBackend


def test_activation_does_not_create_a_motion_command():
    backend = RecordingBackend()

    backend.activate()

    assert backend.active is True
    assert backend.commands == []
    assert backend.events == [("activate",)]


def test_records_only_planar_velocity_values():
    backend = RecordingBackend()
    backend.activate()

    backend.command_velocity(0.2, -0.1, 0.3)

    assert backend.commands == [(0.2, -0.1, 0.3)]
    assert backend.events[-1] == (
        "command_velocity",
        0.2,
        -0.1,
        0.3,
    )


def test_rejects_command_before_activation():
    backend = RecordingBackend()

    with pytest.raises(RuntimeError, match="not active"):
        backend.command_velocity(0.1, 0.0, 0.0)

    assert backend.commands == []


def test_stop_is_safe_and_idempotent():
    backend = RecordingBackend()
    backend.activate()
    backend.command_velocity(0.2, 0.0, 0.0)

    backend.stop()
    backend.stop()

    assert backend.stopped is True
    assert backend.stop_count == 2
    assert backend.commands == [(0.2, 0.0, 0.0)]


def test_health_is_controllable_without_hardware():
    backend = RecordingBackend()

    assert backend.healthy() is True

    backend.set_healthy(False)

    assert backend.healthy() is False


def test_deactivation_requires_no_hardware_and_is_idempotent():
    backend = RecordingBackend()
    backend.activate()
    backend.stop()

    backend.deactivate()
    backend.deactivate()

    assert backend.active is False
    assert backend.events[-1] == ("deactivate",)
