"""Tests for the lease-aware navigation command gate."""

import math

import pytest

from g1_navigation_control.command_gate import (
    NavigationCommandGate,
)
from g1_robot_interfaces.locomotion import (
    OWNER_NAVIGATION,
    PlanarVelocityLimits,
)


@pytest.fixture
def gate():
    """Return a conservative forward-only command gate."""
    return NavigationCommandGate(
        limits=PlanarVelocityLimits(
            forward_mps=0.15,
            lateral_mps=0.01,
            yaw_rate_rps=0.40,
        ),
        allow_lateral=False,
    )


def test_does_not_create_commands_without_a_lease(gate):
    """Nav2 has no command authority before lease acquisition."""
    assert gate.translate(
        linear_x=0.10,
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.20,
    ) is None


def test_attaches_navigation_owner_and_active_lease(gate):
    """Authorized commands carry navigation ownership."""
    gate.grant("navigation-lease")

    command = gate.translate(
        linear_x=0.10,
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=-0.20,
    )

    assert command.owner == OWNER_NAVIGATION
    assert command.lease_id == "navigation-lease"
    assert command.linear_x == pytest.approx(0.10)
    assert command.linear_y == pytest.approx(0.0)
    assert command.angular_z == pytest.approx(-0.20)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("linear_x", math.inf),
        ("linear_x", 0.16),
        ("linear_y", 0.01),
        ("angular_z", 0.41),
    ),
)
def test_rejects_unsafe_navigation_velocities(
    gate,
    field,
    value,
):
    """Unsafe Nav2 output never becomes a locomotion command."""
    gate.grant("navigation-lease")
    values = {
        "linear_x": 0.10,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angular_x": 0.0,
        "angular_y": 0.0,
        "angular_z": 0.20,
    }
    values[field] = value

    with pytest.raises(ValueError):
        gate.translate(**values)


def test_creates_zero_stop_with_active_lease(gate):
    """Stopping retains authority long enough to send zero."""
    gate.grant("navigation-lease")

    command = gate.stop()

    assert command.owner == OWNER_NAVIGATION
    assert command.lease_id == "navigation-lease"
    assert command.linear_x == 0.0
    assert command.linear_y == 0.0
    assert command.angular_z == 0.0


def test_clearing_lease_removes_command_authority(gate):
    """A cleared session cannot emit commands or stops."""
    gate.grant("navigation-lease")
    gate.clear()

    assert gate.stop() is None
    assert gate.translate(
        linear_x=0.0,
        linear_y=0.0,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    ) is None
