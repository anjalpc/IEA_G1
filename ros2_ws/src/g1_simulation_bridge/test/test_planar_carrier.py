"""Tests for the simulation-only planar carrier."""

import math
from pathlib import Path
import sys

import pytest


SIMULATION_ROOT = (
    Path(__file__).parents[4]
    / "simulation"
    / "mujoco"
)
sys.path.insert(0, str(SIMULATION_ROOT))

from navigation.planar_carrier import (  # noqa: E402
    PlanarCarrier,
    PlanarPose,
)


def carrier(**overrides):
    """Return a conservative forward-only carrier."""
    arguments = {
        "command_timeout_seconds": 0.15,
        "max_forward_mps": 0.15,
        "max_lateral_mps": 0.01,
        "max_yaw_rate_rps": 0.40,
        "allow_lateral": False,
    }
    arguments.update(overrides)
    return PlanarCarrier(**arguments)


def test_remains_stationary_without_a_command():
    """Creating the fixture never causes motion."""
    instance = carrier()

    pose = instance.advance(simulation_time=1.0)

    assert pose == PlanarPose(x=0.0, y=0.0, yaw=0.0)
    assert instance.stopped is True


def test_integrates_body_forward_velocity_in_world_frame():
    """Forward follows the carrier's current heading."""
    instance = carrier(
        initial_pose=PlanarPose(
            x=1.0,
            y=2.0,
            yaw=math.pi / 2.0,
        )
    )
    instance.command(
        forward_mps=0.10,
        lateral_mps=0.0,
        yaw_rate_rps=0.0,
        simulation_time=5.0,
    )

    pose = instance.advance(simulation_time=5.10)

    assert pose.x == pytest.approx(1.0, abs=1.0e-12)
    assert pose.y == pytest.approx(2.01)
    assert pose.yaw == pytest.approx(math.pi / 2.0)


def test_integrates_and_normalizes_yaw():
    """Yaw remains in the canonical minus-pi to pi range."""
    instance = carrier(
        initial_pose=PlanarPose(
            x=0.0,
            y=0.0,
            yaw=math.pi - 0.01,
        )
    )
    instance.command(
        forward_mps=0.0,
        lateral_mps=0.0,
        yaw_rate_rps=0.20,
        simulation_time=2.0,
    )

    pose = instance.advance(simulation_time=2.10)

    assert -math.pi <= pose.yaw < math.pi
    assert pose.yaw == pytest.approx(
        -math.pi + 0.01,
        abs=1.0e-12,
    )


def test_timeout_integrates_only_until_deadline():
    """A stale command cannot move for the full late interval."""
    instance = carrier()
    instance.command(
        forward_mps=0.10,
        lateral_mps=0.0,
        yaw_rate_rps=0.0,
        simulation_time=0.0,
    )

    pose = instance.advance(simulation_time=0.30)

    assert pose.x == pytest.approx(0.015)
    assert pose.y == 0.0
    assert instance.stopped is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("forward_mps", math.inf),
        ("forward_mps", 0.16),
        ("lateral_mps", 0.01),
        ("yaw_rate_rps", 0.41),
    ),
)
def test_rejects_unsafe_commands(field, value):
    """The simulation fixture enforces its own limits."""
    instance = carrier()
    values = {
        "forward_mps": 0.10,
        "lateral_mps": 0.0,
        "yaw_rate_rps": 0.20,
        "simulation_time": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError):
        instance.command(**values)


def test_stop_prevents_future_motion():
    """An explicit stop freezes the planar carrier."""
    instance = carrier()
    instance.command(
        forward_mps=0.10,
        lateral_mps=0.0,
        yaw_rate_rps=0.0,
        simulation_time=0.0,
    )
    instance.advance(simulation_time=0.05)
    instance.stop()
    stopped_pose = instance.pose

    assert instance.advance(
        simulation_time=0.10
    ) == stopped_pose
    assert instance.stopped is True


def test_rejects_time_moving_backwards():
    """Simulation resets cannot silently corrupt integration."""
    instance = carrier()
    instance.advance(simulation_time=1.0)

    with pytest.raises(
        ValueError,
        match="simulation time moved backwards",
    ):
        instance.advance(simulation_time=0.5)
