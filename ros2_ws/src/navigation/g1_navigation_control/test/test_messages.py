"""Tests for navigation locomotion-message construction."""

from builtin_interfaces.msg import Time

from g1_locomotion_interfaces.msg import LocomotionCommand

from g1_navigation_control.command_gate import (
    NavigationCommand,
)
from g1_navigation_control.messages import (
    build_locomotion_command,
)
from g1_robot_interfaces.locomotion import OWNER_NAVIGATION


def test_builds_leased_planar_locomotion_command():
    """A validated navigation command maps without widening it."""
    stamp = Time(sec=12, nanosec=345)
    command = NavigationCommand(
        owner=OWNER_NAVIGATION,
        lease_id="navigation-lease",
        linear_x=0.10,
        linear_y=0.0,
        angular_z=-0.20,
    )

    message = build_locomotion_command(
        command=command,
        stamp=stamp,
    )

    assert isinstance(message, LocomotionCommand)
    assert message.header.stamp == stamp
    assert message.owner == message.OWNER_NAVIGATION
    assert message.lease_id == "navigation-lease"
    assert message.twist.linear.x == 0.10
    assert message.twist.linear.y == 0.0
    assert message.twist.linear.z == 0.0
    assert message.twist.angular.x == 0.0
    assert message.twist.angular.y == 0.0
    assert message.twist.angular.z == -0.20


def test_rejects_unvalidated_command_objects():
    """Only the pure command gate may feed the ROS boundary."""
    try:
        build_locomotion_command(
            command=object(),
            stamp=Time(),
        )
    except ValueError as error:
        assert str(error) == (
            "command must be NavigationCommand"
        )
    else:
        raise AssertionError("unvalidated command was accepted")
