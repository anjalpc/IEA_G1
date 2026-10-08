"""ROS message construction for validated navigation commands."""

from g1_locomotion_interfaces.msg import LocomotionCommand

from g1_navigation_control.command_gate import (
    NavigationCommand,
)


def build_locomotion_command(*, command, stamp):
    """Build a ROS command from a validated pure command."""
    if not isinstance(command, NavigationCommand):
        raise ValueError(
            "command must be NavigationCommand"
        )

    message = LocomotionCommand()
    message.header.stamp = stamp
    message.owner = command.owner
    message.lease_id = command.lease_id

    message.twist.linear.x = command.linear_x
    message.twist.linear.y = command.linear_y
    message.twist.linear.z = 0.0
    message.twist.angular.x = 0.0
    message.twist.angular.y = 0.0
    message.twist.angular.z = command.angular_z

    return message
