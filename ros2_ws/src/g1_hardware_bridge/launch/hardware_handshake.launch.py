"""Launch the physical bridge and handshake app from one configuration."""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Start both production nodes with the installed feature profile."""
    configuration = join(
        get_package_share_directory("g1_handshake"),
        "config", "handshake.yaml")
    return LaunchDescription([
        Node(
            package="g1_hardware_bridge",
            executable="hardware_bridge",
            name="g1_hardware_bridge",
            output="screen",
            parameters=[configuration],
        ),
        Node(
            package="g1_handshake",
            executable="handshake_node",
            name="event_handshake",
            output="screen",
            parameters=[configuration],
        ),
    ])
