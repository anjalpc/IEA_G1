"""Launch the handshake application against a selected bridge."""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Start the handshake node with the shared feature configuration."""
    configuration = join(
        get_package_share_directory("g1_handshake"),
        "config", "handshake.yaml")
    return LaunchDescription([
        Node(
            package="g1_handshake",
            executable="handshake_node",
            name="event_handshake",
            output="screen",
            parameters=[configuration, {"bridge_name": "simulation"}],
        ),
    ])
