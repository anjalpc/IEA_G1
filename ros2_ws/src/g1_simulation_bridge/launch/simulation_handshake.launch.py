"""Launch the simulation bridge and handshake as one supervised feature."""

import os
import shlex
from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    """Stop the complete simulation feature if either node exits."""
    simulation_config = join(get_package_share_directory(
        "g1_simulation_bridge"), "config", "simulation.yaml")
    handshake_config = join(get_package_share_directory(
        "g1_handshake"), "config", "handshake.yaml")
    simulation = Node(
        package="g1_simulation_bridge", executable="simulation_bridge",
        name="g1_simulation_bridge", output="screen",
        arguments=shlex.split(os.environ.get("G1_SIMULATION_ARGUMENTS", "")),
        parameters=[simulation_config])
    handshake = Node(
        package="g1_handshake", executable="handshake_node",
        name="event_handshake", output="screen",
        parameters=[handshake_config, {"bridge_name": "simulation"}])
    return LaunchDescription([
        simulation, handshake,
        RegisterEventHandler(OnProcessExit(
            target_action=simulation,
            on_exit=[EmitEvent(event=Shutdown(
                reason="simulation bridge exited"))])),
        RegisterEventHandler(OnProcessExit(
            target_action=handshake,
            on_exit=[EmitEvent(event=Shutdown(
                reason="handshake application exited"))])),
    ])
