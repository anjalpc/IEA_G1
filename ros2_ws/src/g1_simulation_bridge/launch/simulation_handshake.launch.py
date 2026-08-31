"""Launch the simulation bridge and handshake as one supervised stack."""

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
    """Stop the complete simulation stack if either required node exits."""
    handshake_configuration = join(
        get_package_share_directory("g1_handshake"),
        "config", "handshake.yaml")
    simulation_configuration = join(
        get_package_share_directory("g1_simulation_bridge"),
        "config", "simulation.yaml")

    simulation_arguments = shlex.split(
        os.environ.get("G1_SIMULATION_ARGUMENTS", ""))

    simulation_node = Node(
        package="g1_simulation_bridge",
        executable="simulation_bridge",
        name="g1_simulation_bridge",
        output="screen",
        arguments=simulation_arguments,
        parameters=[simulation_configuration],
    )

    handshake_node = Node(
        package="g1_handshake",
        executable="handshake_node",
        name="event_handshake",
        output="screen",
        parameters=[
            handshake_configuration,
            {"bridge_name": "simulation"},
        ],
    )

    stop_after_simulation = RegisterEventHandler(
        OnProcessExit(
            target_action=simulation_node,
            on_exit=[
                EmitEvent(event=Shutdown(
                    reason="simulation bridge exited"))
            ],
        )
    )

    stop_after_handshake = RegisterEventHandler(
        OnProcessExit(
            target_action=handshake_node,
            on_exit=[
                EmitEvent(event=Shutdown(
                    reason="handshake node exited"))
            ],
        )
    )

    return LaunchDescription([
        simulation_node,
        handshake_node,
        stop_after_simulation,
        stop_after_handshake,
    ])
