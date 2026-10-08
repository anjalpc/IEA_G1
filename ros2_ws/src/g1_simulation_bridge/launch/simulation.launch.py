"""Launch the application-neutral G1 simulation bridge."""

import os
import shlex
from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch and supervise the shared simulation infrastructure."""
    default_simulation_config = join(
        get_package_share_directory("g1_simulation_bridge"),
        "config",
        "simulation.yaml",
    )
    params_file = LaunchConfiguration("params_file")
    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_simulation_config,
        description=(
            "Full path to the simulation bridge parameter file"
        ),
    )
    simulation = Node(
        package="g1_simulation_bridge",
        executable="simulation_bridge",
        name="g1_simulation_bridge",
        output="screen",
        arguments=shlex.split(
            os.environ.get("G1_SIMULATION_ARGUMENTS", "")
        ),
        parameters=[params_file],
    )
    return LaunchDescription([
        declare_params_file,
        simulation,
        RegisterEventHandler(
            OnProcessExit(
                target_action=simulation,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(
                            reason="simulation bridge exited"
                        )
                    )
                ],
            )
        ),
    ])
