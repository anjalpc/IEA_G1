"""Launch the recording-safe navigation command boundary."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Compose the arbiter and Nav2 command adapter."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")

    default_params = join(
        get_package_share_directory(
            "g1_navigation_bringup"
        ),
        "config",
        "navigation_control.yaml",
    )
    common_parameters = [
        params_file,
        {"use_sim_time": use_sim_time},
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use the ROS simulation clock",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description=(
                "Full path to navigation control parameters"
            ),
        ),
        Node(
            package="g1_locomotion_control",
            executable="g1_locomotion_arbiter",
            name="g1_locomotion_arbiter",
            output="screen",
            parameters=common_parameters,
        ),
        Node(
            package="g1_navigation_control",
            executable="navigation_command_adapter",
            name="g1_navigation_command_adapter",
            output="screen",
            parameters=common_parameters,
        ),
    ])
