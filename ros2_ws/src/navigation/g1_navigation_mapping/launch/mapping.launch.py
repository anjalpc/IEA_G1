"""Launch asynchronous workstation-side G1 mapping."""

from os import path

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch upstream SLAM Toolbox with the G1 mapping contract."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_params_file = LaunchConfiguration("slam_params_file")

    default_parameters = path.join(
        get_package_share_directory(
            "g1_navigation_mapping"
        ),
        "config",
        "online_async.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use the ROS simulation clock",
        ),
        DeclareLaunchArgument(
            "slam_params_file",
            default_value=default_parameters,
            description=(
                "Full path to the SLAM Toolbox parameter file"
            ),
        ),
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[
                slam_params_file,
                {"use_sim_time": use_sim_time},
            ],
        ),
    ])
