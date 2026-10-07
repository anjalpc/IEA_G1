"""Launch the navigation-relevant G1 transform runtime."""

from pathlib import Path

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return the robot-state runtime launch description."""

    share = Path(
        get_package_share_directory(
            "g1_robot_description"
        )
    )
    robot_description = (
        share / "urdf/g1_navigation.urdf"
    ).read_text(encoding="utf-8")
    use_sim_time = LaunchConfiguration(
        "use_sim_time"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use the ROS simulation clock",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="g1_robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
                "publish_frequency": 50.0,
                "ignore_timestamp": False,
                "frame_prefix": "",
            }],
        ),
    ])
