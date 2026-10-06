"""Launch the validated state-estimation boundary."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch validated odometry publication."""

    config = join(
        get_package_share_directory(
            "g1_state_estimation"
        ),
        "config",
        "state_estimation.yaml",
    )

    return LaunchDescription([
        Node(
            package="g1_state_estimation",
            executable="state_estimation_node",
            name="g1_state_estimation",
            output="screen",
            parameters=[config],
        ),
    ])
