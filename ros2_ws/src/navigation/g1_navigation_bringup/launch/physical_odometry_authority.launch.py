"""Launch physical odometry with explicit canonical authority."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Wrap staging with one physical odometry authority."""

    authority_parameters = LaunchConfiguration(
        "authority_parameters"
    )
    bringup_share = get_package_share_directory(
        "g1_navigation_bringup"
    )
    state_share = get_package_share_directory(
        "g1_state_estimation"
    )

    staging = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(
                bringup_share,
                "launch",
                "physical_odometry.launch.py",
            )
        )
    )
    authority = Node(
        package="g1_state_estimation",
        executable="physical_authority_node",
        name="g1_physical_odometry_authority",
        output="screen",
        parameters=[authority_parameters],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "authority_parameters",
            default_value=join(
                state_share,
                "config",
                "physical_authority.yaml",
            ),
            description=(
                "Physical odometry authority parameters"
            ),
        ),
        staging,
        authority,
    ])
