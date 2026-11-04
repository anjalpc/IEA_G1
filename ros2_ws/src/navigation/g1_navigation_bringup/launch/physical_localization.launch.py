"""Launch read-only physical localization on the workstation."""

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


def _launch_file(package_name, launch_name):
    """Resolve one package-owned launch file."""

    return PythonLaunchDescriptionSource(
        join(
            get_package_share_directory(package_name),
            "launch",
            launch_name,
        )
    )


def generate_launch_description():
    """Compose authority, scan projection, and localization."""

    map_file = LaunchConfiguration("map")

    authority = IncludeLaunchDescription(
        _launch_file(
            "g1_navigation_bringup",
            "physical_odometry_authority.launch.py",
        )
    )
    perception = IncludeLaunchDescription(
        _launch_file(
            "g1_navigation_perception",
            "perception.launch.py",
        ),
        launch_arguments={
            "use_sim_time": "false",
        }.items(),
    )
    localization = IncludeLaunchDescription(
        _launch_file(
            "g1_navigation_localization",
            "localization.launch.py",
        ),
        launch_arguments={
            "map": map_file,
            "use_sim_time": "false",
            "autostart": "true",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "map",
            description=(
                "Full path to the saved physical occupancy-map YAML"
            ),
        ),
        authority,
        perception,
        localization,
    ])
