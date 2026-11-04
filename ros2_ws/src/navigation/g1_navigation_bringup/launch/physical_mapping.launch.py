"""Launch read-only physical mapping on the workstation."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)


def _package_launch(package_name, launch_name):
    """Return one package-owned launch inclusion."""

    package_share = get_package_share_directory(
        package_name
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(
                package_share,
                "launch",
                launch_name,
            )
        ),
        launch_arguments={
            "use_sim_time": "false",
        }.items(),
    )


def generate_launch_description():
    """Compose authority, scan projection, and mapping."""

    authority = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(
                get_package_share_directory(
                    "g1_navigation_bringup"
                ),
                "launch",
                "physical_odometry_authority.launch.py",
            )
        )
    )
    perception = _package_launch(
        "g1_navigation_perception",
        "perception.launch.py",
    )
    mapping = _package_launch(
        "g1_navigation_mapping",
        "mapping.launch.py",
    )

    return LaunchDescription([
        authority,
        perception,
        mapping,
    ])
