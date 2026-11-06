"""Launch planning-only physical navigation."""

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
    """Compose localization and planning without control."""

    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")

    default_params = join(
        get_package_share_directory(
            "g1_navigation_bringup"
        ),
        "config",
        "navigation.yaml",
    )

    localization = IncludeLaunchDescription(
        _launch_file(
            "g1_navigation_bringup",
            "physical_localization.launch.py",
        ),
        launch_arguments={
            "map": map_file,
        }.items(),
    )

    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": False},
        ],
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_planning",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "autostart": True,
                "node_names": ["planner_server"],
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "map",
            description=(
                "Full path to the saved physical occupancy-map YAML"
            ),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Full path to Nav2 planner parameters",
        ),
        localization,
        planner,
        lifecycle_manager,
    ])
