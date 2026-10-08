"""Launch upstream Map Server and AMCL for the G1."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch lifecycle-managed map-based localization."""
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")

    default_params = join(
        get_package_share_directory(
            "g1_navigation_localization"
        ),
        "config",
        "amcl.yaml",
    )
    common_parameters = [
        params_file,
        {"use_sim_time": use_sim_time},
    ]
    tf_remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "map",
            description=(
                "Full path to the saved occupancy-map YAML"
            ),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Full path to AMCL parameters",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use the ROS simulation clock",
        ),
        DeclareLaunchArgument(
            "autostart",
            default_value="false",
            description=(
                "Configure and activate Map Server and AMCL"
            ),
        ),
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[
                *common_parameters,
                {"yaml_filename": map_file},
            ],
        ),
        Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            output="screen",
            parameters=common_parameters,
            remappings=tf_remappings,
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "autostart": autostart,
                    "node_names": [
                        "map_server",
                        "amcl",
                    ],
                }
            ],
        ),
    ])
