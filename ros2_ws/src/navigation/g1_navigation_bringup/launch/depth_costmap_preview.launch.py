"""Standalone depth costmap preview; no navigation or locomotion nodes."""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = join(get_package_share_directory("g1_navigation_bringup"),
                  "config", "depth_costmap_preview.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),
        Node(
            package="nav2_costmap_2d",
            executable="nav2_costmap_2d",
            name="costmap",
            namespace="costmap",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
            remappings=[
                ("costmap", "/g1/diagnostics/depth/costmap"),
                ("costmap_raw", "/g1/diagnostics/depth/costmap_raw"),
                ("costmap_updates", "/g1/diagnostics/depth/costmap_updates"),
            ],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_depth_costmap_preview",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "autostart": True,
                "node_names": ["/costmap/costmap"],
                # Standalone Costmap2DROS does not create a lifecycle bond.
                "bond_timeout": 0.0,
            }],
        ),
    ])
