"""Opt-in depth cloud preview; no navigation or command nodes."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Convert driver-owned depth and calibration to a diagnostic cloud."""
    return LaunchDescription([Node(
        package="depth_image_proc",
        executable="point_cloud_xyz_node",
        namespace="/d435/depth",
        name="g1_depth_cloud_preview",
        output="screen",
        parameters=[{"use_sim_time": False}],
        remappings=[
            ("image_rect", "/d435/depth/image_rect_raw"),
            ("camera_info", "/d435/depth/camera_info"),
            ("points", "/g1/diagnostics/depth/points"),
        ],
    )])
