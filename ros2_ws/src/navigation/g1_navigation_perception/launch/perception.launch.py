"""Launch workstation-side navigation perception."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """Launch only upstream sensor-processing components."""
    use_sim_time = LaunchConfiguration(
        "use_sim_time"
    )
    configuration = PathJoinSubstitution([
        FindPackageShare(
            "g1_navigation_perception"
        ),
        "config",
        "lidar_projection.yaml",
    ])

    projection = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            configuration,
            {
                "use_sim_time": use_sim_time,
            },
        ],
        remappings=[
            (
                "cloud_in",
                "/g1/sensors/lidar/points",
            ),
            (
                "scan",
                "/scan",
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description=(
                "Use the ROS simulation clock"
            ),
        ),
        projection,
    ])
