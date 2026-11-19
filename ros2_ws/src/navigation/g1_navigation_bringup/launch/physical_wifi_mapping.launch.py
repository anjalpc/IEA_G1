"""Launch read-only mapping through one Wi-Fi cloud relay."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)


def generate_launch_description():
    """Compose the tested Wi-Fi relay and reliable GLIM mapping."""
    share = get_package_share_directory("g1_navigation_bringup")
    physical = join(share, "config", "physical")
    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(share, "launch", "physical_mapping.launch.py")
        ),
        launch_arguments={
            "glim_configuration": join(physical, "glim_mid360_wifi"),
        }.items(),
    )
    relay = Node(
        package="topic_tools", executable="relay",
        name="g1_wifi_cloud_relay", output="screen",
        arguments=[
            "/g1/transport/lidar/points_wifi",
            "/g1/sensors/lidar/points",
        ],
    )
    return LaunchDescription([
        SetEnvironmentVariable(name="ROS_DOMAIN_ID", value="42"),
        SetEnvironmentVariable(name="ROS_LOCALHOST_ONLY", value="0"),
        SetEnvironmentVariable(
            name="RMW_IMPLEMENTATION", value="rmw_cyclonedds_cpp"
        ),
        SetEnvironmentVariable(
            name="CYCLONEDDS_URI",
            value="file://" + join(physical, "cyclonedds_laptop_wifi.xml"),
        ),
        relay,
        mapping,
    ])
