"""PC2 navigation-only bringup. Source both Foxy overlays first."""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    project = Path(os.environ.get("G1_PC2_PROJECT_ROOT") or Path.home() / "IEA_G1")
    livox = project / "livox_ws" / "config"
    hardware = Path(get_package_share_directory("g1_hardware_bridge")) / "config"
    locomotion = Path(get_package_share_directory("g1_locomotion_control")) / "config"
    profile = Path(os.environ.get("G1_PC2_DEPLOYMENT_PROFILE") or
                   locomotion / "physical_pc2_deployment.yaml")
    dds = livox / "cyclonedds_pc2_wifi.xml"
    mid360 = livox / "mid360_pc2.json"
    state_config = hardware / "hardware.yaml"
    depth_config = hardware / "d435i_depth_wifi.yaml"

    for path in (profile, dds, mid360, state_config, depth_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    return LaunchDescription([
        SetEnvironmentVariable("ROS_DOMAIN_ID", "42"),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "0"),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        SetEnvironmentVariable("CYCLONEDDS_URI", dds.as_uri()),
        RegisterEventHandler(OnProcessExit(on_exit=[
            EmitEvent(event=Shutdown(reason="A PC2 navigation process exited")),
        ])),
        Node(package="g1_hardware_bridge", executable="hardware_state",
             parameters=[str(state_config)], output="screen"),
        Node(package="realsense2_camera", executable="realsense2_camera_node",
             namespace="d435", name="d435", parameters=[str(depth_config)], output="screen"),
        Node(package="livox_ros_driver2", executable="livox_ros_driver2_node",
             parameters=[{"xfer_format": 0, "multi_topic": 0, "data_src": 0,
                          "publish_freq": 10.0, "output_data_type": 0,
                          "frame_id": "mid360_link", "user_config_path": str(mid360)}],
             remappings=[("/livox/lidar", "/g1/transport/lidar/points_wifi"),
                         ("/livox/imu", "/g1/sensors/lidar/imu_raw")], output="screen"),
        Node(package="g1_locomotion_control", executable="g1_physical_locomotion_arbiter",
             parameters=[str(profile)], output="screen"),
    ])
