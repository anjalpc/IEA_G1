"""Launch the physical bridge and handshake as one supervised stack."""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    """Stop the complete hardware stack if either required node exits."""
    handshake_configuration = join(
        get_package_share_directory("g1_handshake"),
        "config", "handshake.yaml")
    hardware_configuration = join(
        get_package_share_directory("g1_hardware_bridge"),
        "config", "hardware.yaml")
    camera_configuration = join(
        get_package_share_directory("g1_hardware_bridge"),
        "config", "camera.yaml")
    presence_configuration = join(
        get_package_share_directory("g1_handshake"),
        "config", "hand_presence.yaml")

    rgb_helper = Node(
        package="g1_hardware_bridge",
        executable="rgb_helper",
        name="unitree_rgb_helper",
        output="screen",
    )

    camera_bridge = Node(
        package="g1_hardware_bridge",
        executable="camera_bridge",
        name="g1_camera_bridge",
        output="screen",
        parameters=[camera_configuration],
    )

    hardware_node = Node(
        package="g1_hardware_bridge",
        executable="hardware_bridge",
        name="g1_hardware_bridge",
        output="screen",
        parameters=[hardware_configuration],
        sigterm_timeout="35.0",
        sigkill_timeout="5.0",
    )

    handshake_node = Node(
        package="g1_handshake",
        executable="handshake_node",
        name="event_handshake",
        output="screen",
        parameters=[handshake_configuration],
    )

    presence_node = Node(
        package="g1_handshake",
        executable="hand_presence_node",
        name="hand_presence",
        output="screen",
        parameters=[presence_configuration],
    )

    stop_after_hardware = RegisterEventHandler(
        OnProcessExit(
            target_action=hardware_node,
            on_exit=[
                EmitEvent(event=Shutdown(
                    reason="hardware bridge exited"))
            ],
        )
    )

    stop_after_handshake = RegisterEventHandler(
        OnProcessExit(
            target_action=handshake_node,
            on_exit=[
                EmitEvent(event=Shutdown(
                    reason="handshake node exited"))
            ],
        )
    )

    stop_after_rgb = RegisterEventHandler(
        OnProcessExit(
            target_action=rgb_helper,
            on_exit=[EmitEvent(event=Shutdown(reason="RGB helper exited"))],
        )
    )

    stop_after_camera = RegisterEventHandler(
        OnProcessExit(
            target_action=camera_bridge,
            on_exit=[EmitEvent(event=Shutdown(
                reason="camera bridge exited"))],
        )
    )

    stop_after_presence = RegisterEventHandler(
        OnProcessExit(
            target_action=presence_node,
            on_exit=[EmitEvent(event=Shutdown(
                reason="hand presence node exited"))],
        )
    )

    return LaunchDescription([
        rgb_helper,
        camera_bridge,
        hardware_node,
        handshake_node,
        presence_node,
        stop_after_hardware,
        stop_after_handshake,
        stop_after_rgb,
        stop_after_camera,
        stop_after_presence,
    ])
