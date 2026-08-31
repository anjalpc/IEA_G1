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

    return LaunchDescription([
        hardware_node,
        handshake_node,
        stop_after_hardware,
        stop_after_handshake,
    ])
