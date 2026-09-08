"""Launch the complete physical event-interaction ROS stack."""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    """Stop the complete event stack if any required node exits."""

    handshake_configuration = join(
        get_package_share_directory("g1_handshake"),
        "config",
        "handshake.yaml",
    )

    hardware_configuration = join(
        get_package_share_directory("g1_hardware_bridge"),
        "config",
        "hardware.yaml",
    )

    camera_configuration = join(
        get_package_share_directory("g1_hardware_bridge"),
        "config",
        "camera.yaml",
    )

    presence_configuration = join(
        get_package_share_directory("g1_handshake"),
        "config",
        "hand_presence.yaml",
    )

    card_configuration = join(
        get_package_share_directory("g1_card_presentation"),
        "config",
        "card_presentation.yaml",
    )

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

    card_node = Node(
        package="g1_card_presentation",
        executable="card_presentation",
        name="event_card_presentation",
        output="screen",
        parameters=[card_configuration],
    )

    required_nodes = [
        (hardware_node, "hardware bridge exited"),
        (handshake_node, "handshake node exited"),
        (rgb_helper, "RGB helper exited"),
        (camera_bridge, "camera bridge exited"),
        (presence_node, "hand presence node exited"),
        (card_node, "card presentation node exited"),
    ]

    shutdown_handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(reason=reason)
                    )
                ],
            )
        )
        for node, reason in required_nodes
    ]

    return LaunchDescription([
        rgb_helper,
        camera_bridge,
        hardware_node,
        handshake_node,
        presence_node,
        card_node,
        *shutdown_handlers,
    ])