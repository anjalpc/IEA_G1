"""Launch the guarded handshake node with safe defaults."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare deployment settings and start the handshake node."""
    hardware = LaunchConfiguration("hardware_enabled")
    simulation = LaunchConfiguration("simulation_enabled")
    hand_control = LaunchConfiguration("hand_control_enabled")
    interface = LaunchConfiguration("network_interface")
    return LaunchDescription([
        DeclareLaunchArgument("hardware_enabled", default_value="false"),
        DeclareLaunchArgument("simulation_enabled", default_value="false"),
        DeclareLaunchArgument("hand_control_enabled", default_value="false"),
        DeclareLaunchArgument("network_interface", default_value="eth0"),
        Node(
            package="g1_robot_bridge",
            executable="handshake_node",
            name="event_handshake",
            output="screen",
            parameters=[{
                "hardware_enabled": ParameterValue(hardware, value_type=bool),
                "simulation_enabled": ParameterValue(simulation, value_type=bool),
                "hand_control_enabled": ParameterValue(
                    hand_control, value_type=bool),
                "network_interface": interface,
            }],
        ),
    ])
