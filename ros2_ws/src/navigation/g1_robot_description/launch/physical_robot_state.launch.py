"""Launch measured physical G1 kinematic transforms."""

import math
from pathlib import Path
from xml.etree import ElementTree as ET

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node


MID360_JOINT = "mid360_joint"


def invert_mid360_mount(description):
    """Apply the measured local-X inversion to the MID-360 mount."""
    root = ET.fromstring(description)
    joints = [
        joint
        for joint in root.findall("joint")
        if joint.attrib.get("name") == MID360_JOINT
    ]

    if len(joints) != 1:
        raise ValueError(
            "physical description requires exactly one "
            "mid360_joint"
        )

    origin = joints[0].find("origin")
    if origin is None:
        raise ValueError("mid360_joint has no origin")

    values = origin.attrib.get("rpy", "").split()
    if len(values) != 3:
        raise ValueError("mid360_joint has invalid rpy")

    _, pitch, yaw = (
        float(value)
        for value in values
    )
    origin.attrib["rpy"] = (
        f"{math.pi} {pitch} {yaw}"
    )

    return ET.tostring(
        root,
        encoding="unicode",
    ) + "\n"


def physical_robot_description(description):
    """Build the physical mount without claiming GLIM frames."""
    return invert_mid360_mount(description)


def generate_launch_description():
    """Launch one publisher for measured physical joint state."""
    share = Path(
        get_package_share_directory(
            "g1_robot_description"
        )
    )
    source_description = (
        share / "urdf/g1_navigation.urdf"
    ).read_text(encoding="utf-8")
    robot_description = physical_robot_description(
        source_description
    )

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="g1_physical_robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": False,
                "publish_frequency": 50.0,
                "ignore_timestamp": False,
                "frame_prefix": "",
            }],
        ),
    ])
