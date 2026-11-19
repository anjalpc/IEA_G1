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


def physical_visual_description(description):
    """Add package-resolvable full-body visuals without rubber hands."""
    root = ET.fromstring(physical_robot_description(description))
    removed = {
        link.attrib["name"] for link in root.findall("link")
        if link.attrib["name"].endswith("_rubber_hand")
    }
    for link in root.findall("link"):
        if link.attrib["name"] in removed:
            root.remove(link)
    for joint in root.findall("joint"):
        if (joint.find("child").attrib["link"] in removed or
                joint.find("parent").attrib["link"] in removed):
            root.remove(joint)
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if not filename.startswith("meshes/"):
            raise ValueError("Unexpected Unitree mesh path")
        mesh.attrib["filename"] = (
            "package://g1_robot_description/" + filename
        )
    return ET.tostring(root, encoding="unicode") + chr(10)



def attach_static_brainco(description, hand_root):
    """Attach open, fixed visual hands; no measured finger state."""
    root = ET.fromstring(description)
    for side in ("left", "right"):
        hand = ET.parse(hand_root / "urdf" / f"revo2_{side}_hand.urdf").getroot()
        for link in hand.findall("link"):
            link.attrib["name"] = "visual_" + link.attrib["name"]
            for tag in ("collision", "inertial"):
                for element in link.findall(tag):
                    link.remove(element)
        for joint in hand.findall("joint"):
            joint.attrib["name"] = "visual_" + joint.attrib["name"]
            name = joint.attrib["name"]
            if name.endswith(("_thumb_proximal_joint", "_thumb_distal_joint")):
                angle = 0.5 if name.endswith("_proximal_joint") else 0.54725
                origin = joint.find("origin")
                angles = [float(v) for v in origin.attrib["rpy"].split()]
                angles[0] += (1 if side == "right" else -1) * angle
                origin.attrib["rpy"] = " ".join(str(v) for v in angles)
            joint.attrib["type"] = "fixed"
            for tag in ("parent", "child"):
                element = joint.find(tag)
                element.attrib["link"] = "visual_" + element.attrib["link"]
            for element in list(joint):
                if element.tag not in ("origin", "parent", "child"):
                    joint.remove(element)
        for mesh in hand.findall(".//mesh"):
            filename = mesh.attrib["filename"]
            prefix = "package://revo2_description/"
            if not filename.startswith(prefix):
                raise ValueError("Unexpected BrainCo mesh path")
            mesh.attrib["filename"] = (
                "package://g1_robot_description/vendor/brainco_revo2/"
                + filename[len(prefix):]
            )
        root.extend(hand.findall("link") + hand.findall("joint"))
        mount = ET.SubElement(root, "joint",
            name=side + "_brainco_visual_mount", type="fixed")
        ET.SubElement(mount, "parent", link=side + "_wrist_yaw_link")
        ET.SubElement(mount, "child", link="visual_" + side + "_base_link")
        sign = 1 if side == "right" else -1
        ET.SubElement(mount, "origin",
            xyz=f"0.0415 {-sign * 0.003} 0",
            rpy=f"{sign * math.pi / 2} 0 {sign * math.pi / 2}")
    return ET.tostring(root, encoding="unicode") + chr(10)



def generate_launch_description():
    """Launch one publisher for measured physical joint state."""
    share = Path(
        get_package_share_directory(
            "g1_robot_description"
        )
    )
    source_description = (
        share / "vendor/g1_29dof.urdf"
    ).read_text(encoding="utf-8")
    robot_description = physical_visual_description(
        source_description
    )

    robot_description = attach_static_brainco(
        robot_description, share / "vendor/brainco_revo2"
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
