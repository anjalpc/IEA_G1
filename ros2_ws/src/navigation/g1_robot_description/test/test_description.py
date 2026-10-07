"""Contract for the navigation-only G1 description."""

from pathlib import Path
from xml.etree import ElementTree as ET


URDF_PATH = (
    Path(__file__).parents[1]
    / "urdf"
    / "g1_navigation.urdf"
)

EXPECTED_JOINTS = {
    "waist_yaw_joint": {
        "type": "revolute",
        "parent": "pelvis",
        "child": "waist_yaw_link",
        "xyz": "0 0 0",
        "rpy": "0 0 0",
        "axis": "0 0 1",
        "lower": "-2.618",
        "upper": "2.618",
        "effort": "88",
        "velocity": "32",
    },
    "waist_roll_joint": {
        "type": "revolute",
        "parent": "waist_yaw_link",
        "child": "waist_roll_link",
        "xyz": "-0.0039635 0 0.035",
        "rpy": "0 0 0",
        "axis": "1 0 0",
        "lower": "-0.52",
        "upper": "0.52",
        "effort": "35",
        "velocity": "30",
    },
    "waist_pitch_joint": {
        "type": "revolute",
        "parent": "waist_roll_link",
        "child": "torso_link",
        "xyz": "0 0 0.019",
        "rpy": "0 0 0",
        "axis": "0 1 0",
        "lower": "-0.52",
        "upper": "0.52",
        "effort": "35",
        "velocity": "30",
    },
    "head_joint": {
        "type": "fixed",
        "parent": "torso_link",
        "child": "head_link",
        "xyz": "0.0039635 0 -0.054",
        "rpy": "0 0 0",
    },
    "imu_in_torso_joint": {
        "type": "fixed",
        "parent": "torso_link",
        "child": "imu_in_torso",
        "xyz": "-0.03959 -0.00224 0.13792",
        "rpy": "0 0 0",
    },
    "imu_in_pelvis_joint": {
        "type": "fixed",
        "parent": "pelvis",
        "child": "imu_in_pelvis",
        "xyz": "0.04525 0 -0.08339",
        "rpy": "0 0 0",
    },
    "d435_joint": {
        "type": "fixed",
        "parent": "torso_link",
        "child": "d435_link",
        "xyz": "0.0576235 0.01753 0.41987",
        "rpy": "0 0.8307767239493009 0",
    },
}

EXPECTED_LINKS = {
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_link",
    "imu_in_torso",
    "imu_in_pelvis",
    "d435_link",
}


def required_child(element, tag):
    child = element.find(tag)
    assert child is not None, (
        f"{element.attrib['name']} is missing {tag}"
    )
    return child


def test_navigation_description_contract():
    assert URDF_PATH.is_file(), (
        f"navigation URDF is missing: {URDF_PATH}"
    )

    robot = ET.parse(URDF_PATH).getroot()
    assert robot.tag == "robot"
    assert robot.attrib["name"] == "g1_navigation"

    links = {
        link.attrib["name"]: link
        for link in robot.findall("link")
    }
    joints = {
        joint.attrib["name"]: joint
        for joint in robot.findall("joint")
    }

    assert set(links) == EXPECTED_LINKS
    assert set(joints) == set(EXPECTED_JOINTS)

    children = []
    adjacency = {}
    for name, joint in joints.items():
        parent = required_child(
            joint,
            "parent",
        ).attrib["link"]
        child = required_child(
            joint,
            "child",
        ).attrib["link"]
        children.append(child)
        adjacency.setdefault(parent, set()).add(child)

        expected = EXPECTED_JOINTS[name]
        assert joint.attrib["type"] == expected["type"]
        assert parent == expected["parent"]
        assert child == expected["child"]

        origin = required_child(joint, "origin")
        assert origin.attrib["xyz"] == expected["xyz"]
        assert origin.attrib["rpy"] == expected["rpy"]

        if expected["type"] == "revolute":
            axis = required_child(joint, "axis")
            limit = required_child(joint, "limit")
            assert axis.attrib["xyz"] == expected["axis"]
            for field in (
                "lower",
                "upper",
                "effort",
                "velocity",
            ):
                assert (
                    limit.attrib[field]
                    == expected[field]
                )

    assert len(children) == len(set(children))
    assert set(links) - set(children) == {"pelvis"}

    reachable = {"pelvis"}
    pending = ["pelvis"]
    while pending:
        parent = pending.pop()
        for child in adjacency.get(parent, set()):
            if child not in reachable:
                reachable.add(child)
                pending.append(child)
    assert reachable == EXPECTED_LINKS

    forbidden_frames = {
        "world",
        "map",
        "odom",
        "base_footprint",
    }
    assert forbidden_frames.isdisjoint(links)

    assert not any(
        "rubber_hand" in name
        for name in links
    )

    forbidden_elements = (
        "visual",
        "collision",
        "inertial",
        "geometry",
        "mesh",
    )
    for tag in forbidden_elements:
        assert robot.find(f".//{tag}") is None
