"""Installed-resource contract for g1_robot_description."""

from pathlib import Path

from ament_index_python.packages import (
    get_package_share_directory,
)


def test_installs_description_and_provenance():
    share = Path(
        get_package_share_directory(
            "g1_robot_description"
        )
    )

    required = (
        share / "urdf/g1_navigation.urdf",
        share / "launch/robot_state.launch.py",
        share / "vendor/UNITREE_SOURCE.md",
        share / "vendor/LICENSE",
    )

    for path in required:
        assert path.is_file(), (
            f"installed resource is missing: {path}"
        )
        assert path.stat().st_size > 0


def test_installs_full_body_meshes():
    """Installed physical visuals resolve every required mesh."""
    from xml.etree import ElementTree as ET
    share = Path(get_package_share_directory("g1_robot_description"))
    root = ET.parse(share / "vendor/g1_29dof.urdf").getroot()
    paths = {
        share / mesh.attrib["filename"]
        for link in root.findall("link")
        if not link.attrib["name"].endswith("_rubber_hand")
        for mesh in link.findall(".//mesh")
    }
    assert len(paths) == 34
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_installs_brainco_visual_assets():
    """Both static hand descriptions resolve their installed meshes."""
    from xml.etree import ElementTree as ET
    share = Path(get_package_share_directory("g1_robot_description"))
    hand_root = share / "vendor/brainco_revo2"
    assert (hand_root / "LICENSE").is_file()
    for side in ("left", "right"):
        root = ET.parse(hand_root / "urdf" / f"revo2_{side}_hand.urdf").getroot()
        for mesh in root.findall(".//mesh"):
            relative = mesh.attrib["filename"].split("package://revo2_description/", 1)[1]
            assert (hand_root / relative).is_file()
