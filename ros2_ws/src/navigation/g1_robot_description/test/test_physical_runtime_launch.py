"""Physical robot-state launch contract."""

import importlib.util
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_PATH = (
    PACKAGE_ROOT
    / "launch"
    / "physical_robot_state.launch.py"
)


def load_launch_module():
    """Load the physical launch module from source."""
    assert LAUNCH_PATH.is_file(), str(LAUNCH_PATH)
    spec = importlib.util.spec_from_file_location(
        "g1_physical_robot_state_launch",
        LAUNCH_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def description(mid360_count=1):
    """Return a small description with the requested MID-360 joints."""
    joints = """
  <joint name="mid360_joint" type="fixed">
    <origin xyz="1 2 3" rpy="0 0.04014257279586953 0"/>
    <parent link="torso_link"/>
    <child link="mid360_link"/>
  </joint>
""" * mid360_count
    return (
        '<robot name="fixture">'
        '<link name="torso_link"/>'
        '<link name="mid360_link"/>'
        f"{joints}"
        "</robot>"
    )


def test_inverts_only_the_physical_mid360_mount():
    """Physical description applies the measured local-X inversion."""
    module = load_launch_module()

    result = module.physical_robot_description(
        description()
    )
    root = ET.fromstring(result)
    origin = root.find(
        "./joint[@name='mid360_joint']/origin"
    )

    assert origin is not None
    assert origin.attrib["xyz"] == "1 2 3"

    roll, pitch, yaw = (
        float(value)
        for value in origin.attrib["rpy"].split()
    )
    assert roll == pytest.approx(math.pi)
    assert pitch == pytest.approx(
        0.04014257279586953
    )
    assert yaw == pytest.approx(0.0)

    assert root.find(
        "./link[@name='livox_frame']"
    ) is None
    assert root.find(
        "./joint[@name='mid360_imu_joint']"
    ) is None


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_missing_or_duplicate_mid360_mount(count):
    """Never silently publish an ambiguous physical sensor TF."""
    module = load_launch_module()

    with pytest.raises(ValueError):
        module.invert_mid360_mount(
            description(mid360_count=count)
        )

def test_full_physical_visual_description_preserves_navigation():
    """Full visuals retain the measured navigation chain."""
    module = load_launch_module()
    source = (PACKAGE_ROOT / "vendor/g1_29dof.urdf").read_text()
    root = ET.fromstring(module.physical_visual_description(source))
    joints = {j.attrib["name"]: j for j in root.findall("joint")}
    assert sum(j.attrib["type"] == "revolute" for j in joints.values()) == 29
    assert not any("rubber_hand" in l.attrib["name"] for l in root.findall("link"))
    meshes = root.findall(".//visual/geometry/mesh")
    assert len(meshes) >= 30
    assert all(m.attrib["filename"].startswith(
        "package://g1_robot_description/meshes/"
    ) for m in meshes)
    navigation = (PACKAGE_ROOT / "urdf/g1_navigation.urdf").read_text()
    baseline = ET.fromstring(module.physical_robot_description(navigation))
    for expected in baseline.findall("joint"):
        actual = joints[expected.attrib["name"]]
        assert actual.attrib == expected.attrib
        for element in expected:
            assert actual.find(element.tag).attrib == element.attrib


def test_static_brainco_visuals_follow_wrist_frames():
    """Static hand visuals must not claim measured finger motion."""
    module = load_launch_module()
    hand_root = PACKAGE_ROOT.parents[3] / "simulation/mujoco/vendor/brainco_revo2"
    source = (PACKAGE_ROOT / "vendor/g1_29dof.urdf").read_text()
    body = module.physical_visual_description(source)
    root = ET.fromstring(module.attach_static_brainco(body, hand_root))
    joints = {j.attrib["name"]: j for j in root.findall("joint")}
    assert sum(j.attrib["type"] == "revolute" for j in joints.values()) == 29
    for side in ("left", "right"):
        mount = joints[side + "_brainco_visual_mount"]
        assert mount.attrib["type"] == "fixed"
        assert mount.find("parent").attrib["link"] == side + "_wrist_yaw_link"
        assert mount.find("child").attrib["link"] == "visual_" + side + "_base_link"
        sign = 1 if side == "right" else -1
        for segment, angle in (("proximal", 0.83153), ("distal", 0.44253)):
            thumb = joints[f"visual_{side}_thumb_{segment}_joint"]
            roll = float(thumb.find("origin").attrib["rpy"].split()[0])
            assert roll == pytest.approx(sign * angle)
    fingers = [j for name, j in joints.items() if name.startswith("visual_")]
    assert fingers and all(j.attrib["type"] == "fixed" for j in fingers)
