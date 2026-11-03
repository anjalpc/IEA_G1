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
