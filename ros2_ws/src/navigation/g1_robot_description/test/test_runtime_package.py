"""Package contract for the G1 robot-state runtime."""

from pathlib import Path
from xml.etree import ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]


def test_declares_robot_state_runtime_dependencies():
    package = ET.parse(
        PACKAGE_ROOT / "package.xml"
    ).getroot()

    exec_dependencies = {
        element.text.strip()
        for element in package.findall("exec_depend")
    }

    assert {
        "ament_index_python",
        "launch",
        "launch_ros",
        "robot_state_publisher",
    }.issubset(exec_dependencies)
    assert "joint_state_publisher" not in exec_dependencies


def test_installs_launch_directory():
    cmake = (
        PACKAGE_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "DIRECTORY urdf vendor launch" in cmake

def test_registers_runtime_contract_tests():
    cmake = (
        PACKAGE_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "test_runtime_package" in cmake
    assert "test/test_runtime_package.py" in cmake
    assert "test_runtime_launch" in cmake
    assert "test/test_runtime_launch.py" in cmake

