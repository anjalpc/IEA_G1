"""Contract tests for the navigation command adapter package."""

from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]


def test_package_has_installable_ament_python_layout():
    """The adapter is an independently installable ROS package."""
    expected = (
        "package.xml",
        "setup.py",
        "setup.cfg",
        "resource/g1_navigation_control",
        "g1_navigation_control/__init__.py",
    )

    for relative_path in expected:
        assert (PACKAGE_ROOT / relative_path).is_file()


def test_manifest_declares_only_public_runtime_dependencies():
    """The adapter crosses the boundary through public contracts."""
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    dependencies = {
        element.text
        for tag in ("depend", "exec_depend")
        for element in root.findall(tag)
    }

    assert {
        "geometry_msgs",
        "g1_locomotion_interfaces",
        "g1_robot_interfaces",
        "rclpy",
    } <= dependencies

    assert "g1_locomotion_control" not in dependencies


def test_package_exposes_navigation_command_adapter_entry_point():
    """ROS can start the adapter through one stable executable."""
    setup = (
        PACKAGE_ROOT / "setup.py"
    ).read_text(encoding="utf-8")
    compact = "".join(setup.split())

    assert (
        "navigation_command_adapter="
        "g1_navigation_control.node:main"
    ) in compact
