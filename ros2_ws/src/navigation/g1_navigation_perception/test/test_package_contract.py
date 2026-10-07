"""Package contract for workstation navigation perception."""

from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = Path(__file__).parents[5]
PACKAGE_NAME = "g1_navigation_perception"


def test_package_uses_navigation_grouping():
    assert PACKAGE_ROOT.parent.name == "navigation"
    assert PACKAGE_ROOT.name == PACKAGE_NAME


def test_package_has_installable_ament_python_layout():
    required = (
        PACKAGE_ROOT / "package.xml",
        PACKAGE_ROOT / "setup.py",
        PACKAGE_ROOT / "setup.cfg",
        PACKAGE_ROOT / "resource" / PACKAGE_NAME,
        (
            PACKAGE_ROOT
            / PACKAGE_NAME
            / "__init__.py"
        ),
    )

    for path in required:
        assert path.is_file(), str(path)


def test_manifest_declares_upstream_converter_runtime():
    root = ET.parse(
        PACKAGE_ROOT / "package.xml"
    ).getroot()

    dependencies = {
        element.text
        for element in root.findall("exec_depend")
    }

    assert "pointcloud_to_laserscan" in dependencies
    assert "launch" in dependencies
    assert "launch_ros" in dependencies
    assert "sensor_msgs" in dependencies
    assert "tf2_ros" in dependencies


def test_development_image_installs_upstream_converter():
    dockerfile = (
        REPOSITORY_ROOT
        / "docker/development/Dockerfile"
    ).read_text(encoding="utf-8")

    assert (
        "ros-humble-pointcloud-to-laserscan"
        in dockerfile
    )


def test_package_contains_no_custom_projection_algorithm():
    source_root = PACKAGE_ROOT / PACKAGE_NAME
    forbidden = (
        "PointCloud2ConstIterator",
        "laser_geometry",
        "create_subscription(PointCloud2",
    )

    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source
