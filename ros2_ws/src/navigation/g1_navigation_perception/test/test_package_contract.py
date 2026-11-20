"""Package contract for workstation navigation perception."""

import ast
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
    )

    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name != "depth_obstacle_preview.py":
            assert "create_subscription(PointCloud2" not in source
        for token in forbidden:
            assert token not in source


def test_depth_diagnostic_node_has_only_approved_endpoints():
    path = PACKAGE_ROOT / PACKAGE_NAME / "depth_obstacle_preview.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    expected = {
        "create_publisher": {
            "/g1/diagnostics/depth/obstacles", "/g1/diagnostics/depth/status",
            "/g1/diagnostics/depth/health"},
        "create_subscription": {
            "/g1/diagnostics/depth/points", "/g1/state/imu_raw"},
    }
    for method, topics in expected.items():
        actual = set()
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != method:
                continue
            assert len(call.args) >= 2
            assert isinstance(call.args[1], ast.Constant)
            actual.add(call.args[1].value)
        assert actual == topics
