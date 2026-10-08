"""Package contract for workstation-side G1 mapping."""

from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = Path(__file__).parents[5]
PACKAGE_NAME = "g1_navigation_mapping"


def test_package_uses_navigation_grouping():
    assert PACKAGE_ROOT.parent.name == "navigation"
    assert PACKAGE_ROOT.name == PACKAGE_NAME


def test_package_has_configuration_only_ament_cmake_layout():
    required = (
        PACKAGE_ROOT / "CMakeLists.txt",
        PACKAGE_ROOT / "package.xml",
    )

    for path in required:
        assert path.is_file(), str(path)

    assert not (PACKAGE_ROOT / PACKAGE_NAME).exists()
    assert not (PACKAGE_ROOT / "setup.py").exists()
    assert not (PACKAGE_ROOT / "setup.cfg").exists()


def test_manifest_declares_upstream_mapping_runtime():
    root = ET.parse(
        PACKAGE_ROOT / "package.xml"
    ).getroot()

    build_tools = {
        element.text
        for element in root.findall("buildtool_depend")
    }
    runtime_dependencies = {
        element.text
        for element in root.findall("exec_depend")
    }
    test_dependencies = {
        element.text
        for element in root.findall("test_depend")
    }

    assert "ament_cmake" in build_tools
    assert {
        "launch",
        "launch_ros",
        "slam_toolbox",
    } <= runtime_dependencies
    assert {
        "ament_cmake_pytest",
        "python3-pytest",
    } <= test_dependencies

    build_type = root.find("./export/build_type")
    assert build_type is not None
    assert build_type.text == "ament_cmake"


def test_development_image_installs_upstream_slam_toolbox():
    dockerfile = (
        REPOSITORY_ROOT
        / "docker/development/Dockerfile"
    ).read_text(encoding="utf-8")

    assert "ros-humble-slam-toolbox" in dockerfile


def test_package_contains_no_custom_slam_implementation():
    forbidden_source_roots = (
        PACKAGE_ROOT / "src",
        PACKAGE_ROOT / "include",
        PACKAGE_ROOT / PACKAGE_NAME,
    )

    for path in forbidden_source_roots:
        assert not path.exists(), str(path)
