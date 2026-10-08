"""Package contract for workstation-side G1 Nav2 bringup."""

from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = Path(__file__).parents[5]
PACKAGE_NAME = "g1_navigation_bringup"


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


def test_manifest_declares_upstream_nav2_runtime():
    manifest = PACKAGE_ROOT / "package.xml"
    assert manifest.is_file(), str(manifest)

    root = ET.parse(manifest).getroot()
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
        "ament_index_python",
        "launch",
        "launch_ros",
        "nav2_behaviors",
        "nav2_bt_navigator",
        "nav2_controller",
        "nav2_lifecycle_manager",
        "nav2_planner",
        "nav2_smoother",
        "nav2_velocity_smoother",
        "nav2_waypoint_follower",
    } <= runtime_dependencies
    assert {
        "ament_cmake_pytest",
        "python3-pytest",
        "python3-yaml",
    } <= test_dependencies

    build_type = root.find("./export/build_type")
    assert build_type is not None
    assert build_type.text == "ament_cmake"


def test_development_image_uses_exact_nav2_lock():
    dockerfile = (
        REPOSITORY_ROOT
        / "docker/development/Dockerfile"
    ).read_text(encoding="utf-8")
    lock_file = (
        REPOSITORY_ROOT
        / "docker/development/nav2-packages.lock"
    )

    assert lock_file.is_file()
    entries = lock_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(entries) == 35
    assert len(entries) == len(set(entries))
    assert all("=" in entry for entry in entries)
    assert any(
        entry.startswith("ros-humble-navigation2=")
        for entry in entries
    )
    assert any(
        entry.startswith("ros-humble-nav2-bringup=")
        for entry in entries
    )
    assert "COPY nav2-packages.lock" in dockerfile
    assert "/tmp/nav2-packages.lock" in dockerfile


def test_package_contains_no_custom_nav2_implementation():
    forbidden_source_roots = (
        PACKAGE_ROOT / "src",
        PACKAGE_ROOT / "include",
        PACKAGE_ROOT / PACKAGE_NAME,
    )
    for path in forbidden_source_roots:
        assert not path.exists(), str(path)
