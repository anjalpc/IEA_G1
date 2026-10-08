"""Package contract for G1 map-based localization."""

from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
PACKAGE_NAME = "g1_navigation_localization"


def test_package_uses_navigation_grouping():
    """Localization remains grouped with navigation packages."""
    assert PACKAGE_ROOT.parent.name == "navigation"
    assert PACKAGE_ROOT.name == PACKAGE_NAME


def test_package_has_configuration_only_layout():
    """The package owns configuration, not localization algorithms."""
    required = (
        PACKAGE_ROOT / "CMakeLists.txt",
        PACKAGE_ROOT / "package.xml",
    )
    for path in required:
        assert path.is_file(), str(path)

    assert not (PACKAGE_ROOT / PACKAGE_NAME).exists()
    assert not (PACKAGE_ROOT / "setup.py").exists()
    assert not (PACKAGE_ROOT / "setup.cfg").exists()


def test_manifest_declares_upstream_localization_runtime():
    """Only upstream Map Server and AMCL provide localization."""
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()

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
        "nav2_amcl",
        "nav2_lifecycle_manager",
        "nav2_map_server",
    } <= runtime_dependencies
    assert {
        "ament_cmake_pytest",
        "python3-pytest",
        "python3-yaml",
    } <= test_dependencies

    build_type = root.find("./export/build_type")
    assert build_type is not None
    assert build_type.text == "ament_cmake"


def test_package_contains_no_custom_localization_algorithm():
    """Project code must not reimplement AMCL or Map Server."""
    forbidden_source_roots = (
        PACKAGE_ROOT / "src",
        PACKAGE_ROOT / "include",
        PACKAGE_ROOT / PACKAGE_NAME,
    )
    for path in forbidden_source_roots:
        assert not path.exists(), str(path)
