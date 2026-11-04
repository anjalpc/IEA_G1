"""Contract for read-only physical mapping bringup."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch/physical_mapping.launch.py"
)


def launch_constants():
    """Return string constants from physical mapping launch."""

    assert LAUNCH_FILE.is_file(), str(LAUNCH_FILE)
    tree = ast.parse(
        LAUNCH_FILE.read_text(encoding="utf-8")
    )
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def test_composes_authority_perception_and_mapping():
    values = launch_constants()

    assert {
        "g1_navigation_bringup",
        "physical_odometry_authority.launch.py",
        "g1_navigation_perception",
        "perception.launch.py",
        "g1_navigation_mapping",
        "mapping.launch.py",
    } <= values


def test_uses_wall_clock_for_physical_mapping():
    values = launch_constants()

    assert "use_sim_time" in values
    assert "false" in values


def test_starts_no_localization_navigation_or_commands():
    values = launch_constants()

    forbidden = {
        "localization.launch.py",
        "navigation.launch.py",
        "navigation_control.launch.py",
        "amcl",
        "nav2_controller",
        "g1_locomotion_arbiter",
        "cmd_vel",
    }
    assert forbidden.isdisjoint(values)


def test_cmake_registers_physical_mapping_contract():
    cmake = (
        PACKAGE_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "test_physical_mapping_launch" in cmake
    assert "test/test_physical_mapping_launch.py" in cmake
