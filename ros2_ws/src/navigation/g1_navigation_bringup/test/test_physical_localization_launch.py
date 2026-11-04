"""Contract for read-only physical localization bringup."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch/physical_localization.launch.py"
)


def launch_tree():
    """Parse the physical localization launch."""

    assert LAUNCH_FILE.is_file(), str(LAUNCH_FILE)
    return ast.parse(
        LAUNCH_FILE.read_text(encoding="utf-8")
    )


def launch_constants():
    """Return string constants from the launch."""

    return {
        node.value
        for node in ast.walk(launch_tree())
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def test_composes_authority_perception_and_localization():
    values = launch_constants()

    assert {
        "g1_navigation_bringup",
        "physical_odometry_authority.launch.py",
        "g1_navigation_perception",
        "perception.launch.py",
        "g1_navigation_localization",
        "localization.launch.py",
        "map",
        "use_sim_time",
        "false",
        "autostart",
        "true",
    } <= values


def test_requires_an_explicit_saved_map():
    tree = launch_tree()

    declarations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DeclareLaunchArgument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "map"
    ]

    assert len(declarations) == 1
    assert not any(
        keyword.arg == "default_value"
        for keyword in declarations[0].keywords
    )


def test_starts_no_mapping_navigation_or_commands():
    values = launch_constants()

    forbidden = {
        "mapping.launch.py",
        "slam_toolbox",
        "navigation.launch.py",
        "navigation_control.launch.py",
        "nav2_controller",
        "g1_locomotion_arbiter",
        "cmd_vel",
    }
    assert forbidden.isdisjoint(values)


def test_cmake_registers_physical_localization_contract():
    cmake = (
        PACKAGE_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "test_physical_localization_launch" in cmake
    assert (
        "test/test_physical_localization_launch.py"
        in cmake
    )
