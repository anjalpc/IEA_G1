"""Contract for planning-only physical navigation."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch/physical_planning.launch.py"
)


def launch_tree():
    """Parse the physical planning launch."""

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


def test_composes_localization_and_planner_only():
    values = launch_constants()

    assert {
        "g1_navigation_bringup",
        "physical_localization.launch.py",
        "nav2_planner",
        "planner_server",
        "nav2_lifecycle_manager",
        "lifecycle_manager_planning",
        "map",
        "params_file",
        "use_sim_time",
        "autostart",
        "node_names",
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


def test_starts_no_controller_or_command_authority():
    values = launch_constants()

    forbidden = {
        "nav2_controller",
        "controller_server",
        "nav2_behaviors",
        "behavior_server",
        "nav2_bt_navigator",
        "bt_navigator",
        "nav2_velocity_smoother",
        "velocity_smoother",
        "navigation_control.launch.py",
        "g1_locomotion_arbiter",
        "cmd_vel",
    }
    assert forbidden.isdisjoint(values)


def test_passes_boolean_lifecycle_parameters():
    source = LAUNCH_FILE.read_text(encoding="utf-8")

    assert '{"use_sim_time": False}' in source
    assert '"autostart": True' in source
    assert '"use_sim_time": "false"' not in source
    assert '"autostart": "true"' not in source


def test_cmake_registers_physical_planning_contract():
    cmake = (
        PACKAGE_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "test_physical_planning_launch" in cmake
    assert (
        "test/test_physical_planning_launch.py"
        in cmake
    )
