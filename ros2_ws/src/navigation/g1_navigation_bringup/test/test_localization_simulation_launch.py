"""Contract for saved-map navigation simulation."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT
    / "launch/localization_simulation.launch.py"
)


def source_tree():
    """Parse the localization simulation launch."""
    assert LAUNCH_FILE.is_file(), str(LAUNCH_FILE)
    return ast.parse(
        LAUNCH_FILE.read_text(encoding="utf-8")
    )


def constants():
    """Return launch string literals."""
    return {
        node.value
        for node in ast.walk(source_tree())
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def test_replaces_mapping_with_saved_map_localization():
    """AMCL owns map-to-odom; SLAM Toolbox is absent."""
    values = constants()

    assert {
        "g1_navigation_localization",
        "localization.launch.py",
        "g1_simulation_bridge",
        "g1_state_estimation",
        "g1_robot_description",
        "g1_navigation_perception",
        "g1_navigation_bringup",
        "navigation.launch.py",
        "navigation_control.launch.py",
    } <= values

    assert "g1_navigation_mapping" not in values
    assert "mapping.launch.py" not in values
    assert "slam_toolbox" not in values


def test_requires_map_and_forwards_it_to_localization():
    """The selected occupancy map is explicit."""
    tree = source_tree()
    source = LAUNCH_FILE.read_text(encoding="utf-8")

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
    assert '"map": map_file' in source


def test_scopes_includes_and_fixes_safe_simulation_defaults():
    """Arguments cannot leak and the stack is headless/autostarted."""
    tree = source_tree()
    values = constants()
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "package_launch"
    )
    returns = [
        node.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Return)
    ]

    assert len(returns) == 1
    assert isinstance(returns[0], ast.Call)
    assert isinstance(returns[0].func, ast.Name)
    assert returns[0].func.id == "GroupAction"
    assert "G1_SIMULATION_ARGUMENTS" in values
    assert "--headless" in values
    assert "true" in values


def test_bringup_declares_localization_runtime_dependency():
    """The included localization package is explicit in the manifest."""
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    dependencies = {
        element.text
        for element in root.findall("exec_depend")
    }

    assert "g1_navigation_localization" in dependencies
