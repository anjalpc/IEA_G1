"""Contract for complete navigation simulation composition."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch/navigation_simulation.launch.py"
)


def launch_constants():
    """Return string literals from the composition launch."""
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


def test_composes_existing_package_owned_launch_files():
    """The umbrella launch reuses boundaries instead of duplicating nodes."""
    constants = launch_constants()

    assert {
        "g1_simulation_bridge",
        "simulation.launch.py",
        "navigation_simulation.yaml",
        "g1_state_estimation",
        "state_estimation.launch.py",
        "g1_robot_description",
        "robot_state.launch.py",
        "g1_navigation_perception",
        "perception.launch.py",
        "g1_navigation_mapping",
        "mapping.launch.py",
        "g1_navigation_bringup",
        "navigation.launch.py",
        "navigation_control.launch.py",
        "navigation_simulation_control.yaml",
    } <= constants


def test_forces_safe_simulation_runtime_defaults():
    """Simulation composition is headless, timed, and lifecycle managed."""
    constants = launch_constants()

    assert "G1_SIMULATION_ARGUMENTS" in constants
    assert "--headless" in constants
    assert "true" in constants


def test_manifest_declares_composed_runtime_packages():
    """Every included package is an explicit runtime dependency."""
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    dependencies = {
        element.text
        for element in root.findall("exec_depend")
    }

    assert {
        "g1_simulation_bridge",
        "g1_state_estimation",
        "g1_robot_description",
        "g1_navigation_perception",
        "g1_navigation_mapping",
    } <= dependencies

def test_scopes_each_included_launch_configuration():
    """Generic argument names must not leak between includes."""
    tree = ast.parse(
        LAUNCH_FILE.read_text(encoding="utf-8")
    )
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

