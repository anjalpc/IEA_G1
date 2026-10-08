"""Contract for safe Nav2-to-locomotion composition."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
CONFIGURATION = PACKAGE_ROOT / "config/navigation_control.yaml"
SIMULATION_CONFIGURATION = (
    PACKAGE_ROOT / "config/navigation_simulation_control.yaml"
)
LAUNCH_FILE = PACKAGE_ROOT / "launch/navigation_control.launch.py"


def load_configuration():
    """Load the shared adapter and arbiter parameters."""
    assert CONFIGURATION.is_file(), str(CONFIGURATION)
    return yaml.safe_load(
        CONFIGURATION.read_text(encoding="utf-8")
    )


def launch_constants():
    """Return literal strings from the control launch file."""
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


def parameters(configuration, node_name):
    """Return one node's ROS parameter dictionary."""
    return configuration[node_name]["ros__parameters"]


def test_contains_control_configuration_and_launch():
    """Bringup owns the runtime composition artifacts."""
    assert CONFIGURATION.is_file()
    assert LAUNCH_FILE.is_file()


def test_uses_recording_backend_only():
    """The first integrated path cannot reach Sport hardware."""
    configuration = load_configuration()
    arbiter = parameters(
        configuration,
        "g1_locomotion_arbiter",
    )

    assert arbiter["backend"] == "recording"


def test_adapter_and_arbiter_share_conservative_limits():
    """Both sides enforce the same initial velocity envelope."""
    configuration = load_configuration()
    arbiter = parameters(
        configuration,
        "g1_locomotion_arbiter",
    )
    adapter = parameters(
        configuration,
        "g1_navigation_command_adapter",
    )

    for name in (
        "max_forward_mps",
        "max_lateral_mps",
        "max_yaw_rate_rps",
    ):
        assert adapter[name] == arbiter[name]

    assert adapter["max_forward_mps"] == 0.15
    assert adapter["max_yaw_rate_rps"] == 0.40
    assert adapter["allow_lateral"] is False

    assert adapter["command_timeout_seconds"] == 0.15
    assert arbiter["command_timeout_seconds"] == 0.25
    assert (
        adapter["command_timeout_seconds"]
        < arbiter["command_timeout_seconds"]
    )


def test_launch_composes_only_public_runtime_nodes():
    """Bringup starts both packages through executables."""
    constants = launch_constants()

    assert {
        "g1_navigation_bringup",
        "navigation_control.yaml",
        "g1_locomotion_control",
        "g1_locomotion_arbiter",
        "g1_navigation_control",
        "navigation_command_adapter",
    } <= constants


def test_manifest_declares_control_runtime_dependencies():
    """Bringup explicitly owns both runtime dependencies."""
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    dependencies = {
        element.text
        for element in root.findall("exec_depend")
    }

    assert "g1_locomotion_control" in dependencies
    assert "g1_navigation_control" in dependencies

def test_simulation_control_uses_simulation_backend():
    """Navigation simulation explicitly selects its transport backend."""
    assert SIMULATION_CONFIGURATION.is_file()
    configuration = yaml.safe_load(
        SIMULATION_CONFIGURATION.read_text(encoding="utf-8")
    )
    arbiter = parameters(
        configuration,
        "g1_locomotion_arbiter",
    )
    adapter = parameters(
        configuration,
        "g1_navigation_command_adapter",
    )

    assert arbiter["backend"] == "simulation"

    for name in (
        "max_forward_mps",
        "max_lateral_mps",
        "max_yaw_rate_rps",
    ):
        assert adapter[name] == arbiter[name]

    assert adapter["allow_lateral"] is False
    assert adapter["command_timeout_seconds"] == 0.15
    assert arbiter["command_timeout_seconds"] == 0.25
