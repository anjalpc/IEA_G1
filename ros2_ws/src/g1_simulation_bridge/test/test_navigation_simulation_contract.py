"""Contract for the opt-in navigation simulation carrier."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config/simulation.yaml"
NAVIGATION_CONFIG = PACKAGE_ROOT / "config/navigation_simulation.yaml"
LAUNCH_FILE = PACKAGE_ROOT / "launch/simulation.launch.py"


def load_parameters(path):
    """Load simulation bridge ROS parameters."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["g1_simulation_bridge"]["ros__parameters"]


def launch_constants():
    """Return string constants from the simulation launch file."""
    tree = ast.parse(LAUNCH_FILE.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def test_standard_simulation_keeps_planar_carrier_disabled():
    """Normal MuJoCo operation must not silently become kinematic."""
    parameters = load_parameters(DEFAULT_CONFIG)

    assert parameters["planar_carrier_enabled"] is False


def test_navigation_simulation_explicitly_enables_carrier():
    """The navigation fixture opts into planar carrier motion."""
    parameters = load_parameters(NAVIGATION_CONFIG)

    assert parameters["planar_carrier_enabled"] is True
    assert parameters["planar_carrier_command_timeout_seconds"] == 0.15
    assert parameters["planar_carrier_max_forward_mps"] == 0.15
    assert parameters["planar_carrier_max_lateral_mps"] == 0.01
    assert parameters["planar_carrier_max_yaw_rate_rps"] == 0.40
    assert parameters["planar_carrier_allow_lateral"] is False
    assert parameters["planar_carrier_joint_name"] == "floating_base_joint"


def test_navigation_config_preserves_sensor_contract():
    """Carrier mode must not alter state or lidar configuration."""
    standard = load_parameters(DEFAULT_CONFIG)
    navigation = load_parameters(NAVIGATION_CONFIG)

    for name in (
        "state_period_seconds",
        "imu_frame_id",
        "odom_frame_id",
        "odom_child_frame_id",
        "lidar_enabled",
        "lidar_period_seconds",
        "lidar_frame_id",
        "lidar_range_min",
        "lidar_range_max",
    ):
        assert navigation[name] == standard[name]


def test_simulation_launch_accepts_a_parameter_file_override():
    """One launch file serves standard and navigation simulation."""
    constants = launch_constants()

    assert "params_file" in constants
    assert "Full path to the simulation bridge parameter file" in constants

def test_manifest_declares_simulation_command_message_dependency():
    """The bridge directly imports geometry_msgs at runtime."""
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    dependencies = {
        element.text
        for element in root.findall("exec_depend")
    }

    assert "geometry_msgs" in dependencies

