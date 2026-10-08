"""Contract for upstream map-based G1 localization."""

import ast
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
CONFIGURATION = PACKAGE_ROOT / "config/amcl.yaml"
LAUNCH_FILE = PACKAGE_ROOT / "launch/localization.launch.py"


def load_configuration():
    """Load the project-owned upstream node parameters."""
    assert CONFIGURATION.is_file(), str(CONFIGURATION)
    return yaml.safe_load(
        CONFIGURATION.read_text(encoding="utf-8")
    )


def launch_constants():
    """Return string literals from the localization launch."""
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


def test_uses_navigation_frame_and_sensor_contract():
    """AMCL consumes the established planar G1 contract."""
    document = load_configuration()
    parameters = document["amcl"]["ros__parameters"]

    assert parameters["global_frame_id"] == "map"
    assert parameters["odom_frame_id"] == "odom"
    assert parameters["base_frame_id"] == "base_footprint"
    assert parameters["scan_topic"] == "/scan"
    assert parameters["tf_broadcast"] is True


def test_matches_project_lidar_and_motion_limits():
    """Localization uses the projected MID-360 operating envelope."""
    document = load_configuration()
    parameters = document["amcl"]["ros__parameters"]

    assert parameters["laser_min_range"] == 0.25
    assert parameters["laser_max_range"] == 40.0
    assert (
        parameters["robot_model_type"]
        == "nav2_amcl::DifferentialMotionModel"
    )
    assert parameters["min_particles"] == 500
    assert parameters["max_particles"] == 2000


def test_does_not_hide_an_initial_pose_in_configuration():
    """Operators or simulation bringup must provide initial pose."""
    document = load_configuration()
    parameters = document["amcl"]["ros__parameters"]

    assert "initial_pose" not in parameters
    assert parameters["set_initial_pose"] is False


def test_launches_only_upstream_localization_nodes():
    """The package owns composition, not localization algorithms."""
    constants = launch_constants()

    assert {
        "nav2_map_server",
        "map_server",
        "nav2_amcl",
        "amcl",
        "nav2_lifecycle_manager",
        "lifecycle_manager",
        "lifecycle_manager_localization",
        "map",
        "params_file",
        "use_sim_time",
        "autostart",
    } <= constants

    assert "slam_toolbox" not in constants


def test_map_argument_is_required_and_configuration_is_installed():
    """Every localization run names its saved map explicitly."""
    source = LAUNCH_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = launch_constants()

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
    assert "g1_navigation_localization" in constants
    assert "amcl.yaml" in constants
    assert '"yaml_filename": map_file' in source
