"""Contract for asynchronous G1 online mapping."""

import ast
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
CONFIGURATION = PACKAGE_ROOT / "config/online_async.yaml"
LAUNCH_FILE = PACKAGE_ROOT / "launch/mapping.launch.py"


def load_parameters():
    configuration = yaml.safe_load(
        CONFIGURATION.read_text(encoding="utf-8")
    )
    return configuration[
        "slam_toolbox"
    ]["ros__parameters"]


def load_launch_source():
    return LAUNCH_FILE.read_text(encoding="utf-8")


def launch_string_constants(source):
    tree = ast.parse(source)
    return {
        node.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
    }


def test_contains_mapping_configuration_and_launch():
    assert CONFIGURATION.is_file()
    assert LAUNCH_FILE.is_file()


def test_uses_established_navigation_frame_contract():
    parameters = load_parameters()

    assert parameters["map_frame"] == "map"
    assert parameters["odom_frame"] == "odom"
    assert parameters["base_frame"] == "base_footprint"
    assert parameters["scan_topic"] == "/scan"


def test_starts_in_mapping_mode_with_explicit_geometry():
    parameters = load_parameters()

    assert parameters["mode"] == "mapping"
    assert parameters["resolution"] == 0.05
    assert parameters["max_laser_range"] == 40.0
    assert parameters["minimum_time_interval"] >= 0.1
    assert parameters["transform_timeout"] > 0.0
    assert parameters["map_update_interval"] > 0.0


def test_launches_only_upstream_asynchronous_slam():
    source = load_launch_source()
    constants = launch_string_constants(source)

    assert "slam_toolbox" in constants
    assert "async_slam_toolbox_node" in constants
    assert "sync_slam_toolbox_node" not in constants
    assert "Node" in source

    forbidden = (
        "nav2_",
        "cmd_vel",
        "LocomotionCommand",
        "robot_state_publisher",
        "pointcloud_to_laserscan",
    )
    for token in forbidden:
        assert token not in source


def test_launch_exposes_clock_selection_without_owning_clock():
    source = load_launch_source()

    assert '"use_sim_time"' in source
    assert "DeclareLaunchArgument" in source
    assert "LaunchConfiguration" in source
    assert 'default_value="false"' in source
    assert "/clock" not in source


def test_launch_supplies_installed_configuration():
    source = load_launch_source()
    constants = launch_string_constants(source)

    assert "g1_navigation_mapping" in constants
    assert "online_async.yaml" in constants
    assert "get_package_share_directory" in source
    assert "parameters" in source
