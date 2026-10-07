"""Contract for MID-360 navigation scan projection."""

import ast
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
CONFIGURATION = (
    PACKAGE_ROOT / "config/lidar_projection.yaml"
)
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch/perception.launch.py"
)


def load_parameters():
    configuration = yaml.safe_load(
        CONFIGURATION.read_text(encoding="utf-8")
    )
    return configuration[
        "pointcloud_to_laserscan"
    ]["ros__parameters"]


def test_installs_launch_and_configuration_files():
    setup = (
        PACKAGE_ROOT / "setup.py"
    ).read_text(encoding="utf-8")

    assert CONFIGURATION.is_file()
    assert LAUNCH_FILE.is_file()
    assert 'glob("config/*.yaml")' in setup
    assert 'glob("launch/*.launch.py")' in setup


def test_uses_explicit_navigation_frames():
    parameters = load_parameters()

    assert parameters["target_frame"] == "base_footprint"
    assert parameters["transform_tolerance"] == 0.05


def test_uses_explicit_conservative_height_slice():
    parameters = load_parameters()

    assert parameters["min_height"] == 0.05
    assert parameters["max_height"] == 1.75


def test_uses_explicit_scan_geometry_and_range():
    parameters = load_parameters()

    assert parameters["angle_min"] == -3.141592653589793
    assert parameters["angle_max"] == 3.141592653589793
    assert parameters["angle_increment"] == 0.008726646259971648
    assert parameters["scan_time"] == 0.1
    assert parameters["range_min"] == 0.25
    assert parameters["range_max"] == 40.0
    assert parameters["use_inf"] is True
    assert parameters["queue_size"] == 4


def test_launches_only_upstream_projection_node():
    launch = LAUNCH_FILE.read_text(encoding="utf-8")

    assert 'package="pointcloud_to_laserscan"' in launch
    assert 'executable="pointcloud_to_laserscan_node"' in launch
    assert 'name="pointcloud_to_laserscan"' in launch

    tree = ast.parse(launch)
    string_pairs = {
        tuple(element.value for element in node.elts)
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and all(
                isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                for element in node.elts
            )
        )
    }

    assert (
        "cloud_in",
        "/g1/sensors/lidar/points",
    ) in string_pairs
    assert (
        "scan",
        "/scan",
    ) in string_pairs

    forbidden = (
        "livox_ros_driver",
        "slam_toolbox",
        "nav2_",
        "robot_state_publisher",
    )
    for token in forbidden:
        assert token not in launch


def test_launch_exposes_simulation_time_without_owning_clock():
    launch = LAUNCH_FILE.read_text(encoding="utf-8")

    assert '"use_sim_time"' in launch
    assert "DeclareLaunchArgument" in launch
    assert "LaunchConfiguration" in launch
    assert 'default_value="false"' in launch
