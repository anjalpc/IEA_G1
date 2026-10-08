"""Contract for recording-safe G1 Nav2 bringup."""

import ast
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
CONFIGURATION = PACKAGE_ROOT / "config/navigation.yaml"
LAUNCH_FILE = PACKAGE_ROOT / "launch/navigation.launch.py"
RAW_COMMAND_TOPIC = "/g1/navigation/cmd_vel_raw"


def load_configuration():
    assert CONFIGURATION.is_file(), str(CONFIGURATION)
    return yaml.safe_load(
        CONFIGURATION.read_text(encoding="utf-8")
    )


def load_launch_source():
    assert LAUNCH_FILE.is_file(), str(LAUNCH_FILE)
    return LAUNCH_FILE.read_text(encoding="utf-8")


def launch_string_constants(source):
    tree = ast.parse(source)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def parameters(configuration, node_name):
    return configuration[node_name]["ros__parameters"]


def test_contains_navigation_configuration_and_launch():
    assert CONFIGURATION.is_file()
    assert LAUNCH_FILE.is_file()


def test_uses_established_navigation_frames_and_topics():
    configuration = load_configuration()

    bt = parameters(configuration, "bt_navigator")
    controller = parameters(configuration, "controller_server")
    local = configuration[
        "local_costmap"
    ]["local_costmap"]["ros__parameters"]
    global_costmap = configuration[
        "global_costmap"
    ]["global_costmap"]["ros__parameters"]

    assert bt["global_frame"] == "map"
    assert bt["robot_base_frame"] == "base_footprint"
    assert bt["odom_topic"] == "/g1/state/odometry"

    assert controller["odom_topic"] == "/g1/state/odometry"

    assert local["global_frame"] == "odom"
    assert local["robot_base_frame"] == "base_footprint"
    assert global_costmap["global_frame"] == "map"
    assert (
        global_costmap["robot_base_frame"]
        == "base_footprint"
    )


def test_local_costmap_dimensions_use_humble_integer_types():
    configuration = load_configuration()
    local = configuration[
        "local_costmap"
    ]["local_costmap"]["ros__parameters"]

    assert local["width"] == 4
    assert local["height"] == 4
    assert isinstance(local["width"], int)
    assert isinstance(local["height"], int)


def test_costmaps_consume_the_approved_scan():
    configuration = load_configuration()
    local = configuration[
        "local_costmap"
    ]["local_costmap"]["ros__parameters"]
    global_costmap = configuration[
        "global_costmap"
    ]["global_costmap"]["ros__parameters"]

    local_scan = local["obstacle_layer"]["scan"]
    global_scan = global_costmap["obstacle_layer"]["scan"]

    for scan in (local_scan, global_scan):
        assert scan["topic"] == "/scan"
        assert scan["data_type"] == "LaserScan"
        assert scan["marking"] is True
        assert scan["clearing"] is True


def test_uses_upstream_navfn_and_dwb_plugins():
    configuration = load_configuration()
    planner = parameters(configuration, "planner_server")
    controller = parameters(
        configuration,
        "controller_server",
    )

    assert (
        planner["GridBased"]["plugin"]
        == "nav2_navfn_planner/NavfnPlanner"
    )
    assert (
        controller["FollowPath"]["plugin"]
        == "dwb_core::DWBLocalPlanner"
    )


def test_initial_controller_disables_lateral_motion():
    configuration = load_configuration()
    follow_path = parameters(
        configuration,
        "controller_server",
    )["FollowPath"]

    assert follow_path["min_vel_y"] == 0.0
    assert follow_path["max_vel_y"] == 0.0
    assert follow_path["acc_lim_y"] == 0.0
    assert follow_path["decel_lim_y"] == 0.0
    assert follow_path["vy_samples"] == 1


def test_initial_velocity_limits_are_conservative():
    configuration = load_configuration()
    follow_path = parameters(
        configuration,
        "controller_server",
    )["FollowPath"]
    behavior = parameters(
        configuration,
        "behavior_server",
    )
    velocity = parameters(
        configuration,
        "velocity_smoother",
    )

    assert 0.0 < follow_path["max_vel_x"] <= 0.20
    assert 0.0 < follow_path["max_vel_theta"] <= 0.50
    assert behavior["max_rotational_vel"] <= 0.50

    assert velocity["max_velocity"][0] <= 0.20
    assert velocity["max_velocity"][1] == 0.0
    assert velocity["max_velocity"][2] <= 0.50


def test_launch_starts_only_upstream_nav2_nodes():
    source = load_launch_source()
    constants = launch_string_constants(source)

    required = {
        "g1_navigation_bringup",
        "navigation.yaml",
        "nav2_behaviors",
        "behavior_server",
        "nav2_bt_navigator",
        "bt_navigator",
        "nav2_controller",
        "controller_server",
        "nav2_lifecycle_manager",
        "lifecycle_manager",
        "nav2_planner",
        "planner_server",
        "nav2_smoother",
        "smoother_server",
        "nav2_velocity_smoother",
        "velocity_smoother",
        "nav2_waypoint_follower",
        "waypoint_follower",
    }
    assert required <= constants
    assert "IncludeLaunchDescription" not in source


def test_launch_defaults_inactive_and_remaps_final_commands():
    source = load_launch_source()

    assert '"autostart"' in source
    assert 'default_value="false"' in source
    assert "SetRemap" not in source
    assert source.count('"cmd_vel_nav"') >= 2
    assert source.count(RAW_COMMAND_TOPIC) == 1
    assert source.count("RAW_COMMAND_TOPIC") >= 3


def test_bringup_contains_no_hardware_or_lease_logic():
    source = load_launch_source()
    forbidden = (
        "LocomotionCommand",
        "/g1/locomotion/command",
        "/g1/locomotion/acquire",
        "unitree",
        "sport",
        "lowcmd",
    )
    for token in forbidden:
        assert token not in source.lower()
