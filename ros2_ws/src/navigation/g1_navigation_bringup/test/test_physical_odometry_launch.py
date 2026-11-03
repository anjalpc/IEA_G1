"""Contract for guarded physical MID-360 odometry bringup."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = PACKAGE_ROOT / "launch/physical_odometry.launch.py"


def launch_constants():
    """Return every string literal from the physical launch."""
    assert LAUNCH_FILE.is_file(), str(LAUNCH_FILE)
    tree = ast.parse(LAUNCH_FILE.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def test_starts_glim_health_and_planar_staging():
    """Physical bringup owns sensing and staged qualification only."""
    values = launch_constants()

    assert {
        "g1_navigation_bringup",
        "glim_mid360",
        "glim_ros",
        "glim_rosnode",
        "config_path",
        "g1_state_estimation",
        "glim_health_node",
        "glim_health.yaml",
        "glim_planar_node",
        "physical_body_node",
        "physical_body.yaml",
        "g1_robot_description",
        "physical_robot_state.launch.py",
    } <= values

    assert "robot_state.launch.py" not in values


def test_does_not_start_navigation_mapping_or_control():
    """Odometry proof must not acquire navigation or motion authority."""
    values = launch_constants()

    forbidden = {
        "navigation.launch.py",
        "mapping.launch.py",
        "localization.launch.py",
        "navigation_control.launch.py",
        "slam_toolbox",
        "nav2_controller",
        "g1_locomotion_arbiter",
        "/g1/state/odometry",
        "map",
        "odom",
    }

    assert forbidden.isdisjoint(values)


def test_uses_project_owned_physical_configuration():
    """Every runtime configuration is installed with bringup."""
    physical = PACKAGE_ROOT / "config/physical"
    glim = physical / "glim_mid360"

    assert (physical / "cyclonedds_laptop_wired.xml").is_file()
    pc2_dds = physical / "cyclonedds_pc2_wired.xml"
    assert pc2_dds.is_file()

    pc2_root = ET.parse(pc2_dds).getroot()
    pc2_domain = pc2_root.find("Domain")
    assert pc2_domain is not None
    assert pc2_domain.attrib["id"] == "42"

    required = {
        "config.json",
        "config_global_mapping_cpu.json",
        "config_logging.json",
        "config_odometry_cpu.json",
        "config_preprocess.json",
        "config_ros.json",
        "config_sensors.json",
        "config_sub_mapping_cpu.json",
        "config_viewer.json",
    }

    assert required == {
        path.name
        for path in glim.iterdir()
        if path.is_file()
    }

    ros_configuration = (
        glim / "config_ros.json"
    ).read_text(encoding="utf-8")
    assert '"publish_imu2lidar": false' in (
        ros_configuration
    )
    assert '"publish_imu2lidar": true' not in (
        ros_configuration
    )


def test_manifest_declares_physical_odometry_dependencies():
    """Bringup declares both runtime packages explicitly."""
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    dependencies = {
        element.text
        for element in root.findall("exec_depend")
    }

    assert "glim_ros" in dependencies
    assert "g1_state_estimation" in dependencies
