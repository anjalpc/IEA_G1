"""Contract for opt-in physical odometry authority."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = Path(__file__).parents[5]
LAUNCH_FILE = (
    PACKAGE_ROOT
    / "launch/physical_odometry_authority.launch.py"
)
STATE_PACKAGE = (
    REPOSITORY_ROOT
    / "ros2_ws/src/navigation/g1_state_estimation"
)


def launch_constants():
    """Return string literals from the authority launch."""

    assert LAUNCH_FILE.is_file(), str(LAUNCH_FILE)
    tree = ast.parse(LAUNCH_FILE.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def test_explicitly_wraps_staging_with_one_authority_node():
    values = launch_constants()

    assert {
        "g1_navigation_bringup",
        "physical_odometry.launch.py",
        "g1_state_estimation",
        "physical_authority_node",
        "g1_physical_odometry_authority",
        "physical_authority.yaml",
    } <= values


def test_does_not_start_mapping_navigation_or_control():
    values = launch_constants()

    forbidden = {
        "mapping.launch.py",
        "localization.launch.py",
        "navigation.launch.py",
        "navigation_control.launch.py",
        "slam_toolbox",
        "nav2_controller",
        "g1_locomotion_arbiter",
    }
    assert forbidden.isdisjoint(values)


def test_installs_authority_entrypoint_and_configuration():
    setup_text = (
        STATE_PACKAGE / "setup.py"
    ).read_text(encoding="utf-8")
    configuration = (
        STATE_PACKAGE / "config/physical_authority.yaml"
    )

    assert "physical_authority_node = " in setup_text
    assert configuration.is_file()

    text = configuration.read_text(encoding="utf-8")
    assert "warmup_samples: 20" in text
    assert "stale_after_seconds: 0.25" in text
    assert "maximum_translation_step: 0.15" in text
    assert "maximum_rotation_step_degrees: 15.0" in text
