"""Contract for recording-only physical controller validation."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT
    / "launch"
    / "physical_control_dry_run.launch.py"
)


def launch_constants():
    """Return literal strings from the dry-run launch."""
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


def test_composes_planning_controller_and_recording_boundary():
    """Dry run connects physical planning to recorded control."""
    constants = launch_constants()

    assert {
        "physical_planning.launch.py",
        "navigation_control.launch.py",
        "nav2_controller",
        "controller_server",
        "nav2_velocity_smoother",
        "velocity_smoother",
        "nav2_lifecycle_manager",
        "lifecycle_manager_control_dry_run",
    } <= constants


def test_routes_controller_output_through_smoother():
    """Only smoothed velocity reaches the command adapter."""
    constants = launch_constants()

    assert {
        "cmd_vel_nav",
        "/g1/navigation/cmd_vel_raw",
    } <= constants


def test_uses_recording_control_configuration():
    """Dry run explicitly selects the recording boundary."""
    constants = launch_constants()

    assert {
        "navigation_control.yaml",
        "params_file",
        "control_params_file",
    } <= constants


def test_starts_no_physical_or_behavior_authority():
    """Dry run cannot contact hardware or execute behavior trees."""
    constants = launch_constants()

    forbidden = {
        "sport",
        "unitree",
        "bt_navigator",
        "behavior_server",
        "waypoint_follower",
        "/cmd_vel",
    }
    assert forbidden.isdisjoint(constants)
