"""Deployment contract for the physical LowState runtime."""

import ast
from pathlib import Path

import yaml

from g1_hardware_bridge.low_state import (
    node as hardware_state_node,
)


REPOSITORY_ROOT = Path(__file__).parents[4]
HARDWARE_PACKAGE = Path(__file__).parents[1]


def test_exposes_hardware_state_console_entry_point():
    setup = (
        HARDWARE_PACKAGE / "setup.py"
    ).read_text(encoding="utf-8")

    string_literals = {
        node.value
        for node in ast.walk(ast.parse(setup))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
    }
    assert (
        "hardware_state = "
        "g1_hardware_bridge.low_state.node:main"
    ) in string_literals
    assert callable(hardware_state_node.main)


def test_configures_explicit_fifty_hertz_period():
    configuration = yaml.safe_load(
        (
            HARDWARE_PACKAGE
            / "config/hardware.yaml"
        ).read_text(encoding="utf-8")
    )

    assert (
        "state_publish_period_s"
        not in configuration[
            "g1_hardware_bridge"
        ]["ros__parameters"]
    )

    parameters = configuration[
        "g1_hardware_state"
    ]["ros__parameters"]

    assert (
        parameters["state_publish_period_s"]
        == 0.02
    )
    assert "joint_state_publish_period_s" not in parameters


def test_pc2_bringup_launches_joint_state_source():
    launch = (
        REPOSITORY_ROOT
        / "ros2_ws/src/g1_bringup/"
        "launch/event_interaction.launch.py"
    ).read_text(encoding="utf-8")

    assert 'executable="hardware_state"' in launch
    assert 'name="g1_hardware_state"' in launch
    assert "parameters=[hardware_configuration]" in launch


def test_pc2_bringup_supervises_joint_state_source():
    launch = (
        REPOSITORY_ROOT
        / "ros2_ws/src/g1_bringup/"
        "launch/event_interaction.launch.py"
    ).read_text(encoding="utf-8")

    assert (
        '(hardware_state_node, '
        '"hardware state source exited")'
    ) in launch


def test_deployment_guide_documents_both_physical_outputs():
    guide = (
        REPOSITORY_ROOT
        / "docs/setup/robot-deployment.md"
    ).read_text(encoding="utf-8")

    assert "/joint_states" in guide
    assert "/g1/state/imu_raw" in guide
    assert "imu_in_pelvis" in guide
    assert "unknown covariance" in guide
