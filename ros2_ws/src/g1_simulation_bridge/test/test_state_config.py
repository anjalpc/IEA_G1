"""Contract tests for simulation state publisher parameters."""

import math

import pytest

from g1_simulation_bridge.state_config import (
    SimulationLidarPublisherConfig,
    SimulationLocomotionConfig,
    SimulationStatePublisherConfig,
    load_lidar_publisher_config,
    load_simulation_locomotion_config,
    load_state_publisher_config,
)


class FakeParameter:
    def __init__(self, value):
        self.value = value


class FakeNode:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.declarations = {}

    def declare_parameter(self, name, default):
        self.declarations[name] = default
        return FakeParameter(
            self.overrides.get(name, default)
        )


def test_declares_and_loads_safe_default_configuration():
    node = FakeNode()

    config = load_state_publisher_config(node)

    assert config == SimulationStatePublisherConfig(
        state_period_seconds=0.02,
        imu_frame_id="pelvis",
        odom_frame_id="odom",
        child_frame_id="pelvis",
        covariance_variance=1e-6,
    )
    assert node.declarations == {
        "state_period_seconds": 0.02,
        "imu_frame_id": "pelvis",
        "odom_frame_id": "odom",
        "odom_child_frame_id": "pelvis",
        "state_covariance_variance": 1e-6,
    }


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("state_period_seconds", 0.0),
        ("state_period_seconds", -0.02),
        ("state_period_seconds", math.nan),
        ("state_period_seconds", math.inf),
        ("state_covariance_variance", 0.0),
        ("state_covariance_variance", -1e-6),
        ("state_covariance_variance", math.nan),
        ("state_covariance_variance", math.inf),
    ],
)
def test_rejects_invalid_positive_numeric_parameter(
    parameter,
    value,
):
    node = FakeNode({parameter: value})

    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        load_state_publisher_config(node)


@pytest.mark.parametrize(
    "parameter",
    [
        "imu_frame_id",
        "odom_frame_id",
        "odom_child_frame_id",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        42,
    ],
)
def test_rejects_invalid_frame_parameter(
    parameter,
    value,
):
    node = FakeNode({parameter: value})

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        load_state_publisher_config(node)


def test_declares_safe_default_lidar_configuration():
    node = FakeNode()

    config = load_lidar_publisher_config(node)

    assert config == SimulationLidarPublisherConfig(
        enabled=False,
        period_seconds=0.1,
        frame_id="mid360_link",
        range_min=0.25,
        range_max=40.0,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
    )
    assert node.declarations == {
        "lidar_enabled": False,
        "lidar_period_seconds": 0.1,
        "lidar_frame_id": "mid360_link",
        "lidar_range_min": 0.25,
        "lidar_range_max": 40.0,
        "lidar_torso_body_name": "torso_link",
        "lidar_robot_root_body_name": "pelvis",
    }


def test_loads_explicit_lidar_configuration():
    node = FakeNode({
        "lidar_enabled": True,
        "lidar_period_seconds": 0.2,
        "lidar_frame_id": "custom_lidar",
        "lidar_range_min": 0.5,
        "lidar_range_max": 25.0,
        "lidar_torso_body_name": "custom_torso",
        "lidar_robot_root_body_name": "custom_root",
    })

    config = load_lidar_publisher_config(node)

    assert config == SimulationLidarPublisherConfig(
        enabled=True,
        period_seconds=0.2,
        frame_id="custom_lidar",
        range_min=0.5,
        range_max=25.0,
        torso_body_name="custom_torso",
        robot_root_body_name="custom_root",
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        "true",
        None,
    ],
)
def test_rejects_nonboolean_lidar_enable(value):
    node = FakeNode({
        "lidar_enabled": value,
    })

    with pytest.raises(
        ValueError,
        match="boolean",
    ):
        load_lidar_publisher_config(node)


@pytest.mark.parametrize(
    "parameter",
    [
        "lidar_period_seconds",
        "lidar_range_min",
        "lidar_range_max",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.1,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_lidar_positive_number(
    parameter,
    value,
):
    node = FakeNode({
        parameter: value,
    })

    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        load_lidar_publisher_config(node)


def test_rejects_reversed_lidar_range():
    node = FakeNode({
        "lidar_range_min": 5.0,
        "lidar_range_max": 4.0,
    })

    with pytest.raises(
        ValueError,
        match="greater than lidar_range_min",
    ):
        load_lidar_publisher_config(node)


@pytest.mark.parametrize(
    "parameter",
    [
        "lidar_frame_id",
        "lidar_torso_body_name",
        "lidar_robot_root_body_name",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "",
        "has whitespace",
        42,
    ],
)
def test_rejects_invalid_lidar_name(
    parameter,
    value,
):
    node = FakeNode({
        parameter: value,
    })

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        load_lidar_publisher_config(node)

def test_declares_disabled_planar_carrier_defaults():
    """Ordinary simulation cannot move kinematically."""
    node = FakeNode()

    config = load_simulation_locomotion_config(node)

    assert config == SimulationLocomotionConfig(
        enabled=False,
        command_timeout_seconds=0.15,
        max_forward_mps=0.15,
        max_lateral_mps=0.01,
        max_yaw_rate_rps=0.40,
        allow_lateral=False,
        joint_name="floating_base_joint",
    )
    assert node.declarations == {
        "planar_carrier_enabled": False,
        "planar_carrier_command_timeout_seconds": 0.15,
        "planar_carrier_max_forward_mps": 0.15,
        "planar_carrier_max_lateral_mps": 0.01,
        "planar_carrier_max_yaw_rate_rps": 0.40,
        "planar_carrier_allow_lateral": False,
        "planar_carrier_joint_name": "floating_base_joint",
    }


def test_loads_explicit_planar_carrier_configuration():
    """Navigation simulation can explicitly enable the fixture."""
    node = FakeNode({
        "planar_carrier_enabled": True,
        "planar_carrier_command_timeout_seconds": 0.20,
        "planar_carrier_max_forward_mps": 0.10,
        "planar_carrier_max_lateral_mps": 0.02,
        "planar_carrier_max_yaw_rate_rps": 0.30,
        "planar_carrier_allow_lateral": True,
        "planar_carrier_joint_name": "custom_free_joint",
    })

    config = load_simulation_locomotion_config(node)

    assert config == SimulationLocomotionConfig(
        enabled=True,
        command_timeout_seconds=0.20,
        max_forward_mps=0.10,
        max_lateral_mps=0.02,
        max_yaw_rate_rps=0.30,
        allow_lateral=True,
        joint_name="custom_free_joint",
    )


@pytest.mark.parametrize(
    "parameter",
    (
        "planar_carrier_enabled",
        "planar_carrier_allow_lateral",
    ),
)
@pytest.mark.parametrize("value", (0, 1, "true", None))
def test_rejects_nonboolean_planar_carrier_values(
    parameter,
    value,
):
    """Carrier enablement cannot rely on truthiness."""
    node = FakeNode({parameter: value})

    with pytest.raises(ValueError, match="boolean"):
        load_simulation_locomotion_config(node)


@pytest.mark.parametrize(
    "parameter",
    (
        "planar_carrier_command_timeout_seconds",
        "planar_carrier_max_forward_mps",
        "planar_carrier_max_lateral_mps",
        "planar_carrier_max_yaw_rate_rps",
    ),
)
@pytest.mark.parametrize(
    "value",
    (0.0, -0.1, math.nan, math.inf),
)
def test_rejects_invalid_planar_carrier_limits(
    parameter,
    value,
):
    """Carrier limits must remain finite and positive."""
    node = FakeNode({parameter: value})

    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        load_simulation_locomotion_config(node)
