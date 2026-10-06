"""Contract tests for simulation state publisher parameters."""

import math

import pytest

from g1_simulation_bridge.state_config import (
    SimulationStatePublisherConfig,
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
