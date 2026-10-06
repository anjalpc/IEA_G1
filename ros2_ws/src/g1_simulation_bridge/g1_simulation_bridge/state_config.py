"""Validated ROS parameters for simulation state publication."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class SimulationStatePublisherConfig:
    """Validated runtime configuration for raw simulation state."""

    state_period_seconds: float
    imu_frame_id: str
    odom_frame_id: str
    child_frame_id: str
    covariance_variance: float


def _positive_number(value, *, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and positive"
        )
    return float(value)


def _frame_id(value, *, name):
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            f"{name} must be a non-empty string "
            "without whitespace"
        )
    return value


def load_state_publisher_config(node):
    """Declare, read, and validate state publisher parameters."""

    state_period_seconds = node.declare_parameter(
        "state_period_seconds",
        0.02,
    ).value
    imu_frame_id = node.declare_parameter(
        "imu_frame_id",
        "pelvis",
    ).value
    odom_frame_id = node.declare_parameter(
        "odom_frame_id",
        "odom",
    ).value
    child_frame_id = node.declare_parameter(
        "odom_child_frame_id",
        "pelvis",
    ).value
    covariance_variance = node.declare_parameter(
        "state_covariance_variance",
        1e-6,
    ).value

    return SimulationStatePublisherConfig(
        state_period_seconds=_positive_number(
            state_period_seconds,
            name="state_period_seconds",
        ),
        imu_frame_id=_frame_id(
            imu_frame_id,
            name="imu_frame_id",
        ),
        odom_frame_id=_frame_id(
            odom_frame_id,
            name="odom_frame_id",
        ),
        child_frame_id=_frame_id(
            child_frame_id,
            name="odom_child_frame_id",
        ),
        covariance_variance=_positive_number(
            covariance_variance,
            name="state_covariance_variance",
        ),
    )
