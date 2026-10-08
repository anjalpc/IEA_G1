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


@dataclass(frozen=True)
class SimulationLocomotionConfig:
    """Validated simulation-only planar carrier settings."""

    enabled: bool
    command_timeout_seconds: float
    max_forward_mps: float
    max_lateral_mps: float
    max_yaw_rate_rps: float
    allow_lateral: bool
    joint_name: str


@dataclass(frozen=True)
class SimulationLidarPublisherConfig:
    """Validated runtime configuration for simulated lidar."""

    enabled: bool
    period_seconds: float
    frame_id: str
    range_min: float
    range_max: float
    torso_body_name: str
    robot_root_body_name: str


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


def _boolean(value, *, name):
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be boolean"
        )
    return value


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


def load_lidar_publisher_config(node):
    """Declare, read, and validate simulated lidar parameters."""
    enabled = node.declare_parameter(
        "lidar_enabled",
        False,
    ).value
    period_seconds = node.declare_parameter(
        "lidar_period_seconds",
        0.1,
    ).value
    frame_id = node.declare_parameter(
        "lidar_frame_id",
        "mid360_link",
    ).value
    range_min = node.declare_parameter(
        "lidar_range_min",
        0.25,
    ).value
    range_max = node.declare_parameter(
        "lidar_range_max",
        40.0,
    ).value
    torso_body_name = node.declare_parameter(
        "lidar_torso_body_name",
        "torso_link",
    ).value
    robot_root_body_name = (
        node.declare_parameter(
            "lidar_robot_root_body_name",
            "pelvis",
        ).value
    )

    period_seconds = _positive_number(
        period_seconds,
        name="lidar_period_seconds",
    )
    range_min = _positive_number(
        range_min,
        name="lidar_range_min",
    )
    range_max = _positive_number(
        range_max,
        name="lidar_range_max",
    )
    if range_max <= range_min:
        raise ValueError(
            "lidar_range_max must be greater "
            "than lidar_range_min"
        )

    return SimulationLidarPublisherConfig(
        enabled=_boolean(
            enabled,
            name="lidar_enabled",
        ),
        period_seconds=period_seconds,
        frame_id=_frame_id(
            frame_id,
            name="lidar_frame_id",
        ),
        range_min=range_min,
        range_max=range_max,
        torso_body_name=_frame_id(
            torso_body_name,
            name="lidar_torso_body_name",
        ),
        robot_root_body_name=_frame_id(
            robot_root_body_name,
            name="lidar_robot_root_body_name",
        ),
    )

def load_simulation_locomotion_config(node):
    """Declare and validate the planar carrier parameters."""
    enabled = node.declare_parameter(
        "planar_carrier_enabled",
        False,
    ).value
    command_timeout_seconds = node.declare_parameter(
        "planar_carrier_command_timeout_seconds",
        0.15,
    ).value
    max_forward_mps = node.declare_parameter(
        "planar_carrier_max_forward_mps",
        0.15,
    ).value
    max_lateral_mps = node.declare_parameter(
        "planar_carrier_max_lateral_mps",
        0.01,
    ).value
    max_yaw_rate_rps = node.declare_parameter(
        "planar_carrier_max_yaw_rate_rps",
        0.40,
    ).value
    allow_lateral = node.declare_parameter(
        "planar_carrier_allow_lateral",
        False,
    ).value
    joint_name = node.declare_parameter(
        "planar_carrier_joint_name",
        "floating_base_joint",
    ).value

    return SimulationLocomotionConfig(
        enabled=_boolean(
            enabled,
            name="planar_carrier_enabled",
        ),
        command_timeout_seconds=_positive_number(
            command_timeout_seconds,
            name=(
                "planar_carrier_command_timeout_seconds"
            ),
        ),
        max_forward_mps=_positive_number(
            max_forward_mps,
            name="planar_carrier_max_forward_mps",
        ),
        max_lateral_mps=_positive_number(
            max_lateral_mps,
            name="planar_carrier_max_lateral_mps",
        ),
        max_yaw_rate_rps=_positive_number(
            max_yaw_rate_rps,
            name="planar_carrier_max_yaw_rate_rps",
        ),
        allow_lateral=_boolean(
            allow_lateral,
            name="planar_carrier_allow_lateral",
        ),
        joint_name=_frame_id(
            joint_name,
            name="planar_carrier_joint_name",
        ),
    )
