"""Isaac Lab adapter for the physical G1 step-up detector."""

from __future__ import annotations

from dataclasses import dataclass
import math

from simulation.isaac_lab.g1_stairs.contact_sensor_contract import (
    resolve_available_body_indices,
)
from simulation.isaac_lab.g1_stairs.step_up_state import (
    FootSupportObservation,
    StepUpObservation,
)


FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

FORBIDDEN_CONTACT_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_rubber_hand",
    "right_rubber_hand",
)


@dataclass(frozen=True)
class StepUpIsaacConfig:
    """Geometry and physical thresholds for a single stair."""

    first_riser_x_m: float = 1.0
    tread_depth_m: float = 0.25
    riser_height_m: float = 0.05

    nominal_ankle_height_m: float = 0.10
    foot_height_tolerance_m: float = 0.025
    tread_x_margin_m: float = 0.02

    maximum_supported_foot_speed_mps: float = 0.15
    minimum_upright_gravity_z: float = 0.80

    # Must be replaced with a value obtained from measured stance forces.
    minimum_vertical_support_force_n: float = 1.0

    # Reference standing base height measured before the stair.
    lower_level_base_height_m: float = 0.0
    minimum_base_height_gain_m: float = 0.025

    # Require the base to move at least slightly beyond the riser plane.
    base_past_riser_margin_m: float = 0.02

    forbidden_contact_force_threshold_n: float = 5.0


def _resolve_indices(names, required):
    index_by_name = {
        name: index
        for index, name in enumerate(names)
    }

    missing = [
        name
        for name in required
        if name not in index_by_name
    ]

    if missing:
        raise RuntimeError(
            "required bodies missing: "
            + ", ".join(missing)
        )

    return {
        name: index_by_name[name]
        for name in required
    }


def _body_linear_velocity(data):
    """Use the Isaac Lab body linear-velocity tensor available at runtime."""

    if hasattr(data, "body_link_lin_vel_w"):
        return data.body_link_lin_vel_w

    if hasattr(data, "body_lin_vel_w"):
        return data.body_lin_vel_w

    raise RuntimeError(
        "robot articulation data has no body linear velocity tensor"
    )


def capture_step_up_observation(
    env,
    *,
    env_index: int,
    config: StepUpIsaacConfig,
    upper_forward_continuation_m: float = 0.0,
) -> StepUpObservation:
    """Convert one live Isaac environment into one detector observation."""

    import torch

    scene = env.scene
    robot = scene["robot"]

    if "contact_forces" not in scene.sensors:
        raise RuntimeError(
            "required contact_forces sensor is missing"
        )

    sensor = scene.sensors["contact_forces"]

    robot_indices = _resolve_indices(
        tuple(robot.body_names),
        FOOT_BODY_NAMES,
    )

    sensor_indices = _resolve_indices(
        tuple(sensor.body_names),
        FOOT_BODY_NAMES,
    )
    forbidden_sensor_indices = resolve_available_body_indices(
        tuple(sensor.body_names),
        FORBIDDEN_CONTACT_BODY_NAMES,
    )

    data = robot.data
    body_velocity_w = _body_linear_velocity(data)

    env_origin = scene.env_origins[env_index]

    tread_start_x = (
        config.first_riser_x_m
        + config.tread_x_margin_m
    )
    tread_end_x = (
        config.first_riser_x_m
        + config.tread_depth_m
        - config.tread_x_margin_m
    )

    expected_ankle_height = (
        config.nominal_ankle_height_m
        + config.riser_height_m
    )

    def foot_observation(name: str) -> FootSupportObservation:
        robot_index = robot_indices[name]
        sensor_index = sensor_indices[name]

        position = data.body_pos_w[
            env_index,
            robot_index,
        ]

        velocity = body_velocity_w[
            env_index,
            robot_index,
        ]

        local_x = float(
            position[0].item()
            - env_origin[0].item()
        )

        local_z = float(
            position[2].item()
            - env_origin[2].item()
        )

        vertical_force_n = max(
            0.0,
            float(
                sensor.data.net_forces_w[
                    env_index,
                    sensor_index,
                    2,
                ].item()
            ),
        )

        speed_mps = float(
            torch.linalg.vector_norm(
                velocity
            ).item()
        )

        return FootSupportObservation(
            in_upper_region=(
                tread_start_x
                <= local_x
                <= tread_end_x
            ),
            height_ok=(
                abs(
                    local_z
                    - expected_ankle_height
                )
                <= config.foot_height_tolerance_m
            ),
            vertical_force_n=vertical_force_n,
            speed_mps=speed_mps,
        )

    left = foot_observation(
        "left_ankle_roll_link"
    )
    right = foot_observation(
        "right_ankle_roll_link"
    )

    forbidden_contact = False

    for sensor_index in forbidden_sensor_indices.values():
        force = sensor.data.net_forces_w[
            env_index,
            sensor_index,
        ]

        force_norm = float(
            torch.linalg.vector_norm(
                force
            ).item()
        )

        if (
            force_norm
            >= config.forbidden_contact_force_threshold_n
        ):
            forbidden_contact = True
            break

    local_base_x = float(
        data.root_pos_w[
            env_index,
            0,
        ].item()
        - env_origin[0].item()
    )

    local_base_z = float(
        data.root_pos_w[
            env_index,
            2,
        ].item()
        - env_origin[2].item()
    )

    upright_gravity_z = float(
        -data.projected_gravity_b[
            env_index,
            2,
        ].item()
    )

    values = (
        left.vertical_force_n,
        left.speed_mps,
        right.vertical_force_n,
        right.speed_mps,
        local_base_x,
        local_base_z,
        upright_gravity_z,
    )

    if not all(math.isfinite(v) for v in values):
        raise RuntimeError(
            "non-finite step-up detector observation"
        )

    return StepUpObservation(
        left=left,
        right=right,
        upright_gravity_z=upright_gravity_z,
        base_past_riser=(
            local_base_x
            >= (
                config.first_riser_x_m
                + config.base_past_riser_margin_m
            )
        ),
        base_height_ok=(
            local_base_z
            >= (
                config.lower_level_base_height_m
                + config.minimum_base_height_gain_m
            )
        ),
        forbidden_contact=forbidden_contact,
        upper_forward_continuation_m=(
            upper_forward_continuation_m
        ),
    )
