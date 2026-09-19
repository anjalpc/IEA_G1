"""Convert live Isaac Lab measurements into independent evaluator samples."""

from __future__ import annotations

from dataclasses import dataclass
import math

from simulation.isaac_lab.g1_stairs.geometry import nominal_staircase
from simulation.isaac_lab.g1_stairs.evaluation.stair_outcomes import StepSample
from simulation.isaac_lab.g1_stairs.evaluation.stair_outcomes import StairOutcomeConfig


CONTACT_BODY_LABELS = {
    "pelvis": "pelvis",
    "torso_link": "torso",
    "left_knee_link": "left_knee",
    "right_knee_link": "right_knee",
    "left_elbow_link": "left_forearm",
    "right_elbow_link": "right_forearm",
    "left_rubber_hand": "left_hand",
    "right_rubber_hand": "right_hand",
}
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class IsaacLabStepState:
    """Simulator values captured for one environment at one control step."""

    time_s: float
    root_position_world_m: tuple[float, float, float]
    environment_origin_world_m: tuple[float, float, float]
    root_quaternion_wxyz: tuple[float, float, float, float]
    contacts: tuple[str, ...] = ()
    joint_violation: bool = False
    velocity_violation: bool = False
    torque_violation: bool = False
    slip_violation: bool = False
    timed_out: bool = False


def quaternion_wxyz_to_roll_pitch(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return roll and pitch from a normalized-or-near-normalized quaternion."""
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        return (math.nan, math.nan)

    w, x, y, z = (value / norm for value in quaternion)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    return (roll, pitch)


def course_x_m(
    state: IsaacLabStepState,
    *,
    start_x_m: float | None = None,
) -> float:
    """Convert world X into the evaluator's staircase-course coordinate."""
    if start_x_m is None:
        start_x_m = nominal_staircase().fixed_start_xy_yaw[0]
    return (
        state.root_position_world_m[0]
        - state.environment_origin_world_m[0]
        + start_x_m
    )


def step_index_from_course_x(
    base_x_m: float,
    *,
    step_edges_m: tuple[float, ...] | None = None,
) -> int:
    """Return the highest stair level whose leading edge was reached."""
    if step_edges_m is None:
        spec = nominal_staircase()
        step_edges_m = tuple(
            spec.approach_length + index * spec.tread_depth
            for index in range(spec.step_count)
        )
    return sum(base_x_m >= edge for edge in step_edges_m)


def to_step_sample(
    state: IsaacLabStepState,
    *,
    config: StairOutcomeConfig | None = None,
) -> StepSample:
    """Create one reward-independent evaluator sample."""
    if config is None:
        config = StairOutcomeConfig()
    step_edges = tuple(
        config.first_riser_x_m + index * config.tread_depth_m
        for index in range(config.step_count)
    )
    base_x = course_x_m(state, start_x_m=config.start_x_m)
    roll, pitch = quaternion_wxyz_to_roll_pitch(
        state.root_quaternion_wxyz
    )
    return StepSample.create(
        time_s=state.time_s,
        base_x_m=base_x,
        base_height_m=(
            state.root_position_world_m[2]
            - state.environment_origin_world_m[2]
        ),
        roll_rad=roll,
        pitch_rad=pitch,
        step_index=step_index_from_course_x(
            base_x,
            step_edges_m=step_edges,
        ),
        contacts=state.contacts,
        joint_violation=state.joint_violation,
        velocity_violation=state.velocity_violation,
        torque_violation=state.torque_violation,
        slip_violation=state.slip_violation,
        timed_out=state.timed_out,
    )


def resolve_body_indices(
    body_names: list[str] | tuple[str, ...],
    required_names: tuple[str, ...],
) -> dict[str, int]:
    """Resolve body indices by name and reject incomplete sensor coverage."""
    index_by_name = {name: index for index, name in enumerate(body_names)}
    missing = [name for name in required_names if name not in index_by_name]
    if missing:
        raise ValueError("missing bodies: " + ", ".join(missing))
    return {name: index_by_name[name] for name in required_names}


def capture_isaac_lab_state(
    env,
    *,
    time_s: float,
    env_index: int = 0,
    timed_out: bool = False,
    contact_threshold_n: float = 1.0,
    slip_speed_threshold_mps: float = 0.20,
) -> IsaacLabStepState:
    """Capture one live Isaac Lab environment using the audited 2.3 tensors."""
    import torch

    scene = env.scene
    robot = scene["robot"]
    if "contact_forces" not in scene.sensors:
        raise RuntimeError("required contact_forces sensor is missing")
    sensor = scene.sensors["contact_forces"]
    forces = sensor.data.net_forces_w[env_index]

    required_contact_bodies = tuple(CONTACT_BODY_LABELS) + FOOT_BODY_NAMES
    sensor_indices = resolve_body_indices(
        tuple(sensor.body_names), required_contact_bodies
    )
    robot_indices = resolve_body_indices(
        tuple(robot.body_names), FOOT_BODY_NAMES
    )

    data = robot.data
    tensors = (
        data.root_pos_w[env_index],
        data.root_quat_w[env_index],
        data.joint_pos[env_index],
        data.joint_vel[env_index],
        data.applied_torque[env_index],
        forces,
    )
    finite = all(bool(torch.all(torch.isfinite(value)).item()) for value in tensors)

    contacts = []
    for body_name, label in CONTACT_BODY_LABELS.items():
        force = forces[sensor_indices[body_name]]
        if bool((torch.linalg.vector_norm(force) > contact_threshold_n).item()):
            contacts.append(label)

    position_limits = data.soft_joint_pos_limits[env_index]
    velocity_limits = data.soft_joint_vel_limits[env_index]
    effort_limits = data.joint_effort_limits[env_index]
    joint_violation = bool(
        torch.any(
            (data.joint_pos[env_index] < position_limits[:, 0])
            | (data.joint_pos[env_index] > position_limits[:, 1])
        ).item()
    )
    velocity_violation = bool(
        torch.any(torch.abs(data.joint_vel[env_index]) > velocity_limits).item()
    )
    torque_violation = bool(
        torch.any(torch.abs(data.applied_torque[env_index]) > effort_limits).item()
    )

    slip_violation = False
    for body_name in FOOT_BODY_NAMES:
        force = forces[sensor_indices[body_name]]
        in_contact = torch.linalg.vector_norm(force) > contact_threshold_n
        velocity = data.body_lin_vel_w[
            env_index, robot_indices[body_name], :2
        ]
        slipping = torch.linalg.vector_norm(velocity) > slip_speed_threshold_mps
        slip_violation = slip_violation or bool((in_contact & slipping).item())

    root_position = tuple(float(value) for value in data.root_pos_w[env_index].tolist())
    quaternion = tuple(float(value) for value in data.root_quat_w[env_index].tolist())
    if not finite:
        root_position = (math.nan, root_position[1], root_position[2])

    return IsaacLabStepState(
        time_s=time_s,
        root_position_world_m=root_position,
        environment_origin_world_m=tuple(
            float(value) for value in scene.env_origins[env_index].tolist()
        ),
        root_quaternion_wxyz=quaternion,
        contacts=tuple(sorted(contacts)),
        joint_violation=joint_violation,
        velocity_violation=velocity_violation,
        torque_violation=torque_violation,
        slip_violation=slip_violation,
        timed_out=timed_out,
    )
