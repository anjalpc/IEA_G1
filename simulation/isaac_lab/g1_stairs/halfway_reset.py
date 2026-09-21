"""Deterministic halfway-support reset for the 5 cm single-step task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class HalfwayResetConfig:
    enabled: bool = True
    leading_leg: str = "left"
    root_x_m: float = 0.92
    root_height_m: float = 0.80
    leading_foot_x_from_root_m: float = 0.13
    leading_foot_lift_m: float = 0.05
    thigh_length_m: float = 0.30001
    shank_length_m: float = 0.30001
    support_force_threshold_n: float = 42.87
    maximum_foot_speed_mps: float = 0.15
    minimum_upright_gravity_z: float = 0.80
    settle_steps: int = 20


def parse_halfway_reset_config(resolved: dict[str, Any]) -> HalfwayResetConfig | None:
    """Parse and validate the optional halfway-reset block."""
    raw = resolved.get("halfway_reset")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("halfway_reset must be a mapping")

    cfg = HalfwayResetConfig(**raw)
    if not cfg.enabled:
        return None
    if cfg.leading_leg not in {"left", "right"}:
        raise ValueError("halfway_reset.leading_leg must be left or right")
    if cfg.leading_foot_lift_m <= 0.0:
        raise ValueError("leading_foot_lift_m must be positive")
    if cfg.support_force_threshold_n <= 0.0:
        raise ValueError("support_force_threshold_n must be positive")
    if cfg.settle_steps < 1:
        raise ValueError("settle_steps must be positive")
    solve_leading_leg_pose(cfg)
    return cfg


def solve_leading_leg_pose(cfg: HalfwayResetConfig) -> tuple[float, float, float]:
    """Solve sagittal two-link IK while keeping the leading foot level.

    The target starts from the confirmed nominal G1 stance (-0.10, +0.30,
    -0.20) and raises the ankle by the configured tread height.  Link lengths
    come from the pinned G1 model; no 29-DoF pose is guessed.
    """
    nominal_hip = -0.10
    nominal_knee = 0.30
    l1 = cfg.thigh_length_m
    l2 = cfg.shank_length_m
    nominal_down = (
        l1 * math.cos(nominal_hip)
        + l2 * math.cos(nominal_hip + nominal_knee)
    )
    target_down = nominal_down - cfg.leading_foot_lift_m
    target_forward = cfg.leading_foot_x_from_root_m
    radius_sq = target_forward**2 + target_down**2
    cosine_knee = (radius_sq - l1**2 - l2**2) / (2.0 * l1 * l2)
    if not -1.0 <= cosine_knee <= 1.0:
        raise ValueError("halfway leading-foot target is outside leg workspace")
    knee = math.acos(cosine_knee)
    hip = math.atan2(target_forward, target_down) - math.atan2(
        l2 * math.sin(knee),
        l1 + l2 * math.cos(knee),
    )
    ankle = -(hip + knee)
    return hip, knee, ankle


def halfway_reset_event_params(cfg: HalfwayResetConfig) -> dict[str, Any]:
    """Return only plain data suitable for Isaac manager configuration."""
    return {"config": asdict(cfg)}


def apply_halfway_reset(env, env_ids, *, config: dict[str, Any]) -> None:
    """Isaac reset event: write the deterministic pose and zero velocities."""
    import torch

    reset_cfg = HalfwayResetConfig(**config)
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    index = {name: i for i, name in enumerate(robot.joint_names)}
    hip, knee, ankle = solve_leading_leg_pose(reset_cfg)
    prefix = reset_cfg.leading_leg
    joint_pos[:, index[f"{prefix}_hip_pitch_joint"]] = hip
    joint_pos[:, index[f"{prefix}_knee_joint"]] = knee
    joint_pos[:, index[f"{prefix}_ankle_pitch_joint"]] = ankle

    root_pose = robot.data.default_root_state[env_ids, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, 0] += reset_cfg.root_x_m
    root_pose[:, 2] = (
        env.scene.env_origins[env_ids, 2] + reset_cfg.root_height_m
    )
    root_pose[:, 3:7] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0),
        dtype=root_pose.dtype,
        device=root_pose.device,
    )
    root_velocity = torch.zeros((len(env_ids), 6), device=env.device)

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def configure_halfway_reset(env_cfg, cfg: HalfwayResetConfig) -> None:
    """Replace the ordinary base/joint resets with one atomic halfway reset."""
    from isaaclab.managers import EventTermCfg as EventTerm

    env_cfg.events.reset_base = EventTerm(
        func=apply_halfway_reset,
        mode="reset",
        # Isaac manager configuration must contain plain serializable values.
        # Reconstruct HalfwayResetConfig only when the event runs, after the
        # scene and articulation have been created.
        params=halfway_reset_event_params(cfg),
    )
    env_cfg.events.reset_robot_joints = None


def validate_halfway_reset(
    env,
    cfg: HalfwayResetConfig,
    *,
    observations=None,
) -> dict[str, Any]:
    """Validate all live environments after settling; raise on any failure."""
    import torch

    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    body_index = {name: i for i, name in enumerate(robot.body_names)}
    sensor_index = {name: i for i, name in enumerate(sensor.body_names)}
    leading = f"{cfg.leading_leg}_ankle_roll_link"
    trailing_leg = "right" if cfg.leading_leg == "left" else "left"
    trailing = f"{trailing_leg}_ankle_roll_link"
    origins = env.scene.env_origins
    if hasattr(robot.data, "body_link_lin_vel_w"):
        velocity = robot.data.body_link_lin_vel_w
    else:
        velocity = robot.data.body_lin_vel_w

    lead_pos = robot.data.body_pos_w[:, body_index[leading]] - origins
    trail_pos = robot.data.body_pos_w[:, body_index[trailing]] - origins
    lead_speed = torch.linalg.vector_norm(velocity[:, body_index[leading]], dim=1)
    lead_force = sensor.data.net_forces_w[:, sensor_index[leading], 2].clamp(min=0)
    leading_region = (lead_pos[:, 0] >= 1.02) & (lead_pos[:, 0] <= 1.23)
    leading_height = torch.abs(lead_pos[:, 2] - 0.15) <= 0.025
    leading_support = (
        leading_region
        & leading_height
        & (lead_force > cfg.support_force_threshold_n)
        & (lead_speed < cfg.maximum_foot_speed_mps)
    )
    trailing_lower = trail_pos[:, 0] < 1.0
    upright = -robot.data.projected_gravity_b[:, 2] >= cfg.minimum_upright_gravity_z

    forbidden = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    from .step_up_isaac import FORBIDDEN_CONTACT_BODY_NAMES
    for name in FORBIDDEN_CONTACT_BODY_NAMES:
        force = sensor.data.net_forces_w[:, sensor_index[name]]
        forbidden |= torch.linalg.vector_norm(force, dim=1) >= 5.0

    # Observations were already generated by env.reset()/env.step().
    # Never re-enter ObservationManager.compute() from this validator.
    finite = torch.ones(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
    )

    def _collect_tensors(value):
        if torch.is_tensor(value):
            return [value]

        if isinstance(value, dict):
            result = []
            for nested in value.values():
                result.extend(_collect_tensors(nested))
            return result

        if isinstance(value, (tuple, list)):
            result = []
            for nested in value:
                result.extend(_collect_tensors(nested))
            return result

        return []

    if observations is not None:
        for tensor in _collect_tensors(observations):
            finite &= (
                torch.isfinite(tensor)
                .reshape(env.num_envs, -1)
                .all(dim=1)
            )

    valid = leading_support & trailing_lower & upright & ~forbidden & finite
    summary = {
        "valid_reset_rate": float(valid.float().mean().item()),
        "leading_support_rate": float(leading_support.float().mean().item()),
        "upright_rate": float(upright.float().mean().item()),
        "forbidden_contact_rate": float(forbidden.float().mean().item()),
        "finite_observation_rate": float(finite.float().mean().item()),
        "minimum_leading_force_n": float(lead_force.min().item()),
        "maximum_leading_speed_mps": float(lead_speed.max().item()),
        "detector_initial_state": "FIRST_SUPPORT",
    }
    if not bool(valid.all().item()):
        raise RuntimeError(f"invalid 021 halfway reset: {summary}")
    return summary
