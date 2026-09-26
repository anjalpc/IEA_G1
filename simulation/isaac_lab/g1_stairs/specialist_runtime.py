"""Configuration-driven runtime for the specialist stair-climbing policy.

This module intentionally contains no experiment-number-specific behavior.
Experiment YAML selects whether the specialist profile is active and defines
its staging distribution. Archived pre-specialist controller/reward runtimes are intentionally not
executable from the active specialist branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpecialistStagingCfg:
    """Validated staging distribution immediately before the first riser."""

    distance_min_m: float
    distance_max_m: float
    lateral_min_m: float
    lateral_max_m: float
    yaw_min_rad: float
    yaw_max_rad: float
    episode_length_s: float | None = None

    @property
    def nominal_distance_m(self) -> float:
        return 0.5 * (self.distance_min_m + self.distance_max_m)

    @property
    def reset_x_range_m(self) -> tuple[float, float]:
        """Root-X offsets that realize the requested riser-distance range.

        The terrain origin is placed ``nominal_distance_m`` before the first
        riser. A positive reset-X offset moves the robot closer to the riser.
        """

        nominal = self.nominal_distance_m
        return (
            nominal - self.distance_max_m,
            nominal - self.distance_min_m,
        )


def _pair(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [min, max]")

    low = float(value[0])
    high = float(value[1])

    if low > high:
        raise ValueError(f"{name} must satisfy min <= max")

    if minimum is not None and low < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    if maximum is not None and high > maximum:
        raise ValueError(f"{name} must be <= {maximum}")

    return low, high


def parse_specialist_staging_config(
    resolved: dict[str, Any],
) -> SpecialistStagingCfg | None:
    """Parse the specialist profile without importing Isaac Lab."""

    specialist = resolved.get("specialist_stair")
    if not isinstance(specialist, dict) or not specialist.get("enabled", False):
        return None

    from .runtime_contract import reject_archived_pre_specialist_runtime

    reject_archived_pre_specialist_runtime(resolved)

    initialization = resolved.get("policy_initialization", {})
    if not isinstance(initialization, dict):
        raise ValueError("policy_initialization must be a mapping")
    initialization_type = initialization.get("type")
    allowed_initializations = {
        "random",
        "rsl_rl_checkpoint_actor_transfer",
    }
    if initialization_type not in allowed_initializations:
        raise ValueError(
            "specialist_stair requires policy_initialization.type to be "
            "random or rsl_rl_checkpoint_actor_transfer"
        )

    if initialization_type == "rsl_rl_checkpoint_actor_transfer":
        source = initialization.get("source")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not source["path"].strip()
        ):
            raise ValueError(
                "rsl_rl_checkpoint_actor_transfer requires "
                "policy_initialization.source.path"
            )

        if initialization.get("critic_initialization", "fresh") != "fresh":
            raise ValueError(
                "checkpoint actor transfer requires a fresh critic"
            )

    staging = specialist.get("staging")
    if not isinstance(staging, dict):
        raise ValueError("specialist_stair.staging must be a mapping")

    distance_min, distance_max = _pair(
        staging.get("distance_before_riser_m"),
        name="specialist_stair.staging.distance_before_riser_m",
        minimum=0.10,
        maximum=1.00,
    )
    lateral_min, lateral_max = _pair(
        staging.get("lateral_offset_m", (0.0, 0.0)),
        name="specialist_stair.staging.lateral_offset_m",
        minimum=-0.25,
        maximum=0.25,
    )
    yaw_min, yaw_max = _pair(
        staging.get("yaw_offset_rad", (0.0, 0.0)),
        name="specialist_stair.staging.yaw_offset_rad",
        minimum=-0.35,
        maximum=0.35,
    )

    episode_length_s = specialist.get("episode_length_s")
    if episode_length_s is not None:
        episode_length_s = float(episode_length_s)
        if not 1.0 <= episode_length_s <= 60.0:
            raise ValueError(
                "specialist_stair.episode_length_s must be in [1, 60]"
            )

    return SpecialistStagingCfg(
        distance_min_m=distance_min,
        distance_max_m=distance_max,
        lateral_min_m=lateral_min,
        lateral_max_m=lateral_max,
        yaw_min_rad=yaw_min,
        yaw_max_rad=yaw_max,
        episode_length_s=episode_length_s,
    )


def _configure_specialist_observations(
    env_cfg: Any,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    observations_cfg = resolved.get("observations", {})
    if not isinstance(observations_cfg, dict):
        raise ValueError("observations must be a mapping")

    history_steps = int(
        observations_cfg.get("proprioception_history_steps", 1)
    )
    if not 1 <= history_steps <= 20:
        raise ValueError(
            "observations.proprioception_history_steps must be in [1, 20]"
        )

    include_velocity_command = bool(
        observations_cfg.get("velocity_command", False)
    )

    groups = [
        env_cfg.observations.policy,
        env_cfg.observations.critic,
    ]

    for group in groups:
        for term_name in (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ):
            term = getattr(group, term_name, None)
            if term is not None:
                term.history_length = history_steps

        velocity_term = getattr(group, "velocity_commands", None)
        if velocity_term is not None:
            if include_velocity_command:
                velocity_term.history_length = history_steps
            else:
                group.velocity_commands = None

    critic_base_lin_vel = getattr(
        env_cfg.observations.critic,
        "base_lin_vel",
        None,
    )
    if critic_base_lin_vel is not None:
        critic_base_lin_vel.history_length = history_steps

    return {
        "proprioception_history_steps": history_steps,
        "velocity_command_observation": include_velocity_command,
    }


def apply_specialist_runtime(
    env_cfg: Any,
    resolved: dict[str, Any],
) -> dict[str, Any] | None:
    """Apply the specialist robot, staging, and observation contracts."""

    staging = parse_specialist_staging_config(resolved)
    if staging is None:
        return None

    # Import simulator-dependent configuration only when the specialist path
    # is actually selected and after AppLauncher is already running.
    from .robot_cfg import G1_SPECIALIST_ROBOT_CFG

    env_cfg.scene.robot = G1_SPECIALIST_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    stair_cfg = (
        env_cfg.scene.terrain.terrain_generator
        .sub_terrains["three_step_ascent"]
    )
    stair_cfg.start_clearance = staging.nominal_distance_m

    pose_range = env_cfg.events.reset_base.params["pose_range"]
    pose_range["x"] = staging.reset_x_range_m
    pose_range["y"] = (
        staging.lateral_min_m,
        staging.lateral_max_m,
    )
    pose_range["yaw"] = (
        staging.yaw_min_rad,
        staging.yaw_max_rad,
    )

    # Specialist episodes always begin from rest. Keep the existing joint
    # reset at the exact nominal pose and preserve zero root/joint velocity.
    velocity_range = env_cfg.events.reset_base.params["velocity_range"]
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        velocity_range[key] = (0.0, 0.0)

    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    if staging.episode_length_s is not None:
        env_cfg.episode_length_s = staging.episode_length_s

    observation_summary = _configure_specialist_observations(
        env_cfg,
        resolved,
    )

    return {
        "enabled": True,
        "robot_profile": "deployment_pd_official_usd",
        "policy_initialization": initialization_type,
        "distance_before_riser_m": [
            staging.distance_min_m,
            staging.distance_max_m,
        ],
        "terrain_origin_distance_m": staging.nominal_distance_m,
        "reset_x_range_m": list(staging.reset_x_range_m),
        "lateral_offset_m": [
            staging.lateral_min_m,
            staging.lateral_max_m,
        ],
        "yaw_offset_rad": [
            staging.yaw_min_rad,
            staging.yaw_max_rad,
        ],
        "episode_length_s": env_cfg.episode_length_s,
        **observation_summary,
    }
