"""Opt-in, single-step 043 contract. Importing this module never starts Isaac."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PhysicalStepConfig:
    first_riser_m: float = 1.0
    height_m: float = 0.06
    tread_m: float = 0.25
    width_m: float = 1.2
    continuation_m: float = 0.4
    footprint_margin_m: float = 0.005
    sole_height_tolerance_m: float = 0.015
    support_force_bodyweight_fraction: float = 0.15
    support_speed_mps: float = 0.15
    support_angular_speed_radps: float = 1.0
    support_dwell_s: float = 0.12
    upright_cos: float = 0.90
    foot_normal_cos: float = 0.94
    forbidden_force_n: float = 5.0
    takeoff_distance_m: float = 0.32
    clearance_margin_m: float = 0.025
    touchdown_downward_speed_mps: float = 0.25
    touchdown_horizontal_speed_mps: float = 0.20
    minimum_base_height_m: float = 0.45
    base_height_gain_fraction: float = 0.70

    def validate(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"physical_step.{name} must be positive and finite")
        for name in ("upright_cos", "foot_normal_cos", "base_height_gain_fraction",
                     "support_force_bodyweight_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"physical_step.{name} must not exceed one")
        if self.tread_m <= 0.18 + 2 * self.footprint_margin_m:
            raise ValueError("tread cannot contain the pinned G1 foot footprint")
        if self.sole_height_tolerance_m >= self.height_m / 2:
            raise ValueError("lower and upper support height bands must not overlap")
        return self


def parse_physical_step_config(resolved):
    entry = resolved.get("physical_step", {})
    if not entry.get("enabled", False):
        return None
    config = PhysicalStepConfig(**{k: v for k, v in entry.items() if k != "enabled"}).validate()
    terrain = resolved["terrain"]
    if terrain["step_count"] != 1 or "riser_height_range_m" in terrain:
        raise ValueError("043 requires exactly one fixed-height step")
    for key, expected in (("riser_height_m", config.height_m), ("tread_depth_m", config.tread_m)):
        if not math.isclose(float(terrain[key]), expected, abs_tol=1e-9):
            raise ValueError(f"physical_step and terrain disagree on {key}")
    if not math.isclose(float(terrain["course_size_m"][1]), config.width_m):
        raise ValueError("physical_step and terrain disagree on width")
    if not math.isclose(float(resolved["start_pose"]["distance_before_first_riser_m"]), config.first_riser_m):
        raise ValueError("physical_step and start pose disagree on first riser")
    if not math.isclose(float(terrain["required_platform_continuation_m"]), config.continuation_m):
        raise ValueError("physical_step and terrain disagree on continuation")
    if resolved.get("halfway_reset", {}).get("enabled", False):
        raise ValueError("043 starts from the normal lower-floor approach")
    return config


EVENT_COMPONENTS = ("first_support", "base_transfer", "both_feet", "complete")
BOUNDED_COMPONENTS = ("approach", "clearance", "supported_rise")
RATE_COMPONENTS = ("early_lift", "riser_risk")

# Historical 043 reward contract. Do not extend this tuple:
# tests and old experiment configs intentionally use it as the exact
# active step043 reward set.
CUSTOM_COMPONENTS = (
    EVENT_COMPONENTS
    + BOUNDED_COMPONENTS
    + RATE_COMPONENTS
    + ("touchdown",)
)

# Later experiments may opt into additional physical-step rewards without
# changing the historical 043/043A/043B contract.
EXPERIMENTAL_COMPONENTS = (
    "face_clearance",
    "tread_insertion",
    "pre_crossing_lift",
)

CONFIGURABLE_COMPONENTS = (
    CUSTOM_COMPONENTS
    + EXPERIMENTAL_COMPONENTS
)


def configure_physical_step(env_cfg, resolved, weights):
    """Only config objects and primitive params are created here, never tensors."""
    config = parse_physical_step_config(resolved)
    if config is None:
        return
    from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
    from . import physical_step_runtime as runtime

    regularizers = {"track_stair_velocity_world", "track_ang_vel_z", "upright",
                    "action_rate", "joint_acc", "dof_pos_limits", "feet_slide",
                    "undesired_contacts", "stair_heading_alignment"}
    allowed = regularizers | {
        f"step043_{name}"
        for name in CONFIGURABLE_COMPONENTS
    }
    if set(weights) - allowed:
        raise ValueError(f"043 unexpected rewards: {sorted(set(weights) - allowed)}")
    params = asdict(config)
    for name, term in list(vars(env_cfg.rewards).items()):
        if isinstance(term, RewardTermCfg):
            if name in weights and name in regularizers:
                term.weight = weights[name]
            else:
                setattr(env_cfg.rewards, name, None)
    for name in CONFIGURABLE_COMPONENTS:
        key = f"step043_{name}"
        if key in weights:
            setattr(env_cfg.rewards, key, RewardTermCfg(
                func=runtime.reward, weight=weights[key],
                params={"config": params, "component": name},
            ))
    env_cfg.rewards.track_stair_velocity_world.params["std"] = 0.15
    # The fixed tile's remaining length is the continuous top platform.
    stair = env_cfg.scene.terrain.terrain_generator.sub_terrains["three_step_ascent"]
    if not math.isclose(stair.start_clearance, config.first_riser_m):
        raise ValueError("043 runtime terrain origin disagrees with detector")
    stair.platform_length = float(resolved["terrain"]["platform_length_m"])
    env_cfg.events.step043_reset = EventTermCfg(func=runtime.reset, mode="reset")
    # Snapshot update is driven by termination, even if all achievement weights
    # are zero. Rewards/metrics merely read the same pre-autoreset snapshot.
    env_cfg.terminations.step043_failure = TerminationTermCfg(
        func=runtime.failure, params={"config": params})
    env_cfg.terminations.step043_success = TerminationTermCfg(
        func=runtime.success, params={"config": params})
    env_cfg.episode_length_s = float(resolved["task"]["episode_length_s"])
    if env_cfg.episode_length_s < 15:
        raise ValueError("043 requires at least 15 seconds for approach and transfer")
