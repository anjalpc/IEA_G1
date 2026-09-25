"""Structured single-step 044 reward contract.

044 is intentionally isolated from the historical 043 runtime and reward
component contract.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class StructuredStep044Config:
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

    clearance_margin_m: float = 0.025
    lift_zone_distance_m: float = 0.25
    meaningful_lift_m: float = 0.04
    riser_ready_distance_m: float = 0.05
    premature_lift_height_m: float = 0.08

    # Reward/state semantics:
    #   1 = historical 044 behavior
    #   2 = body-gated 045 ascent behavior
    #   3 = anti-exploit 047 ordered ascent behavior
    reward_mode: int = 1
    ascent_launch_distance_m: float = 0.25

    touchdown_downward_speed_mps: float = 0.25
    touchdown_horizontal_speed_mps: float = 0.20

    minimum_base_height_m: float = 0.45
    base_height_gain_fraction: float = 0.70

    geometry_tolerance_m: float = 1.0e-6

    def validate(self):
        for name, value in asdict(self).items():
            if (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"structured_step.{name} must be positive and finite"
                )

        for name in (
            "upright_cos",
            "foot_normal_cos",
            "base_height_gain_fraction",
            "support_force_bodyweight_fraction",
        ):
            if getattr(self, name) > 1:
                raise ValueError(
                    f"structured_step.{name} must not exceed one"
                )

        if self.reward_mode not in (1, 2, 3):
            raise ValueError(
                "structured_step.reward_mode must be 1, 2, or 3"
            )

        if self.ascent_launch_distance_m >= self.first_riser_m:
            raise ValueError(
                "ascent launch distance must lie before the first riser"
            )

        nominal_foot_length_m = 0.18
        if self.tread_m <= (
            nominal_foot_length_m
            + 2 * self.footprint_margin_m
        ):
            raise ValueError(
                "tread cannot contain the pinned G1 foot footprint"
            )

        if self.sole_height_tolerance_m >= self.height_m / 2:
            raise ValueError(
                "lower and upper support height bands must not overlap"
            )

        if self.riser_ready_distance_m >= self.lift_zone_distance_m:
            raise ValueError(
                "riser-ready distance must lie inside lift zone"
            )

        return self

    @property
    def required_clearance_m(self):
        return self.height_m + self.clearance_margin_m


def parse_structured_step044_config(resolved):
    entry = resolved.get("structured_step", {})

    if not entry.get("enabled", False):
        return None

    config = StructuredStep044Config(
        **{
            key: value
            for key, value in entry.items()
            if key != "enabled"
        }
    ).validate()

    terrain = resolved["terrain"]

    if (
        terrain["step_count"] != 1
        or "riser_height_range_m" in terrain
    ):
        raise ValueError(
            "044 requires exactly one fixed-height step"
        )

    for key, expected in (
        ("riser_height_m", config.height_m),
        ("tread_depth_m", config.tread_m),
    ):
        if not math.isclose(
            float(terrain[key]),
            expected,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"structured_step and terrain disagree on {key}"
            )

    if not math.isclose(
        float(terrain["course_size_m"][1]),
        config.width_m,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "structured_step and terrain disagree on width"
        )

    if not math.isclose(
        float(
            resolved["start_pose"][
                "distance_before_first_riser_m"
            ]
        ),
        config.first_riser_m,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "structured_step and start pose disagree on first riser"
        )

    if not math.isclose(
        float(
            terrain[
                "required_platform_continuation_m"
            ]
        ),
        config.continuation_m,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "structured_step and terrain disagree on continuation"
        )

    if resolved.get(
        "halfway_reset", {}
    ).get("enabled", False):
        raise ValueError(
            "044 starts from the normal lower-floor approach"
        )

    return config


EVENT_COMPONENTS = (
    "meaningful_lift",
    "riser_ready",
    "valid_crossing",
    "good_touchdown",
    "stable_upper_support",
    "body_transfer",
    "both_feet_supported",
    "complete",
    "low_crossing",
    "clearance_loss",
    "bad_touchdown",
    "failure",
)

BOUNDED_COMPONENTS = (
    "approach",
    "lift_advance",
    "maintained_clearance_advance",
    "tread_insertion",
    "supported_body_rise",
)

RATE_COMPONENTS = (
    "premature_lift",
    "riser_risk",
)

COMPONENTS = (
    EVENT_COMPONENTS
    + BOUNDED_COMPONENTS
    + RATE_COMPONENTS
)


def configure_structured_step044(env_cfg, resolved, weights):
    """Configure the isolated 044 structured single-step objective."""
    config = parse_structured_step044_config(resolved)

    if config is None:
        return

    from isaaclab.managers import (
        EventTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    from . import structured_step044_runtime as runtime

    regularizers = {
        "track_stair_velocity_world",
        "track_ang_vel_z",
        "upright",
        "action_rate",
        "joint_acc",
        "dof_pos_limits",
        "feet_slide",
        "undesired_contacts",
        "stair_heading_alignment",
    }

    allowed = regularizers | {
        f"step044_{name}"
        for name in COMPONENTS
    }

    unexpected = set(weights) - allowed

    if unexpected:
        raise ValueError(
            "044 unexpected rewards: "
            f"{sorted(unexpected)}"
        )

    params = asdict(config)

    # 044 is an exact replacement objective. Disable all existing
    # reward terms except explicitly selected regularizers.
    for name, term in list(vars(env_cfg.rewards).items()):
        if isinstance(term, RewardTermCfg):
            if name in weights and name in regularizers:
                term.weight = weights[name]
            else:
                setattr(env_cfg.rewards, name, None)

    for name in COMPONENTS:
        key = f"step044_{name}"

        if key in weights:
            setattr(
                env_cfg.rewards,
                key,
                RewardTermCfg(
                    func=runtime.reward,
                    weight=weights[key],
                    params={
                        "config": params,
                        "component": name,
                    },
                ),
            )

    env_cfg.rewards.track_stair_velocity_world.params[
        "std"
    ] = 0.15

    if config.reward_mode == 3:
        # Remove the positive per-control-step survival bonus in 047. Perfect
        # commanded-velocity tracking is zero reward; deviations are negative.
        env_cfg.rewards.track_stair_velocity_world.params[
            "zero_centered"
        ] = True

    stair = (
        env_cfg.scene.terrain.terrain_generator
        .sub_terrains["three_step_ascent"]
    )

    if not math.isclose(
        stair.start_clearance,
        config.first_riser_m,
    ):
        raise ValueError(
            "044 runtime terrain origin disagrees with detector"
        )

    stair.platform_length = float(
        resolved["terrain"]["platform_length_m"]
    )

    env_cfg.events.step044_reset = EventTermCfg(
        func=runtime.reset,
        mode="reset",
    )

    env_cfg.terminations.step044_failure = (
        TerminationTermCfg(
            func=runtime.failure,
            params={"config": params},
        )
    )

    env_cfg.terminations.step044_success = (
        TerminationTermCfg(
            func=runtime.success,
            params={"config": params},
        )
    )

    env_cfg.episode_length_s = float(
        resolved["task"]["episode_length_s"]
    )

    if env_cfg.episode_length_s < 15:
        raise ValueError(
            "044 requires at least 15 seconds "
            "for approach and transfer"
        )
