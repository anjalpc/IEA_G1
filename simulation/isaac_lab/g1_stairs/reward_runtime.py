"""Apply resolved experiment reward weights to the Isaac Lab environment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


RAW_TO_RUNTIME_TERM = {
    "forward_progress_raw": "forward_progress",
    "stair_progress_raw": "stair_progress",
    "upright_raw": "upright",
    "platform_entry_raw": "platform_entry",
    "task_completion_raw": "task_completion",
    "track_lin_vel_xy_raw": "track_lin_vel_xy",
    "action_rate_raw": "action_rate",
    "waist_deviation_raw": "waist_deviation",
    "stair_swing_clearance_raw": "stair_swing_clearance",
    "anticipatory_swing_clearance_raw": "anticipatory_swing_clearance",
    "contact_gated_stair_clearance_raw": "contact_gated_stair_clearance",
    "curriculum_riser_clearance_raw": "curriculum_riser_clearance",
    "curriculum_foot_over_tread_raw": "curriculum_foot_over_tread",
    "validated_tread_contact_raw": "validated_tread_contact",
    "base_above_transition_raw": "base_above_transition",
    "both_feet_supported_transition_raw": "both_feet_supported_transition",
    "completed_transition_raw": "completed_transition",
}


def load_resolved_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def active_reward_weights(resolved: dict[str, Any]) -> dict[str, float]:
    """Resolve cumulative runtime rewards through the declared active stage."""
    try:
        stage_plan = resolved["rewards"]["stage_plan"]
        active_stage = resolved["reward_experiment"]["active_stage"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "resolved experiment is missing reward-stage configuration"
        ) from exc

    if not isinstance(stage_plan, dict):
        raise ValueError("rewards.stage_plan must be a mapping")

    stage_names = list(stage_plan)

    if active_stage not in stage_names:
        raise ValueError(
            f"unknown active reward stage: {active_stage!r}"
        )

    weights: dict[str, float] = {}

    for stage_name in stage_names:
        stage = stage_plan[stage_name]

        if not isinstance(stage, dict):
            raise ValueError(
                f"invalid reward stage specification: {stage_name}"
            )

        enabled = stage.get("enabled_terms", {})

        if not isinstance(enabled, dict):
            raise ValueError(
                f"{stage_name}.enabled_terms must be a mapping"
            )

        for raw_name, weight in enabled.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"active reward {raw_name} in {stage_name} "
                    f"has non-numeric weight: {weight!r}"
                )

            runtime_name = RAW_TO_RUNTIME_TERM.get(raw_name)

            if runtime_name is None:
                raise ValueError(
                    f"active reward {raw_name} has no runtime implementation"
                )

            weights[runtime_name] = float(weight)

        if stage_name == active_stage:
            break

    if not weights:
        raise ValueError(
            f"reward stage {active_stage!r} activates no runtime rewards"
        )

    return weights


def apply_resolved_rewards(env_cfg: Any, resolved: dict[str, Any]) -> dict[str, float]:
    """Make the runtime reward set exactly match the resolved experiment."""
    weights = active_reward_weights(resolved)

    rewards_cfg = env_cfg.rewards

    available = {
        name
        for name in RAW_TO_RUNTIME_TERM.values()
        if hasattr(rewards_cfg, name)
    }

    missing = sorted(set(weights) - available)
    if missing:
        raise ValueError(
            "resolved experiment activates reward terms missing from "
            f"StairRewardsCfg: {', '.join(missing)}"
        )

    for runtime_name in available:
        term = getattr(rewards_cfg, runtime_name)

        if runtime_name in weights:
            if term is None:
                raise ValueError(
                    f"runtime reward {runtime_name} is not configured"
                )
            term.weight = weights[runtime_name]
        else:
            setattr(rewards_cfg, runtime_name, None)

    return weights
