"""Apply resolved experiment reward weights to the Isaac Lab environment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .reward_registry import (
    materialize_legacy_reward_config,
    runtime_name,
)


RAW_TO_RUNTIME_TERM = {
    "distance_target_foot_lift_raw": "distance_target_foot_lift",
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
    "pre_riser_foot_lift_raw": "pre_riser_foot_lift",
    "riser_crossing_clearance_raw": "riser_crossing_clearance",
    "riser_face_strike_proxy_raw": "riser_face_strike_proxy",
    "curriculum_foot_over_tread_raw": "curriculum_foot_over_tread",
    "validated_tread_contact_raw": "validated_tread_contact",
    "base_above_transition_raw": "base_above_transition",
    "both_feet_supported_transition_raw": "both_feet_supported_transition",
    "completed_transition_raw": "completed_transition",
}


def load_resolved_config(path: Path) -> dict[str, Any]:
    """Load and normalize a resolved experiment configuration.

    New experiment configs use runtime reward names under
    ``reward_experiment.rewards.stage_plan``.

    The historical ``rewards.terms`` and ``rewards.stage_plan`` structures
    are materialized automatically so existing runtime code remains
    backwards compatible.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(
            f"Resolved configuration must be a mapping: {path}"
        )

    reward_experiment = data.get("reward_experiment")

    # New-style configs provide the experiment-facing reward stage here.
    # Older configs may only contain the historical top-level reward catalog,
    # so leave those unchanged.
    if (
        isinstance(reward_experiment, dict)
        and isinstance(reward_experiment.get("rewards"), dict)
        and reward_experiment["rewards"].get("stage_plan")
    ):
        data = materialize_legacy_reward_config(data)

    return data


def active_reward_weights(resolved: dict[str, Any]) -> dict[str, float]:
    """Resolve cumulative runtime rewards through the declared active stage.

    A reward is valid when either:

    1. it has an explicit historical RAW_TO_RUNTIME_TERM mapping, or
    2. it is declared in the resolved rewards.terms catalog and follows
       the canonical ``*_raw`` naming convention.

    This keeps legacy experiments compatible while allowing new rewards
    to be registered dynamically by reward_registry.
    """
    physical_enabled = resolved.get(
        "physical_step", {}
    ).get("enabled", False)

    structured_enabled = resolved.get(
        "structured_step", {}
    ).get("enabled", False)

    if physical_enabled and structured_enabled:
        raise ValueError(
            "physical_step and structured_step "
            "cannot both be enabled"
        )

    if structured_enabled:
        # 044 is a separate exact-replacement objective.
        from .reward_registry import experiment_stage_weights
        return experiment_stage_weights(resolved)[1]

    if physical_enabled:
        # 043 is a replacement objective, not another cumulative legacy stage.
        from .reward_registry import experiment_stage_weights
        return experiment_stage_weights(resolved)[1]

    try:
        rewards_cfg = resolved["rewards"]
        stage_plan = rewards_cfg["stage_plan"]
        active_stage = resolved["reward_experiment"]["active_stage"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "resolved experiment is missing reward-stage configuration"
        ) from exc

    if not isinstance(rewards_cfg, dict):
        raise ValueError("rewards must be a mapping")

    if not isinstance(stage_plan, dict):
        raise ValueError("rewards.stage_plan must be a mapping")

    declared_terms = rewards_cfg.get("terms", {})

    if not isinstance(declared_terms, dict):
        raise ValueError("rewards.terms must be a mapping")

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

            # Legacy explicitly mapped rewards.
            resolved_name = RAW_TO_RUNTIME_TERM.get(raw_name)

            # New dynamically registered rewards.
            if resolved_name is None:
                if (
                    raw_name.endswith("_raw")
                    and raw_name in declared_terms
                ):
                    resolved_name = runtime_name(raw_name)
                else:
                    raise ValueError(
                        f"active reward {raw_name} "
                        "has no runtime implementation"
                    )

            weights[resolved_name] = float(weight)

        if stage_name == active_stage:
            break

    if not weights:
        raise ValueError(
            f"reward stage {active_stage!r} activates no runtime rewards"
        )

    return weights

def _strict_runtime_reward_set_enabled(resolved: dict[str, Any]) -> bool:
    reward_experiment = resolved.get("reward_experiment", {})
    if not isinstance(reward_experiment, dict):
        return False
    return bool(reward_experiment.get("strict_runtime_set", False))


def _configured_reward_names(rewards_cfg: Any) -> set[str]:
    """Return reward-term attributes currently configured on an env cfg.

    Isaac Lab reward config terms expose both ``func`` and ``weight``.  Using
    that runtime interface keeps pruning independent of experiment numbers or
    a hard-coded catalog of historical reward names.
    """
    names: set[str] = set()
    for name in dir(rewards_cfg):
        if name.startswith("_"):
            continue
        value = getattr(rewards_cfg, name)
        if value is None:
            continue
        if hasattr(value, "func") and hasattr(value, "weight"):
            names.add(name)
    return names


def _apply_strict_runtime_reward_set(
    rewards_cfg: Any,
    weights: dict[str, float],
) -> None:
    """Make the manager-visible reward set exactly match ``weights``.

    Historical configs remain on the compatibility path unless they opt into
    ``reward_experiment.strict_runtime_set``.
    """
    configured = _configured_reward_names(rewards_cfg)
    missing = sorted(set(weights) - configured)
    if missing:
        raise ValueError(
            "resolved experiment activates reward terms missing from "
            f"StairRewardsCfg: {', '.join(missing)}"
        )

    for reward_name in configured:
        term = getattr(rewards_cfg, reward_name)
        if reward_name in weights:
            term.weight = weights[reward_name]
        else:
            setattr(rewards_cfg, reward_name, None)


def apply_resolved_rewards(
    env_cfg: Any,
    resolved: dict[str, Any],
) -> dict[str, float]:
    """Make the runtime reward set exactly match the resolved experiment."""
    weights = active_reward_weights(resolved)

    if _strict_runtime_reward_set_enabled(resolved):
        _apply_strict_runtime_reward_set(env_cfg.rewards, weights)
        return weights

    if resolved.get("structured_step", {}).get("enabled", False):
        from .structured_step044_config import (
            configure_structured_step044,
        )

        configure_structured_step044(
            env_cfg,
            resolved,
            weights,
        )

        return weights

    if resolved.get("physical_step", {}).get("enabled", False):
        from .physical_step_config import configure_physical_step
        configure_physical_step(env_cfg, resolved, weights)
        return weights

    rewards_cfg = env_cfg.rewards

    declared_terms = resolved.get("rewards", {}).get(
        "terms",
        {},
    )

    candidate_names = set(RAW_TO_RUNTIME_TERM.values())

    if isinstance(declared_terms, dict):
        candidate_names.update(
            runtime_name(name)
            for name in declared_terms
            if isinstance(name, str)
            and name.endswith("_raw")
        )

    available = {
        name
        for name in candidate_names
        if hasattr(rewards_cfg, name)
    }

    missing = sorted(set(weights) - available)

    if missing:
        raise ValueError(
            "resolved experiment activates reward terms missing from "
            f"StairRewardsCfg: {', '.join(missing)}"
        )

    for reward_name in available:
        term = getattr(rewards_cfg, reward_name)

        if reward_name in weights:
            if term is None:
                raise ValueError(
                    f"runtime reward {reward_name} is not configured"
                )

            term.weight = weights[reward_name]

        else:
            setattr(
                rewards_cfg,
                reward_name,
                None,
            )

    return weights
