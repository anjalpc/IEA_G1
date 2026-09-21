"""Reward-name normalization and experiment reward configuration.

This module provides one canonical translation between experiment-facing
reward names and the historical ``*_raw`` configuration names.

Experiment configs should use runtime names such as::

    pre_riser_foot_lift

Legacy resolved configs using::

    pre_riser_foot_lift_raw

remain supported.
"""

from __future__ import annotations


RAW_SUFFIX = "_raw"


def runtime_name(name: str) -> str:
    """Return canonical Isaac/runtime reward name."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid reward name: {name!r}")

    if name.endswith(RAW_SUFFIX):
        return name[: -len(RAW_SUFFIX)]

    return name


def raw_name(name: str) -> str:
    """Return historical raw configuration name."""
    canonical = runtime_name(name)
    return f"{canonical}{RAW_SUFFIX}"


def normalize_reward_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize a reward mapping to canonical runtime names."""
    if not isinstance(weights, dict):
        raise TypeError(
            f"Reward weights must be a mapping, got {type(weights).__name__}"
        )

    normalized: dict[str, float] = {}

    for name, weight in weights.items():
        canonical = runtime_name(name)

        if canonical in normalized:
            raise ValueError(
                f"Duplicate reward after normalization: {canonical}"
            )

        normalized[canonical] = float(weight)

    return normalized


def raw_reward_weights(weights: dict[str, float]) -> dict[str, float]:
    """Convert reward mapping to legacy ``*_raw`` names."""
    normalized = normalize_reward_weights(weights)

    return {
        raw_name(name): weight
        for name, weight in normalized.items()
    }


def experiment_stage_weights(
    config: dict,
) -> tuple[str, dict[str, float]]:
    """Resolve active reward stage from experiment-facing configuration.

    Expected preferred schema::

        reward_experiment:
          active_stage: my_stage
          rewards:
            stage_plan:
              my_stage:
                upright: -5.0

    ``enabled_terms`` is also accepted for backwards compatibility.
    """
    reward_exp = config.get("reward_experiment") or {}

    stage = reward_exp.get("active_stage")
    if not stage:
        raise ValueError(
            "reward_experiment.active_stage is required"
        )

    reward_cfg = reward_exp.get("rewards") or {}
    stage_plan = reward_cfg.get("stage_plan") or {}

    if stage not in stage_plan:
        raise ValueError(
            f"Reward stage {stage!r} not found under "
            "reward_experiment.rewards.stage_plan"
        )

    stage_entry = stage_plan[stage]

    if not isinstance(stage_entry, dict):
        raise TypeError(
            f"Reward stage {stage!r} must be a mapping"
        )

    # Accept both:
    #   stage_name:
    #       upright: -5
    #
    # and the old:
    #   stage_name:
    #       enabled_terms:
    #           upright: -5
    weights = stage_entry.get("enabled_terms", stage_entry)

    return stage, normalize_reward_weights(weights)


def materialize_legacy_reward_config(config: dict) -> dict:
    """Populate historical reward catalog from reward_experiment.

    This makes old training/runtime code compatible with the newer
    single-source experiment configuration.

    Existing unrelated metadata under ``rewards`` is preserved.
    """
    stage, runtime_weights = experiment_stage_weights(config)

    raw_weights = raw_reward_weights(runtime_weights)

    legacy = config.setdefault("rewards", {})

    terms = legacy.setdefault("terms", {})

    for name, weight in raw_weights.items():
        entry = terms.setdefault(name, {})
        entry["active_weight"] = weight

    stage_plan = legacy.setdefault("stage_plan", {})
    stage_plan[stage] = {
        "enabled_terms": dict(raw_weights)
    }

    return config
