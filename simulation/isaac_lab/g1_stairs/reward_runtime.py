"""Apply resolved experiment reward weights to the Isaac Lab environment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .reward_registry import (
    materialize_legacy_reward_config,
    runtime_name,
)


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

    Active rewards are declared by experiment YAML and normalized through the
    canonical ``*_raw`` naming convention. Historical numbered reward modes
    are archived and rejected by the active runtime boundary.
    """
    from .runtime_contract import reject_archived_pre_specialist_runtime

    reject_archived_pre_specialist_runtime(resolved)

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

    # Historical experiments accumulate reward stages in declaration order.
    # A strict experiment instead declares one exact manager-visible reward
    # set and must not inherit weights from earlier historical stages.
    reward_experiment = resolved.get("reward_experiment", {})
    strict_runtime_set = (
        isinstance(reward_experiment, dict)
        and bool(
            reward_experiment.get(
                "strict_runtime_set",
                False,
            )
        )
    )

    stages_to_apply = (
        [active_stage]
        if strict_runtime_set
        else stage_names
    )

    weights: dict[str, float] = {}

    for stage_name in stages_to_apply:
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



def _apply_reward_parameter_overrides(
    rewards_cfg: Any,
    weights: dict[str, float],
    resolved: dict[str, Any],
) -> None:
    """Apply recorded scalar parameter overrides to active reward terms only.

    This keeps reward geometry in experiment YAML instead of hiding it in
    experiment-number-specific Python code. Existing SceneEntityCfg parameters
    remain untouched; only keys already declared by the runtime term may be
    overridden.
    """
    overrides = resolved.get("reward_parameters", {})
    if overrides is None:
        return
    if not isinstance(overrides, dict):
        raise ValueError("reward_parameters must be a mapping")

    # Child experiment configs may explicitly set an inherited reward-parameter
    # block to YAML null to remove it when the corresponding reward is no longer
    # active. Keep strict validation for every non-null inactive override.
    overrides = {
        reward_name: params
        for reward_name, params in overrides.items()
        if params is not None
    }

    unknown_terms = sorted(set(overrides) - set(weights))
    if unknown_terms:
        raise ValueError(
            "reward_parameters may only override active rewards: "
            + ", ".join(unknown_terms)
        )

    for reward_name, params in overrides.items():
        if not isinstance(params, dict):
            raise ValueError(
                f"reward_parameters.{reward_name} must be a mapping"
            )
        term = getattr(rewards_cfg, reward_name, None)
        if term is None or not hasattr(term, "params"):
            raise ValueError(
                f"reward parameter override has no runtime term: {reward_name}"
            )

        unknown_params = sorted(set(params) - set(term.params))
        if unknown_params:
            raise ValueError(
                f"reward_parameters.{reward_name} contains undeclared keys: "
                + ", ".join(unknown_params)
            )

        for key, value in params.items():
            if not isinstance(value, (int, float, bool, str)):
                raise ValueError(
                    f"reward_parameters.{reward_name}.{key} must be scalar"
                )
            term.params[key] = value

def gait_reward_contract_required(resolved: dict[str, Any]) -> bool:
    """Return whether C3/C4 gait reward invariants apply to this run."""
    gait = resolved.get("gait", {})
    return bool(
        gait.get("enabled", False)
        and not gait.get("preservation_mode", False)
    )


def apply_resolved_rewards(
    env_cfg: Any,
    resolved: dict[str, Any],
) -> dict[str, float]:
    """Make the runtime reward set exactly match the resolved experiment."""
    if resolved.get("gait", {}).get("enabled", False):
        from .gait_runtime import configure_gait
        configure_gait(env_cfg, resolved)
    weights = active_reward_weights(resolved)
    if gait_reward_contract_required(resolved):
        c4 = resolved.get("gait", {}).get("coordinated_motion", False)
        expected_contact_weight = 0.0 if c4 else -0.10
        if weights.get("phase_swing_contact") != expected_contact_weight:
            raise ValueError("phase_swing_contact weight disagrees with C3/C4 mode")
        if c4 and weights.get("coordinated_motion") != -1.0:
            raise ValueError("C4 requires coordinated_motion=-1.0")
        if not weights.get("terrain_relative_clearance"):
            raise ValueError("C3 requires the existing acquisition update")

    if _strict_runtime_reward_set_enabled(resolved):
        _apply_strict_runtime_reward_set(env_cfg.rewards, weights)
        _apply_reward_parameter_overrides(
            env_cfg.rewards,
            weights,
            resolved,
        )

        # Potential-based clearance must use the exact PPO discount. Both
        # acquisition terms share one runtime update, so their accounting
        # parameters must agree regardless of reward-manager evaluation order.
        clearance_term = getattr(
            env_cfg.rewards, "terrain_relative_clearance", None
        )
        insertion_term = getattr(
            env_cfg.rewards, "forward_insertion", None
        )
        if (
            clearance_term is not None
            and clearance_term.params.get(
                "reversible_clearance_potential", False
            )
        ):
            gamma = resolved.get("ppo", {}).get("discount_gamma")
            if gamma is None:
                raise ValueError(
                    "reversible clearance potential requires "
                    "ppo.discount_gamma"
                )
            gamma = float(gamma)
            for name, term in (
                ("terrain_relative_clearance", clearance_term),
                ("forward_insertion", insertion_term),
            ):
                if term is None:
                    raise ValueError(
                        "reversible clearance potential requires both "
                        "acquisition reward terms"
                    )
                if not term.params.get(
                    "reversible_clearance_potential", False
                ):
                    raise ValueError(
                        f"{name} must enable reversible_clearance_potential"
                    )
                configured_gamma = float(
                    term.params["clearance_potential_discount_gamma"]
                )
                if abs(configured_gamma - gamma) > 1.0e-12:
                    raise ValueError(
                        f"{name} clearance potential gamma "
                        f"{configured_gamma} != PPO gamma {gamma}"
                    )

        return weights

    rewards_cfg = env_cfg.rewards

    declared_terms = resolved.get("rewards", {}).get(
        "terms",
        {},
    )

    candidate_names: set[str] = set()

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

    _apply_reward_parameter_overrides(
        rewards_cfg,
        weights,
        resolved,
    )
    return weights
