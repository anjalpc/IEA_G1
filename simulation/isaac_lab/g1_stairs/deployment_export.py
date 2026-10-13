"""Project-side deployment export compatibility for internal C3 gait actions."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import yaml

from .gait_phase import (
    FREQUENCY_MAX_HZ,
    FREQUENCY_MIN_HZ,
    FREQUENCY_NOMINAL_HZ,
    PHASE_OFFSET,
)


_PHYSICAL_ACTION_TERM = "JointPositionAction"
_INTERNAL_GAIT_TERM = "gait_frequency"


class _ActionManagerExportView:
    """Expose only physical terms to Unitree's joint-action exporter.

    Runtime lookups still delegate to the live manager so observation export can
    query the internal gait term without mutating the training environment.
    """

    def __init__(self, manager, exported_terms: tuple[str, ...]):
        self._manager = manager
        self._exported_terms = exported_terms

    @property
    def active_terms(self):
        return list(self._exported_terms)

    @property
    def _terms(self):
        return OrderedDict(
            (name, self._manager.get_term(name))
            for name in self._exported_terms
        )

    def get_term(self, name):
        return self._manager.get_term(name)

    def __getattr__(self, name):
        return getattr(self._manager, name)


class _EnvExportView:
    def __init__(self, env, action_manager):
        self._env = env
        self.action_manager = action_manager

    def __getattr__(self, name):
        return getattr(self._env, name)


def _write_gait_metadata(log_dir: str | Path, resolved_config: dict) -> Path:
    gait = resolved_config.get("gait", {})
    actions = resolved_config.get("actions", {})
    policy_action_dim = int(actions.get("dimension", 30))
    physical_action_dim = int(actions.get("physical_joint_dimension", 29))

    metadata = {
        "enabled": True,
        "mode": gait.get("mode"),
        "internal_action_term": _INTERNAL_GAIT_TERM,
        "frequency_min_hz": FREQUENCY_MIN_HZ,
        "frequency_nominal_hz": FREQUENCY_NOMINAL_HZ,
        "frequency_max_hz": FREQUENCY_MAX_HZ,
        "phase_offset": PHASE_OFFSET,
        "phase_observation": ["sin", "cos"],
        "physical_action_dim": physical_action_dim,
        "policy_action_dim": policy_action_dim,
    }

    path = Path(log_dir) / "params" / "gait.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return path


def export_deploy_cfg_compatible(
    env,
    log_dir: str | Path,
    resolved_config: dict,
    unitree_exporter: Callable,
) -> None:
    """Run Unitree's exporter without presenting C3's internal gait action.

    Unitree's export_deploy_cfg assumes every active action term is a joint
    actuator term. C3's gait-frequency action is policy/controller state, not a
    robot actuator, so only the physical joint term is exposed to that exporter.
    The live ActionManager is never mutated.
    """

    gait = resolved_config.get("gait", {})
    if not gait.get("enabled", False):
        unitree_exporter(env, str(log_dir))
        return

    manager = env.action_manager
    before_terms = list(manager.active_terms)
    before_dim = int(manager.total_action_dim)
    expected_terms = [_PHYSICAL_ACTION_TERM, _INTERNAL_GAIT_TERM]
    if before_terms != expected_terms:
        raise RuntimeError(
            "C3 deployment export expected action terms "
            f"{expected_terms}, got {before_terms}"
        )

    actions_cfg = resolved_config.get("actions", {})
    physical_dim = int(actions_cfg.get("physical_joint_dimension", -1))
    policy_dim = int(actions_cfg.get("dimension", -1))
    if physical_dim != 29 or policy_dim != 30 or before_dim != 30:
        raise RuntimeError(
            "C3 deployment export requires physical/policy dimensions 29/30; "
            f"resolved={physical_dim}/{policy_dim}, runtime={before_dim}"
        )
    if manager.get_term(_PHYSICAL_ACTION_TERM).action_dim != physical_dim:
        raise RuntimeError("C3 physical action term is not 29-D")
    if manager.get_term(_INTERNAL_GAIT_TERM).action_dim != 1:
        raise RuntimeError("C3 gait-frequency action term is not 1-D")

    action_view = _ActionManagerExportView(
        manager, (_PHYSICAL_ACTION_TERM,)
    )
    env_view = _EnvExportView(env, action_view)
    unitree_exporter(env_view, str(log_dir))

    # Fail closed if export plumbing mutated the live training contract.
    if list(manager.active_terms) != before_terms:
        raise RuntimeError("deployment export mutated live action terms")
    if int(manager.total_action_dim) != before_dim:
        raise RuntimeError("deployment export mutated live action dimension")

    deploy_path = Path(log_dir) / "params" / "deploy.yaml"
    if not deploy_path.is_file():
        raise RuntimeError("Unitree exporter did not create params/deploy.yaml")
    with deploy_path.open("r", encoding="utf-8") as stream:
        deploy_cfg = yaml.safe_load(stream) or {}
    exported_actions = deploy_cfg.get("actions", {})
    if list(exported_actions) != [_PHYSICAL_ACTION_TERM]:
        raise RuntimeError(
            "deploy.yaml must contain only the physical joint action; got "
            f"{list(exported_actions)}"
        )

    _write_gait_metadata(log_dir, resolved_config)
