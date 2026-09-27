"""Frozen balance-teacher plus bounded trainable residual control.

This module deliberately sits outside Isaac Lab's ActionManager.  The
specialist environment therefore keeps its validated 29-DoF
JointPositionAction contract while RSL-RL learns only a bounded correction to
a frozen, feedback-controlled balance policy.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .contracts import BODY_JOINTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_actor_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("balance teacher checkpoint is missing model_state_dict")

    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise RuntimeError("balance teacher checkpoint contains no actor tensors")
    return actor_state


def _linear_layer_indices(actor_state: dict[str, torch.Tensor]) -> list[int]:
    indices = []
    for key in actor_state:
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        try:
            index = int(prefix)
        except ValueError:
            continue
        weight = actor_state[key]
        if getattr(weight, "ndim", None) == 2:
            indices.append(index)
    indices = sorted(set(indices))
    if not indices:
        raise RuntimeError("could not resolve linear layers in balance teacher actor")
    return indices


def load_frozen_rsl_actor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct and freeze the deterministic actor from an RSL-RL checkpoint."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("balance teacher checkpoint must contain a mapping")

    actor_state = _extract_actor_state(checkpoint)
    layer_indices = _linear_layer_indices(actor_state)

    modules: list[torch.nn.Module] = []
    expected_next_in = None
    for order, index in enumerate(layer_indices):
        weight = actor_state[f"{index}.weight"]
        bias_key = f"{index}.bias"
        if bias_key not in actor_state:
            raise RuntimeError(f"balance teacher actor is missing {bias_key}")
        bias = actor_state[bias_key]

        out_features, in_features = map(int, weight.shape)
        if tuple(bias.shape) != (out_features,):
            raise RuntimeError(
                f"balance teacher {bias_key} shape mismatch: {tuple(bias.shape)}"
            )
        if expected_next_in is not None and in_features != expected_next_in:
            raise RuntimeError(
                "balance teacher actor layer dimensions are not contiguous: "
                f"expected input {expected_next_in}, got {in_features}"
            )

        linear = torch.nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        modules.append(linear)
        expected_next_in = out_features

        if order != len(layer_indices) - 1:
            modules.append(torch.nn.ELU())

    actor = torch.nn.Sequential(*modules).to(device)
    actor.eval()
    actor.requires_grad_(False)

    first_weight = actor_state[f"{layer_indices[0]}.weight"]
    last_weight = actor_state[f"{layer_indices[-1]}.weight"]
    obs_dim = int(first_weight.shape[1])
    action_dim = int(last_weight.shape[0])
    if action_dim != len(BODY_JOINTS):
        raise RuntimeError(
            f"balance teacher action dimension must be {len(BODY_JOINTS)}, got {action_dim}"
        )

    metadata = {
        "source": str(path),
        "source_sha256": _sha256(path),
        "source_iteration": checkpoint.get("iter"),
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "linear_layers": layer_indices,
        "frozen": True,
    }
    return actor, metadata


def residual_scale_vector(
    scale_by_joint: dict[str, float],
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return scale vector in canonical BODY_JOINTS order."""

    missing = [name for name in BODY_JOINTS if name not in scale_by_joint]
    extra = sorted(set(scale_by_joint) - set(BODY_JOINTS))
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if extra:
            parts.append("unknown: " + ", ".join(extra))
        raise ValueError("invalid residual joint scale map (" + "; ".join(parts) + ")")

    values = [float(scale_by_joint[name]) for name in BODY_JOINTS]
    if any(value < 0.0 for value in values):
        raise ValueError("residual joint scales must be >= 0")

    return torch.tensor(values, device=device, dtype=dtype)


def compose_balance_residual_action(
    teacher_action: torch.Tensor,
    residual_action: torch.Tensor,
    *,
    residual_scale: torch.Tensor,
    residual_clip: float,
    executed_clip: float,
) -> torch.Tensor:
    """Compose teacher + bounded residual in raw 29-D action units."""

    if teacher_action.shape != residual_action.shape:
        raise ValueError(
            "teacher/residual action shape mismatch: "
            f"{tuple(teacher_action.shape)} vs {tuple(residual_action.shape)}"
        )
    if teacher_action.ndim != 2 or teacher_action.shape[1] != len(BODY_JOINTS):
        raise ValueError(
            f"expected teacher/residual actions [N, {len(BODY_JOINTS)}]"
        )
    if residual_scale.shape != (len(BODY_JOINTS),):
        raise ValueError(
            f"residual_scale must have shape ({len(BODY_JOINTS)},)"
        )
    if residual_clip <= 0.0 or executed_clip <= 0.0:
        raise ValueError("residual_clip and executed_clip must be > 0")
    if not torch.isfinite(teacher_action).all():
        raise RuntimeError("balance teacher produced non-finite action")
    if not torch.isfinite(residual_action).all():
        raise RuntimeError("residual policy produced non-finite action")

    bounded_residual = residual_action.clamp(-residual_clip, residual_clip)
    executed = teacher_action + bounded_residual * residual_scale
    return executed.clamp(-executed_clip, executed_clip)


def _policy_observation(observations: Any) -> torch.Tensor:
    """Resolve the actor observation from RSL-RL's observation container.

    RSL-RL 2.3 uses ``TensorDict`` for grouped observations.  TensorDict is
    intentionally handled by its mapping-like ``get`` API rather than by an
    ``isinstance(..., dict)`` check.  The tuple/list fallback keeps this
    wrapper compatible with older adapter layers that may return
    ``(observations, extras)``.
    """

    if isinstance(observations, torch.Tensor):
        return observations

    getter = getattr(observations, "get", None)
    if callable(getter):
        for key in ("policy", "obs"):
            value = getter(key, None)
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(observations, (tuple, list)):
        for value in observations:
            try:
                return _policy_observation(value)
            except RuntimeError:
                continue

    raise RuntimeError(
        "frozen balance residual controller could not resolve policy observation tensor; "
        f"container_type={type(observations).__module__}.{type(observations).__qualname__}"
    )


class FrozenBalanceResidualWrapper:
    """Interpret RSL-RL actions as bounded residuals around a frozen teacher."""

    def __init__(
        self,
        wrapped,
        *,
        teacher_checkpoint: str | Path,
        scale_by_joint: dict[str, float],
        residual_clip: float = 1.0,
        executed_clip: float = 5.0,
    ) -> None:
        self._wrapped = wrapped
        self._teacher, self.teacher_metadata = load_frozen_rsl_actor(
            teacher_checkpoint,
            device=self.device,
        )
        self._scale = residual_scale_vector(
            scale_by_joint,
            device=self.device,
        )
        self._residual_clip = float(residual_clip)
        self._executed_clip = float(executed_clip)
        if self._residual_clip <= 0.0:
            raise ValueError("residual_clip must be > 0")
        if self._executed_clip <= 0.0:
            raise ValueError("executed_clip must be > 0")

        self._last_observations = None
        self.last_teacher_actions = None
        self.last_residual_actions = None
        self.last_executed_actions = None

        print(
            "[INFO] Frozen balance residual controller:",
            {
                **self.teacher_metadata,
                "residual_clip": self._residual_clip,
                "executed_clip": self._executed_clip,
                "scale_by_joint": {
                    name: float(self._scale[index].item())
                    for index, name in enumerate(BODY_JOINTS)
                },
            },
            flush=True,
        )

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def get_observations(self):
        observations = self._wrapped.get_observations()
        self._last_observations = observations
        return observations

    def step(self, residual_actions: torch.Tensor):
        if self._last_observations is None:
            # Defensive fallback for wrappers/runners that call step before an
            # explicit get_observations() after construction/reset.
            self._last_observations = self._wrapped.get_observations()

        policy_obs = _policy_observation(self._last_observations)
        if policy_obs.ndim != 2:
            raise RuntimeError(
                f"policy observations must be rank-2, got {tuple(policy_obs.shape)}"
            )
        expected_obs = int(self.teacher_metadata["observation_dim"])
        if policy_obs.shape[1] != expected_obs:
            raise RuntimeError(
                "balance teacher observation dimension mismatch: "
                f"teacher={expected_obs}, runtime={policy_obs.shape[1]}"
            )

        with torch.no_grad():
            teacher_actions = self._teacher(policy_obs)

        executed_actions = compose_balance_residual_action(
            teacher_actions,
            residual_actions,
            residual_scale=self._scale,
            residual_clip=self._residual_clip,
            executed_clip=self._executed_clip,
        )

        self.last_teacher_actions = teacher_actions.detach()
        self.last_residual_actions = residual_actions.detach()
        self.last_executed_actions = executed_actions.detach()

        # Make the decomposition available to read-only diagnostics/rewards
        # without changing the ActionManager contract.
        unwrapped = self.unwrapped
        unwrapped._g1_balance_teacher_action = self.last_teacher_actions
        unwrapped._g1_balance_residual_action = self.last_residual_actions
        unwrapped._g1_balance_executed_action = self.last_executed_actions

        result = self._wrapped.step(executed_actions)
        self._last_observations = result[0]
        return result


def wrap_balance_residual_from_config(wrapped, resolved_config: dict[str, Any]):
    """Wrap an RSL-RL env when frozen-balance residual control is enabled."""

    cfg = resolved_config.get("balance_residual_control", {})
    if not isinstance(cfg, dict):
        raise ValueError("balance_residual_control must be a mapping")
    if not cfg.get("enabled", False):
        return wrapped

    teacher = cfg.get("teacher_checkpoint")
    scale_by_joint = cfg.get("scale_by_joint")
    if not isinstance(teacher, str) or not teacher.strip():
        raise ValueError(
            "balance_residual_control.teacher_checkpoint is required"
        )
    if not isinstance(scale_by_joint, dict):
        raise ValueError(
            "balance_residual_control.scale_by_joint must be a mapping"
        )

    return FrozenBalanceResidualWrapper(
        wrapped,
        teacher_checkpoint=teacher,
        scale_by_joint=scale_by_joint,
        residual_clip=float(cfg.get("residual_clip", 1.0)),
        executed_clip=float(cfg.get("executed_clip", 5.0)),
    )
