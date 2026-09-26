"""Generic action diagnostics for the 29-DoF G1 policy contract."""

from __future__ import annotations

import torch

from .contracts import BODY_JOINTS


def _build_action_groups() -> dict[str, tuple[int, ...]]:
    """Build an exhaustive partition of the canonical 29 action channels."""
    groups = {"legs": [], "arms": [], "waist": []}

    for index, name in enumerate(BODY_JOINTS):
        if any(token in name for token in ("_hip_", "_knee_", "_ankle_")):
            groups["legs"].append(index)
        elif any(token in name for token in ("_shoulder_", "_elbow_", "_wrist_")):
            groups["arms"].append(index)
        elif name.startswith("waist_"):
            groups["waist"].append(index)
        else:
            raise RuntimeError(f"unclassified G1 action joint: {name}")

    result = {name: tuple(indices) for name, indices in groups.items()}
    flattened = [index for indices in result.values() for index in indices]
    if sorted(flattened) != list(range(len(BODY_JOINTS))):
        raise RuntimeError(
            "G1 diagnostic action groups must form an exact partition of BODY_JOINTS"
        )
    return result


ACTION_GROUP_INDICES = _build_action_groups()


def action_group_mean_abs(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return mean absolute action magnitude for legs, arms, and waist."""
    if actions.ndim != 2 or actions.shape[1] != len(BODY_JOINTS):
        raise RuntimeError(f"expected G1 actions [N, 29], got {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("G1 actions are non-finite")
    return {
        name: actions[:, indices].abs().mean()
        for name, indices in ACTION_GROUP_INDICES.items()
    }
