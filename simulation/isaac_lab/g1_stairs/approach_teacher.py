"""Training-only preservation of the original Unitree approach gait."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ApproachTeacherConfig:
    coefficient: float
    release_near_m: float
    release_far_m: float
    huber_delta: float


def parse_approach_teacher_config(
    resolved: dict[str, Any],
) -> ApproachTeacherConfig | None:
    raw = resolved.get("approach_teacher", {})
    if not raw.get("enabled", False):
        return None
    cfg = ApproachTeacherConfig(
        coefficient=float(raw["coefficient"]),
        release_near_m=float(raw["release_near_m"]),
        release_far_m=float(raw["release_far_m"]),
        huber_delta=float(raw["huber_delta"]),
    )
    if cfg.coefficient <= 0.0:
        raise ValueError("approach_teacher.coefficient must be positive")
    if not 0.0 <= cfg.release_near_m < cfg.release_far_m:
        raise ValueError(
            "approach_teacher release distances must satisfy 0 <= near < far"
        )
    if cfg.huber_delta <= 0.0:
        raise ValueError("approach_teacher.huber_delta must be positive")
    return cfg


def smooth_release_gate(
    distance_m: torch.Tensor,
    near_m: float,
    far_m: float,
) -> torch.Tensor:
    """One far away, zero near the riser, cubic between the limits."""
    alpha = ((distance_m - near_m) / (far_m - near_m)).clamp(0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def smooth_l1_per_environment(
    student: torch.Tensor,
    teacher: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    error = torch.abs(student - teacher)
    per_joint = torch.where(
        error < delta,
        0.5 * error.square() / delta,
        error - 0.5 * delta,
    )
    return per_joint.mean(dim=1)


def enable_approach_teacher_observations(env_cfg) -> None:
    """Add a separate teacher group; never infer it by slicing actor input."""
    teacher_cls = env_cfg.observations.__class__.ApproachTeacherCfg
    env_cfg.observations.approach_teacher = teacher_cls()
    gate_cls = env_cfg.observations.__class__.ApproachTeacherGateCfg
    env_cfg.observations.approach_teacher_gate = gate_cls()


def configure_approach_teacher_gate(env_cfg, config: ApproachTeacherConfig) -> None:
    term = env_cfg.observations.approach_teacher_gate.gate
    term.params["release_near_m"] = config.release_near_m
    term.params["release_far_m"] = config.release_far_m


def teacher_observation(observations) -> torch.Tensor:
    try:
        value = observations["approach_teacher"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("approach_teacher observation group is missing") from exc
    if value.ndim != 2 or value.shape[1] != 480:
        raise RuntimeError(
            "expected approach_teacher observations [N, 480], got "
            f"{tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise RuntimeError("approach_teacher observations are non-finite")
    return value


def teacher_gate(observations) -> torch.Tensor:
    try:
        value = observations["approach_teacher_gate"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("approach_teacher_gate observation is missing") from exc
    if value.ndim != 2 or value.shape[1] != 1:
        raise RuntimeError(
            "expected approach_teacher_gate [N, 1], got "
            f"{tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise RuntimeError("approach_teacher_gate is non-finite")
    return value


def gated_teacher_loss(
    student_mean: torch.Tensor,
    teacher_action: torch.Tensor,
    gate: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    per_environment = smooth_l1_per_environment(
        student_mean,
        teacher_action,
        delta,
    )
    return (per_environment * gate.squeeze(1)).mean()
