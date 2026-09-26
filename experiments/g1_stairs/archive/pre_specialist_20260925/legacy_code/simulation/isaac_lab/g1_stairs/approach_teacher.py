"""Training-only preservation of the original Unitree approach gait."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import BODY_JOINTS


def _build_action_groups() -> dict[str, tuple[int, ...]]:
    """Build exhaustive action groups from the canonical 29-joint order."""

    groups = {
        "legs": [],
        "arms": [],
        "waist": [],
    }

    for index, name in enumerate(BODY_JOINTS):
        if any(
            token in name
            for token in (
                "_hip_",
                "_knee_",
                "_ankle_",
            )
        ):
            groups["legs"].append(index)
        elif any(
            token in name
            for token in (
                "_shoulder_",
                "_elbow_",
                "_wrist_",
            )
        ):
            groups["arms"].append(index)
        elif name.startswith("waist_"):
            groups["waist"].append(index)
        else:
            raise RuntimeError(
                "unclassified G1 action joint: "
                f"{name}"
            )

    result = {
        name: tuple(indices)
        for name, indices in groups.items()
    }

    flattened = [
        index
        for indices in result.values()
        for index in indices
    ]

    if sorted(flattened) != list(
        range(len(BODY_JOINTS))
    ):
        raise RuntimeError(
            "G1 diagnostic action groups must form an "
            "exact partition of BODY_JOINTS"
        )

    return result


ACTION_GROUP_INDICES = _build_action_groups()


@dataclass(frozen=True)
class ApproachTeacherConfig:
    coefficient: float
    release_near_m: float
    release_far_m: float
    huber_delta: float
    enforce_action_blend: bool = False
    gate_task_actor_loss: bool = False


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
        enforce_action_blend=bool(
            raw.get("enforce_action_blend", False)
        ),
        gate_task_actor_loss=bool(
            raw.get("gate_task_actor_loss", False)
        ),
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


def blend_teacher_student_actions(
    student: torch.Tensor,
    teacher: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Map policy actions to executed actions during stair approach.

    gate == 1 -> exact frozen Unitree teacher action
    gate == 0 -> exact student action
    otherwise -> convex teacher/student blend
    """

    if student.shape != teacher.shape:
        raise RuntimeError(
            "student/teacher action shapes differ: "
            f"{tuple(student.shape)} vs {tuple(teacher.shape)}"
        )

    if student.ndim != 2 or student.shape[1] != len(BODY_JOINTS):
        raise RuntimeError(
            "expected student/teacher actions [N, 29], got "
            f"{tuple(student.shape)}"
        )

    if gate.shape != (student.shape[0], 1):
        raise RuntimeError(
            "expected teacher gate "
            f"{(student.shape[0], 1)}, got {tuple(gate.shape)}"
        )

    if not (
        torch.isfinite(student).all()
        and torch.isfinite(teacher).all()
        and torch.isfinite(gate).all()
    ):
        raise RuntimeError(
            "teacher action blend received non-finite values"
        )

    if bool(((gate < 0.0) | (gate > 1.0)).any()):
        raise RuntimeError(
            "teacher action blend gate must lie in [0, 1]"
        )

    return (
        gate * teacher
        + (1.0 - gate) * student
    )


def student_control_authority(
    gate: torch.Tensor,
) -> torch.Tensor:
    """Return student execution authority as shape [N].

    The approach gate is teacher authority: one far away and zero near the
    riser. Student authority is therefore its complement. This helper is used
    only when an experiment explicitly opts into control-aware actor updates.
    """

    if gate.ndim != 2 or gate.shape[1] != 1:
        raise RuntimeError(
            "expected teacher gate [N, 1], got "
            f"{tuple(gate.shape)}"
        )
    if not torch.isfinite(gate).all():
        raise RuntimeError(
            "student control authority received non-finite gate"
        )
    if bool(((gate < 0.0) | (gate > 1.0)).any()):
        raise RuntimeError(
            "teacher gate must lie in [0, 1]"
        )
    return 1.0 - gate.squeeze(1)


def control_aware_actor_mean(
    values: torch.Tensor,
    gate: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """Average actor-only loss with optional student-authority weighting."""

    if values.ndim != 1:
        raise RuntimeError(
            "expected per-sample actor values [N], got "
            f"{tuple(values.shape)}"
        )
    authority = student_control_authority(gate)
    if authority.shape[0] != values.shape[0]:
        raise RuntimeError(
            "actor values/gate batch sizes differ"
        )
    if not torch.isfinite(values).all():
        raise RuntimeError(
            "actor loss values must be finite"
        )
    if not enabled:
        return values.mean()
    return (values * authority).mean()


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


def action_group_mean_abs(
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mean absolute sampled action for each canonical body group."""

    if (
        actions.ndim != 2
        or actions.shape[1] != len(BODY_JOINTS)
    ):
        raise RuntimeError(
            "expected G1 actions [N, 29], got "
            f"{tuple(actions.shape)}"
        )

    if not torch.isfinite(actions).all():
        raise RuntimeError(
            "G1 actions are non-finite"
        )

    return {
        name: actions[
            :, indices
        ].abs().mean()
        for name, indices
        in ACTION_GROUP_INDICES.items()
    }


def teacher_diagnostic_terms(
    student_mean: torch.Tensor,
    teacher_action: torch.Tensor,
    gate: torch.Tensor,
    delta: float,
) -> dict[str, torch.Tensor]:
    """Return read-only teacher diagnostics aligned to PPO samples."""

    if student_mean.shape != teacher_action.shape:
        raise RuntimeError(
            "student/teacher action shapes differ: "
            f"{tuple(student_mean.shape)} vs "
            f"{tuple(teacher_action.shape)}"
        )

    if (
        student_mean.ndim != 2
        or student_mean.shape[1] != len(BODY_JOINTS)
    ):
        raise RuntimeError(
            "expected student/teacher actions [N, 29], got "
            f"{tuple(student_mean.shape)}"
        )

    if (
        gate.ndim != 2
        or gate.shape != (
            student_mean.shape[0],
            1,
        )
    ):
        raise RuntimeError(
            "expected teacher gate [N, 1], got "
            f"{tuple(gate.shape)}"
        )

    if not (
        torch.isfinite(student_mean).all()
        and torch.isfinite(teacher_action).all()
        and torch.isfinite(gate).all()
    ):
        raise RuntimeError(
            "teacher diagnostics received non-finite values"
        )

    gate_1d = gate.squeeze(1)

    # smooth_release_gate is exactly:
    #   1 far from the stair
    #   0 near the stair
    #   strictly between 0 and 1 in the release band
    epsilon = 1.0e-6

    far = gate_1d >= 1.0 - epsilon
    near = gate_1d <= epsilon
    release = ~(far | near)

    result = {
        "error": smooth_l1_per_environment(
            student_mean,
            teacher_action,
            delta,
        ),
        "far_mask": far,
        "release_mask": release,
        "near_mask": near,
    }

    for name, indices in (
        ACTION_GROUP_INDICES.items()
    ):
        result[
            f"{name}_error"
        ] = smooth_l1_per_environment(
            student_mean[:, indices],
            teacher_action[:, indices],
            delta,
        )

    return result


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
