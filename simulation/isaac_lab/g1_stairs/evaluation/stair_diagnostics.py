"""Streaming statistics used by stair-training diagnostics.

This module is intentionally reward- and policy-agnostic.
It accumulates diagnostic tensor statistics without changing
values consumed by PPO, rewards, terminations, or the physical
success detector.
"""

from __future__ import annotations

import math


class TensorStatsAccumulator:
    """Accumulate finite tensor statistics using tensor reductions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._sum = None
        self._sum_sq = None
        self._abs_sum = None
        self._minimum = None
        self._maximum = None
        self._nonzero = None
        self._positive = None
        self._negative = None

    def update(self, value) -> None:
        import torch

        flat = value.detach().reshape(-1)

        if flat.numel() == 0:
            return

        batch_count = flat.numel()
        batch_sum = flat.sum()
        batch_sum_sq = flat.square().sum()
        batch_abs_sum = flat.abs().sum()
        batch_min = flat.min()
        batch_max = flat.max()
        batch_nonzero = torch.count_nonzero(flat)
        batch_positive = torch.count_nonzero(flat > 0)
        batch_negative = torch.count_nonzero(flat < 0)

        self._count += batch_count

        if self._sum is None:
            self._sum = batch_sum
            self._sum_sq = batch_sum_sq
            self._abs_sum = batch_abs_sum
            self._minimum = batch_min
            self._maximum = batch_max
            self._nonzero = batch_nonzero
            self._positive = batch_positive
            self._negative = batch_negative
            return

        self._sum = self._sum + batch_sum
        self._sum_sq = self._sum_sq + batch_sum_sq
        self._abs_sum = self._abs_sum + batch_abs_sum
        self._minimum = torch.minimum(
            self._minimum,
            batch_min,
        )
        self._maximum = torch.maximum(
            self._maximum,
            batch_max,
        )
        self._nonzero = self._nonzero + batch_nonzero
        self._positive = self._positive + batch_positive
        self._negative = self._negative + batch_negative

    def snapshot(self) -> dict[str, float | int]:
        if self._count == 0:
            return {
                "sample_count": 0,
                "mean": 0.0,
                "mean_abs": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "nonzero_fraction": 0.0,
                "positive_fraction": 0.0,
                "negative_fraction": 0.0,
            }

        total = float(self._sum.item())
        total_sq = float(self._sum_sq.item())
        total_abs = float(self._abs_sum.item())

        values = (
            total,
            total_sq,
            total_abs,
            float(self._minimum.item()),
            float(self._maximum.item()),
        )

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                "diagnostic accumulator contains non-finite values"
            )

        mean = total / self._count

        variance = max(
            0.0,
            total_sq / self._count - mean * mean,
        )

        return {
            "sample_count": self._count,
            "mean": mean,
            "mean_abs": total_abs / self._count,
            "std": math.sqrt(variance),
            "min": float(self._minimum.item()),
            "max": float(self._maximum.item()),
            "nonzero_fraction": (
                float(self._nonzero.item()) / self._count
            ),
            "positive_fraction": (
                float(self._positive.item()) / self._count
            ),
            "negative_fraction": (
                float(self._negative.item()) / self._count
            ),
        }


class NamedTensorStatsAccumulator:
    """Accumulate many named scalar signals in one batched tensor."""

    def __init__(self, names) -> None:
        names = tuple(names)

        if not names:
            raise ValueError(
                "diagnostic names cannot be empty"
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "diagnostic names must be unique"
            )

        self.names = names
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._sum = None
        self._sum_sq = None
        self._abs_sum = None
        self._minimum = None
        self._maximum = None
        self._nonzero = None
        self._positive = None
        self._negative = None

    def update(self, values) -> None:
        import torch

        values = values.detach()

        if values.ndim != 2:
            raise ValueError(
                "named diagnostic values must be [samples, signals]"
            )

        if values.shape[1] != len(self.names):
            raise ValueError(
                "named diagnostic signal count does not match names"
            )

        if values.shape[0] == 0:
            return

        batch_count = int(values.shape[0])

        batch_sum = values.sum(dim=0)
        batch_sum_sq = values.square().sum(dim=0)
        batch_abs_sum = values.abs().sum(dim=0)
        batch_min = values.amin(dim=0)
        batch_max = values.amax(dim=0)

        batch_nonzero = torch.count_nonzero(
            values,
            dim=0,
        )

        batch_positive = torch.count_nonzero(
            values > 0,
            dim=0,
        )

        batch_negative = torch.count_nonzero(
            values < 0,
            dim=0,
        )

        self._count += batch_count

        if self._sum is None:
            self._sum = batch_sum
            self._sum_sq = batch_sum_sq
            self._abs_sum = batch_abs_sum
            self._minimum = batch_min
            self._maximum = batch_max
            self._nonzero = batch_nonzero
            self._positive = batch_positive
            self._negative = batch_negative
            return

        self._sum = self._sum + batch_sum
        self._sum_sq = self._sum_sq + batch_sum_sq
        self._abs_sum = self._abs_sum + batch_abs_sum

        self._minimum = torch.minimum(
            self._minimum,
            batch_min,
        )

        self._maximum = torch.maximum(
            self._maximum,
            batch_max,
        )

        self._nonzero = self._nonzero + batch_nonzero
        self._positive = self._positive + batch_positive
        self._negative = self._negative + batch_negative

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        import torch

        if self._count == 0:
            return {
                name: {
                    "sample_count": 0,
                    "mean": 0.0,
                    "mean_abs": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "nonzero_fraction": 0.0,
                    "positive_fraction": 0.0,
                    "negative_fraction": 0.0,
                }
                for name in self.names
            }

        mean = self._sum / self._count

        variance = (
            self._sum_sq / self._count
            - mean.square()
        ).clamp_min(0.0)

        std = torch.sqrt(variance)

        stacked = torch.stack(
            (
                mean,
                self._abs_sum / self._count,
                std,
                self._minimum,
                self._maximum,
            ),
            dim=0,
        )

        if not bool(torch.isfinite(stacked).all()):
            raise RuntimeError(
                "named diagnostic accumulator contains "
                "non-finite values"
            )

        mean_values = mean.cpu().tolist()
        mean_abs_values = (
            self._abs_sum / self._count
        ).cpu().tolist()
        std_values = std.cpu().tolist()
        minimum_values = self._minimum.cpu().tolist()
        maximum_values = self._maximum.cpu().tolist()
        nonzero_values = self._nonzero.cpu().tolist()
        positive_values = self._positive.cpu().tolist()
        negative_values = self._negative.cpu().tolist()

        result = {}

        for index, name in enumerate(self.names):
            result[name] = {
                "sample_count": self._count,
                "mean": float(mean_values[index]),
                "mean_abs": float(
                    mean_abs_values[index]
                ),
                "std": float(std_values[index]),
                "min": float(
                    minimum_values[index]
                ),
                "max": float(
                    maximum_values[index]
                ),
                "nonzero_fraction": (
                    float(nonzero_values[index])
                    / self._count
                ),
                "positive_fraction": (
                    float(positive_values[index])
                    / self._count
                ),
                "negative_fraction": (
                    float(negative_values[index])
                    / self._count
                ),
            }

        return result


def physical_diagnostic_matrix(diagnostics):
    """Build one [env, signal] matrix from the 043 read-only snapshot."""

    import torch

    required = {
        "root",
        "foot_x_min",
        "foot_x_max",
        "sole_bottom_min",
        "sole_bottom_max",
        "foot_normal_z",
        "vertical_force",
        "bodyweight",
        "loaded",
        "slow_translation",
        "slow_rotation",
        "lateral_contained",
        "support_quality",
        "lower_geometry",
        "upper_geometry",
        "landing_rectangle",
        "stable_lower",
        "stable_upper",
        "swing_zone",
        "swing_ok",
        "clearance_value",
        "distance_to_riser_face",
        "contact",
        "contact_onset",
        "near_upper",
        "precontact_velocity",
        "foot_velocity",
        "foot_angular_velocity",
        "upright",
        "safe",
        "forbidden",
    }

    missing = required - diagnostics.keys()

    if missing:
        raise RuntimeError(
            "physical diagnostic payload is missing: "
            + ", ".join(sorted(missing))
        )

    root = diagnostics["root"]

    if root.ndim != 2 or root.shape[1] != 3:
        raise RuntimeError(
            "physical diagnostic root must be [N, 3]"
        )

    num_envs = root.shape[0]
    dtype = root.dtype

    names = []
    columns = []

    def add(name, value):
        if value.ndim != 1:
            raise RuntimeError(
                f"diagnostic {name} must be one value per environment"
            )

        if value.shape[0] != num_envs:
            raise RuntimeError(
                f"diagnostic {name} has wrong environment count"
            )

        names.append(name)

        if value.dtype == torch.bool:
            value = value.to(dtype=dtype)
        else:
            value = value.to(dtype=dtype)

        columns.append(value)

    add(
        "root_x_m",
        root[:, 0],
    )
    add(
        "root_z_m",
        root[:, 2],
    )

    add(
        "upright_fraction",
        diagnostics["upright"],
    )
    add(
        "safe_fraction",
        diagnostics["safe"],
    )
    add(
        "forbidden_fraction",
        diagnostics["forbidden"],
    )

    bodyweight = diagnostics["bodyweight"]

    if (
        bodyweight.ndim != 1
        or bodyweight.shape[0] != num_envs
    ):
        raise RuntimeError(
            "physical diagnostic bodyweight must be [N]"
        )

    foot_labels = (
        "left",
        "right",
    )

    for index, label in enumerate(foot_labels):
        add(
            f"{label}_foot_x_min_m",
            diagnostics["foot_x_min"][:, index],
        )

        add(
            f"{label}_foot_x_max_m",
            diagnostics["foot_x_max"][:, index],
        )

        add(
            f"{label}_sole_bottom_min_m",
            diagnostics["sole_bottom_min"][:, index],
        )

        add(
            f"{label}_sole_bottom_max_m",
            diagnostics["sole_bottom_max"][:, index],
        )

        add(
            f"{label}_foot_normal_z",
            diagnostics["foot_normal_z"][:, index],
        )

        add(
            f"{label}_vertical_force_bodyweight_fraction",
            (
                diagnostics["vertical_force"][:, index]
                / bodyweight
            ),
        )

        for source, suffix in (
            ("loaded", "loaded_fraction"),
            (
                "slow_translation",
                "slow_translation_fraction",
            ),
            (
                "slow_rotation",
                "slow_rotation_fraction",
            ),
            (
                "lateral_contained",
                "lateral_contained_fraction",
            ),
            (
                "support_quality",
                "support_quality_fraction",
            ),
            (
                "lower_geometry",
                "lower_geometry_fraction",
            ),
            (
                "upper_geometry",
                "upper_geometry_fraction",
            ),
            (
                "landing_rectangle",
                "landing_rectangle_fraction",
            ),
            (
                "stable_lower",
                "stable_lower_fraction",
            ),
            (
                "stable_upper",
                "stable_upper_fraction",
            ),
            (
                "swing_zone",
                "swing_zone_fraction",
            ),
            (
                "swing_ok",
                "swing_ok_fraction",
            ),
            (
                "contact",
                "contact_fraction",
            ),
            (
                "contact_onset",
                "contact_onset_fraction",
            ),
            (
                "near_upper",
                "near_upper_fraction",
            ),
        ):
            add(
                f"{label}_{suffix}",
                diagnostics[source][:, index],
            )

        add(
            f"{label}_clearance_value",
            diagnostics["clearance_value"][:, index],
        )

        add(
            f"{label}_distance_to_riser_face_m",
            diagnostics[
                "distance_to_riser_face"
            ][:, index],
        )

        precontact = diagnostics[
            "precontact_velocity"
        ][:, index]

        velocity = diagnostics[
            "foot_velocity"
        ][:, index]

        angular = diagnostics[
            "foot_angular_velocity"
        ][:, index]

        add(
            f"{label}_precontact_horizontal_speed_mps",
            torch.linalg.vector_norm(
                precontact[:, :2],
                dim=1,
            ),
        )

        add(
            f"{label}_precontact_vertical_speed_mps",
            precontact[:, 2],
        )

        add(
            f"{label}_foot_horizontal_speed_mps",
            torch.linalg.vector_norm(
                velocity[:, :2],
                dim=1,
            ),
        )

        add(
            f"{label}_foot_vertical_speed_mps",
            velocity[:, 2],
        )

        add(
            f"{label}_foot_speed_mps",
            torch.linalg.vector_norm(
                velocity,
                dim=1,
            ),
        )

        add(
            f"{label}_foot_angular_speed_radps",
            torch.linalg.vector_norm(
                angular,
                dim=1,
            ),
        )

    return tuple(names), torch.stack(columns, dim=1)
