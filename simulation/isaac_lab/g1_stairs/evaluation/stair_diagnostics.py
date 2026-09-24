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


class StairEventDiagnostics:
    """Stateful, reward-independent 043 mechanics diagnostics.

    Episode state survives PPO iteration boundaries. Only the per-iteration
    counters/statistics are cleared after reporting.

    Nothing in this class feeds rewards, observations, terminations, resets,
    physical achievement state, or PPO optimization.
    """

    FOOT_LABELS = ("left", "right")

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        first_riser_m: float,
        tread_m: float,
        height_m: float,
        takeoff_distance_m: float,
    ) -> None:
        import torch

        if num_envs <= 0:
            raise ValueError(
                "StairEventDiagnostics requires at least one environment"
            )

        for name, value in (
            ("first_riser_m", first_riser_m),
            ("tread_m", tread_m),
            ("height_m", height_m),
            ("takeoff_distance_m", takeoff_distance_m),
        ):
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        self._torch = torch
        self.num_envs = int(num_envs)
        self.device = device

        self.first_riser_m = float(first_riser_m)
        self.tread_m = float(tread_m)
        self.height_m = float(height_m)
        self.takeoff_distance_m = float(
            takeoff_distance_m
        )

        # This mirrors the existing 043 early-lift threshold:
        #
        #     bottom.amax(dim=2) - 0.08
        #
        # D4 does not alter that reward threshold.
        self.early_lift_height_m = 0.08

        self._seen_near_riser = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=device,
        )

        self._seen_premature_high_lift = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self._seen_swing = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self._seen_crossing = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self._seen_touchdown = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self._seen_stable_upper = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self._pending_touchdown = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self._touchdown_age_s = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=device,
        )

        self._iteration_counts = {}
        self._iteration_stats = {}

    def reset(self, ids) -> None:
        """Reset episode-local diagnostic state for completed environments."""

        if ids is None:
            ids = slice(None)

        for value in (
            self._seen_near_riser,
            self._seen_premature_high_lift,
            self._seen_swing,
            self._seen_crossing,
            self._seen_touchdown,
            self._seen_stable_upper,
            self._pending_touchdown,
            self._touchdown_age_s,
        ):
            value[ids] = 0

    def _count(self, name: str, mask) -> None:
        value = mask.to(
            dtype=self._torch.long
        ).sum()

        if name not in self._iteration_counts:
            self._iteration_counts[name] = value
        else:
            self._iteration_counts[name] = (
                self._iteration_counts[name]
                + value
            )

    def _record(self, name: str, values) -> None:
        if values.numel() == 0:
            return

        accumulator = self._iteration_stats.get(name)

        if accumulator is None:
            accumulator = TensorStatsAccumulator()
            self._iteration_stats[name] = accumulator

        accumulator.update(values)

    def _foot_matrix(self, diagnostics, name: str):
        value = diagnostics[name]

        if (
            value.ndim != 2
            or value.shape != (self.num_envs, 2)
        ):
            raise RuntimeError(
                f"{name} must have shape "
                f"[{self.num_envs}, 2], got "
                f"{tuple(value.shape)}"
            )

        return value

    def update(
        self,
        diagnostics,
        *,
        dt: float,
    ) -> None:
        """Observe one pre-autoreset physical snapshot."""

        torch = self._torch

        if dt <= 0:
            raise ValueError(
                "diagnostic timestep must be positive"
            )

        required = {
            "foot_x_min",
            "foot_x_max",
            "sole_bottom_min",
            "sole_bottom_max",
            "foot_normal_z",
            "stable_upper",
            "swing_ok",
            "distance_to_riser_face",
            "contact_onset",
            "near_upper",
            "precontact_velocity",
        }

        missing = required - diagnostics.keys()

        if missing:
            raise RuntimeError(
                "stair-event diagnostics are missing: "
                + ", ".join(sorted(missing))
            )

        foot_x_min = self._foot_matrix(
            diagnostics,
            "foot_x_min",
        )

        foot_x_max = self._foot_matrix(
            diagnostics,
            "foot_x_max",
        )

        sole_bottom_min = self._foot_matrix(
            diagnostics,
            "sole_bottom_min",
        )

        sole_bottom_max = self._foot_matrix(
            diagnostics,
            "sole_bottom_max",
        )

        foot_normal_z = self._foot_matrix(
            diagnostics,
            "foot_normal_z",
        )

        stable_upper = self._foot_matrix(
            diagnostics,
            "stable_upper",
        ).to(dtype=torch.bool)

        swing_ok = self._foot_matrix(
            diagnostics,
            "swing_ok",
        ).to(dtype=torch.bool)

        distance = self._foot_matrix(
            diagnostics,
            "distance_to_riser_face",
        )

        contact_onset = self._foot_matrix(
            diagnostics,
            "contact_onset",
        ).to(dtype=torch.bool)

        near_upper = self._foot_matrix(
            diagnostics,
            "near_upper",
        ).to(dtype=torch.bool)

        precontact_velocity = diagnostics[
            "precontact_velocity"
        ]

        if precontact_velocity.shape != (
            self.num_envs,
            2,
            3,
        ):
            raise RuntimeError(
                "precontact_velocity must have shape "
                f"[{self.num_envs}, 2, 3], got "
                f"{tuple(precontact_velocity.shape)}"
            )

        # Pending support latency advances from the first upper touchdown.
        self._touchdown_age_s = torch.where(
            self._pending_touchdown,
            self._touchdown_age_s + dt,
            self._touchdown_age_s,
        )

        # ------------------------------------------------------------
        # Funnel 1: first entry into the foot-based takeoff region.
        # ------------------------------------------------------------

        nearest_distance = distance.amin(dim=1)

        near_riser = (
            nearest_distance
            <= self.takeoff_distance_m
        )

        new_near_riser = (
            near_riser
            & ~self._seen_near_riser
        )

        self._count(
            "near_riser_entry_count",
            new_near_riser,
        )

        self._record(
            "near_riser_entry_distance_m",
            nearest_distance[new_near_riser],
        )

        self._seen_near_riser |= near_riser

        # ------------------------------------------------------------
        # Funnel 2: premature high-lift onset while still farther than
        # the existing 043 takeoff distance.
        # ------------------------------------------------------------

        premature_high_lift = (
            (
                sole_bottom_max
                > self.early_lift_height_m
            )
            & (
                distance
                > self.takeoff_distance_m
            )
        )

        new_premature_high_lift = (
            premature_high_lift
            & ~self._seen_premature_high_lift
        )

        for index, label in enumerate(
            self.FOOT_LABELS
        ):
            mask = new_premature_high_lift[
                :, index
            ]

            self._count(
                f"{label}_premature_high_lift_count",
                mask,
            )

            self._record(
                f"{label}_premature_high_lift_distance_m",
                distance[:, index][mask],
            )

            self._record(
                f"{label}_premature_high_lift_height_m",
                sole_bottom_max[:, index][mask],
            )

        self._seen_premature_high_lift |= (
            premature_high_lift
        )

        # ------------------------------------------------------------
        # Funnel 3: first valid swing entry according to the same
        # physical swing gate already used by 043.
        # ------------------------------------------------------------

        new_swing = (
            swing_ok
            & ~self._seen_swing
        )

        for index, label in enumerate(
            self.FOOT_LABELS
        ):
            mask = new_swing[:, index]

            self._count(
                f"{label}_swing_entry_count",
                mask,
            )

            self._record(
                f"{label}_swing_entry_distance_m",
                distance[:, index][mask],
            )

            self._record(
                f"{label}_swing_entry_sole_height_m",
                sole_bottom_min[:, index][mask],
            )

        self._seen_swing |= swing_ok

        # ------------------------------------------------------------
        # Funnel 4: first time the front of each physical foot
        # footprint reaches/crosses the riser plane.
        # ------------------------------------------------------------

        crossed_riser = (
            foot_x_max
            >= self.first_riser_m
        )

        new_crossing = (
            crossed_riser
            & ~self._seen_crossing
        )

        for index, label in enumerate(
            self.FOOT_LABELS
        ):
            mask = new_crossing[:, index]

            self._count(
                f"{label}_riser_crossing_count",
                mask,
            )

            self._record(
                f"{label}_riser_crossing_clearance_m",
                (
                    sole_bottom_min[:, index]
                    - self.height_m
                )[mask],
            )

        self._seen_crossing |= crossed_riser

        # ------------------------------------------------------------
        # Funnel 5: first upper touchdown.
        #
        # The velocity is the detector's existing PRE-CONTACT velocity,
        # not the post-impact velocity.
        # ------------------------------------------------------------

        upper_touchdown = (
            contact_onset
            & near_upper
        )

        new_touchdown = (
            upper_touchdown
            & ~self._seen_touchdown
        )

        precontact_horizontal_speed = (
            torch.linalg.vector_norm(
                precontact_velocity[..., :2],
                dim=2,
            )
        )

        for index, label in enumerate(
            self.FOOT_LABELS
        ):
            mask = new_touchdown[:, index]

            self._count(
                f"{label}_upper_touchdown_count",
                mask,
            )

            self._record(
                f"{label}_touchdown_precontact_horizontal_speed_mps",
                precontact_horizontal_speed[:, index][mask],
            )

            self._record(
                f"{label}_touchdown_precontact_vertical_speed_mps",
                precontact_velocity[:, index, 2][mask],
            )

            self._record(
                f"{label}_touchdown_foot_normal_z",
                foot_normal_z[:, index][mask],
            )

            self._record(
                f"{label}_touchdown_sole_height_error_m",
                (
                    sole_bottom_min[:, index]
                    - self.height_m
                )[mask],
            )

            self._record(
                f"{label}_touchdown_rear_margin_m",
                (
                    foot_x_min[:, index]
                    - self.first_riser_m
                )[mask],
            )

            self._record(
                f"{label}_touchdown_front_margin_m",
                (
                    self.first_riser_m
                    + self.tread_m
                    - foot_x_max[:, index]
                )[mask],
            )

        # Start support-acquisition clocks only on the first upper
        # touchdown of each foot in the episode.
        self._pending_touchdown[
            new_touchdown
        ] = True

        self._touchdown_age_s[
            new_touchdown
        ] = 0.0

        self._seen_touchdown |= upper_touchdown

        # ------------------------------------------------------------
        # Funnel 6: first stable upper support.
        # ------------------------------------------------------------

        new_stable_upper = (
            stable_upper
            & ~self._seen_stable_upper
        )

        for index, label in enumerate(
            self.FOOT_LABELS
        ):
            mask = new_stable_upper[:, index]

            self._count(
                f"{label}_stable_upper_entry_count",
                mask,
            )

            with_touchdown = (
                mask
                & self._pending_touchdown[:, index]
            )

            self._record(
                f"{label}_support_acquisition_latency_s",
                self._touchdown_age_s[
                    :, index
                ][with_touchdown],
            )

        # Once stable support is achieved, that touchdown clock has
        # served its purpose.
        self._pending_touchdown[
            stable_upper
        ] = False

        self._seen_stable_upper |= stable_upper

    def consume_iteration(
        self,
    ) -> tuple[
        dict[str, int],
        dict[str, dict[str, float | int]],
    ]:
        """Return iteration evidence while preserving episode state."""

        counts = {
            name: int(value.item())
            for name, value
            in self._iteration_counts.items()
        }

        stats = {
            name: accumulator.snapshot()
            for name, accumulator
            in self._iteration_stats.items()
        }

        self._iteration_counts = {}
        self._iteration_stats = {}

        return counts, stats
