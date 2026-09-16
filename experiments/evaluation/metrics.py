"""Machine-readable metrics for G1 stair-RL experiments."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiments.evaluation.stair_outcomes import EpisodeResult


TRAINING_FIELDS = (
    "iteration",
    "total_simulated_steps",
    "wall_time_s",
    "episode_return_mean",
    "episode_return_median",
    "episode_return_std",
    "episode_return_min",
    "episode_return_max",
    "episode_length_mean",
    "success_rate",
    "fall_rate",
    "timeout_rate",
    "stall_rate",
    "slip_rate",
    "forbidden_contact_rate",
    "maximum_x_mean_m",
    "maximum_base_height_mean_m",
    "maximum_step_reached_mean",
    "platform_reached_rate",
    "joint_position_max_abs",
    "joint_velocity_max_abs",
    "torque_max_abs",
    "action_saturation_count",
    "joint_limit_violation_count",
    "velocity_limit_violation_count",
    "torque_limit_violation_count",
    "action_rate_mean",
    "action_jerk_mean",
    "joint_jerk_mean",
    "mechanical_power_mean",
    "energy_proxy_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "explained_variance",
    "learning_rate",
    "gradient_norm",
    "simulation_fps",
    "environment_steps_per_second",
    "gpu_utilization_percent",
    "gpu_memory_used_mib",
    "nan_count",
    "reset_count",
    "simulator_error_count",
    "curriculum_level_mean",
    "extra_json",
)

REWARD_FIELDS = (
    "iteration",
    "total_simulated_steps",
    "name",
    "raw_value",
    "weight",
    "weighted_value",
)

EPISODE_FIELDS = (
    "run_id",
    "checkpoint",
    "evaluation_seed",
    "episode_index",
    "outcome",
    "success",
    "termination_time_s",
    "primary_reason",
    "secondary_flags",
    "maximum_x_m",
    "maximum_base_height_m",
    "maximum_step_reached",
    "approach_completed",
    "platform_reached",
    "platform_continuation_m",
    "upright_dwell_s",
)

REQUIRED_TRAINING_FIELDS = frozenset(
    {
        "iteration",
        "total_simulated_steps",
        "wall_time_s",
    }
)


def _require_finite(value: Any, label: str) -> None:
    """Reject non-finite numeric values before writing evidence."""

    if isinstance(value, bool) or value is None:
        return

    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")


def _append_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    """Append rows while creating a stable header once."""

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        if write_header:
            writer.writeheader()

        writer.writerows(rows)


def wilson_interval(
    successes: int,
    attempts: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson binomial confidence interval."""

    if attempts < 0:
        raise ValueError("attempts cannot be negative")

    if successes < 0 or successes > attempts:
        raise ValueError(
            "successes must be between zero and attempts"
        )

    if attempts == 0:
        return (0.0, 0.0)

    proportion = successes / attempts
    denominator = 1.0 + z * z / attempts

    centre = (
        proportion
        + z * z / (2.0 * attempts)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / attempts
            + z * z / (4.0 * attempts * attempts)
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


class RunMetricsWriter:
    """Write stable CSV and JSON evidence for one experiment run."""

    def __init__(
        self,
        output_directory: Path,
        run_id: str,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id cannot be empty")

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.run_id = run_id
        self.training_path = (
            self.output_directory / "training_metrics.csv"
        )
        self.reward_path = (
            self.output_directory / "reward_components.csv"
        )
        self.episode_path = (
            self.output_directory / "evaluation_episodes.csv"
        )
        self.summary_path = (
            self.output_directory / "evaluation_summary.json"
        )

        self._outcome_counts: dict[str, int] = {}
        self._episode_count = 0
        self._success_count = 0

    def record_training_snapshot(
        self,
        snapshot: Mapping[str, Any],
        reward_components: Mapping[
            str,
            Mapping[str, float],
        ],
    ) -> None:
        """Record one training iteration and its reward decomposition."""

        missing = REQUIRED_TRAINING_FIELDS - snapshot.keys()

        if missing:
            raise ValueError(
                "training snapshot is missing: "
                + ", ".join(sorted(missing))
            )

        known_values: dict[str, Any] = {}
        extra_values: dict[str, Any] = {}

        for key, value in snapshot.items():
            _require_finite(value, f"training metric {key}")

            if key in TRAINING_FIELDS and key != "extra_json":
                known_values[key] = value
            else:
                extra_values[key] = value

        row = {
            field: known_values.get(field)
            for field in TRAINING_FIELDS
        }

        row["extra_json"] = json.dumps(
            extra_values,
            sort_keys=True,
        )

        reward_rows: list[dict[str, Any]] = []

        for name in sorted(reward_components):
            component = reward_components[name]

            if "raw" not in component:
                raise ValueError(
                    f"reward component {name} is missing raw"
                )

            if "weight" not in component:
                raise ValueError(
                    f"reward component {name} is missing weight"
                )

            raw = float(component["raw"])
            weight = float(component["weight"])
            weighted = raw * weight

            _require_finite(
                raw,
                f"reward {name} raw",
            )
            _require_finite(
                weight,
                f"reward {name} weight",
            )
            _require_finite(
                weighted,
                f"reward {name} weighted",
            )

            reward_rows.append(
                {
                    "iteration": snapshot["iteration"],
                    "total_simulated_steps": snapshot[
                        "total_simulated_steps"
                    ],
                    "name": name,
                    "raw_value": raw,
                    "weight": weight,
                    "weighted_value": weighted,
                }
            )

        # Do not write either evidence file until the entire snapshot
        # and every reward component have passed validation.
        _append_csv(
            self.training_path,
            TRAINING_FIELDS,
            [row],
        )

        _append_csv(
            self.reward_path,
            REWARD_FIELDS,
            reward_rows,
        )

    def record_episode(
        self,
        result: EpisodeResult,
        *,
        checkpoint: str,
        evaluation_seed: int,
        episode_index: int,
    ) -> None:
        """Record one independently evaluated episode."""

        if episode_index < 0:
            raise ValueError(
                "episode_index cannot be negative"
            )

        metrics = result.metrics

        row = {
            "run_id": self.run_id,
            "checkpoint": checkpoint,
            "evaluation_seed": evaluation_seed,
            "episode_index": episode_index,
            "outcome": result.outcome.value,
            "success": result.success,
            "termination_time_s": result.termination_time_s,
            "primary_reason": result.primary_reason,
            "secondary_flags": json.dumps(
                list(result.secondary_flags)
            ),
            "maximum_x_m": metrics.get("maximum_x_m"),
            "maximum_base_height_m": metrics.get(
                "maximum_base_height_m"
            ),
            "maximum_step_reached": metrics.get(
                "maximum_step_reached"
            ),
            "approach_completed": metrics.get(
                "approach_completed"
            ),
            "platform_reached": metrics.get(
                "platform_reached"
            ),
            "platform_continuation_m": metrics.get(
                "platform_continuation_m"
            ),
            "upright_dwell_s": metrics.get(
                "upright_dwell_s"
            ),
        }

        for key, value in row.items():
            _require_finite(
                value,
                f"episode metric {key}",
            )

        _append_csv(
            self.episode_path,
            EPISODE_FIELDS,
            [row],
        )

        outcome_name = result.outcome.value

        self._outcome_counts[outcome_name] = (
            self._outcome_counts.get(outcome_name, 0)
            + 1
        )

        self._episode_count += 1

        if result.success:
            self._success_count += 1

    def write_summary(self) -> dict[str, Any]:
        """Write an aggregate evaluation summary."""

        lower, upper = wilson_interval(
            self._success_count,
            self._episode_count,
        )

        success_rate = (
            self._success_count / self._episode_count
            if self._episode_count
            else 0.0
        )

        summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "episode_count": self._episode_count,
            "success_count": self._success_count,
            "success_rate": success_rate,
            "success_rate_confidence_95": {
                "lower": lower,
                "upper": upper,
            },
            "outcome_counts": dict(
                sorted(self._outcome_counts.items())
            ),
            "files": {
                "training_metrics": (
                    self.training_path.name
                ),
                "reward_components": (
                    self.reward_path.name
                ),
                "evaluation_episodes": (
                    self.episode_path.name
                ),
            },
        }

        self.summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return summary