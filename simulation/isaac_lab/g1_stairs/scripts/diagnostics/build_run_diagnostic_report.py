#!/usr/bin/env python3
"""Build an offline diagnostic report for one G1 stair-training run.

This script is intentionally independent of Isaac Lab and RSL-RL. It consumes
only persisted run artifacts and never modifies policy, reward, environment,
termination, reset, or training state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


FUNNEL = (
    "episodes_finished_count",
    "episodes_near_riser_count",
    "episodes_swing_count",
    "episodes_riser_crossing_count",
    "episodes_upper_touchdown_count",
    "episodes_stable_upper_count",
    "episodes_first_support_count",
    "episodes_base_transfer_count",
    "episodes_both_feet_count",
    "episodes_completion_count",
)


FUNNEL_LABELS = {
    "episodes_finished_count": "finished episodes",
    "episodes_near_riser_count": "near riser",
    "episodes_swing_count": "swing",
    "episodes_riser_crossing_count": "riser crossing",
    "episodes_upper_touchdown_count": "upper touchdown",
    "episodes_stable_upper_count": "stable upper support",
    "episodes_first_support_count": "first support",
    "episodes_base_transfer_count": "base transfer",
    "episodes_both_feet_count": "both feet",
    "episodes_completion_count": "completion",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        return list(csv.DictReader(stream))


def _number(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip()

        if not text:
            return None

        try:
            result = float(text)
        except ValueError:
            return None

    if not math.isfinite(result):
        return None

    return result


def _integer(value: Any) -> int | None:
    number = _number(value)

    if number is None:
        return None

    integer = int(number)

    if float(integer) != number:
        return None

    return integer


def _latest_training_row(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not rows:
        return {}

    raw = rows[-1]
    result: dict[str, Any] = dict(raw)

    extra_raw = raw.get("extra_json", "")

    if extra_raw:
        try:
            extra = json.loads(extra_raw)
        except json.JSONDecodeError:
            extra = {}

        if isinstance(extra, dict):
            for key, value in extra.items():
                result.setdefault(key, value)

    return result


def _latest_named_rows(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Return the latest row for each long-form diagnostic name."""

    result: dict[str, dict[str, str]] = {}

    for row in rows:
        name = row.get("name")

        if name:
            result[name] = row

    return result


def _sum_named_counts(
    rows: list[dict[str, str]],
) -> dict[str, int]:
    totals: dict[str, int] = {}

    for row in rows:
        name = row.get("name")
        value = _integer(row.get("count"))

        if not name or value is None:
            continue

        totals[name] = totals.get(name, 0) + value

    return totals


def _reward_analysis(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    latest = _latest_named_rows(rows)

    dead = []
    active = []
    weighted = []

    for name, row in latest.items():
        nonzero = _number(
            row.get("nonzero_fraction")
        )
        weighted_abs = _number(
            row.get("weighted_abs_mean")
        )
        weighted_mean = _number(
            row.get("weighted_mean")
        )

        if nonzero is not None:
            if nonzero <= 1e-12:
                dead.append(name)
            else:
                active.append(name)

        if weighted_abs is not None:
            weighted.append(
                {
                    "name": name,
                    "weighted_abs_mean": weighted_abs,
                    "weighted_mean": weighted_mean,
                    "nonzero_fraction": nonzero,
                }
            )

    weighted.sort(
        key=lambda item: item[
            "weighted_abs_mean"
        ],
        reverse=True,
    )

    return {
        "term_count": len(latest),
        "dead_terms": sorted(dead),
        "active_terms": sorted(active),
        "largest_weighted_terms": weighted[:8],
    }


def _episode_analysis(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    counts = _sum_named_counts(rows)

    funnel = {
        name: counts.get(name, 0)
        for name in FUNNEL
    }

    finished = funnel[
        "episodes_finished_count"
    ]

    success = counts.get(
        "terminal_success_count",
        0,
    )

    terminal = {
        name: value
        for name, value in sorted(
            counts.items()
        )
        if (
            name.startswith("terminal_")
            and value > 0
        )
    }

    non_success_classified = sum(
        value
        for name, value in counts.items()
        if (
            name.startswith("terminal_")
            and name != "terminal_success_count"
        )
    )

    bottleneck = _find_bottleneck(
        funnel
    )

    return {
        "counts": counts,
        "funnel": funnel,
        "finished_episodes": finished,
        "successful_episodes": success,
        "classified_non_success": (
            non_success_classified
        ),
        "terminal_nonzero": terminal,
        "bottleneck": bottleneck,
    }


def _find_bottleneck(
    funnel: dict[str, int],
) -> dict[str, Any]:
    finished = funnel.get(
        "episodes_finished_count",
        0,
    )

    if finished <= 0:
        return {
            "status": "insufficient_completed_episodes",
            "message": (
                "No completed episodes are available, "
                "so an episode-level progression bottleneck "
                "cannot yet be inferred."
            ),
        }

    stages = FUNNEL[1:]

    previous_name = (
        "episodes_finished_count"
    )
    previous_value = finished

    largest_drop = None

    for name in stages:
        value = funnel.get(name, 0)

        if previous_value > 0:
            retention = value / previous_value
            drop = 1.0 - retention

            drop_count = (
                previous_value
                - value
            )

            candidate = {
                "from": previous_name,
                "to": name,
                "from_count": previous_value,
                "to_count": value,
                "drop_count": drop_count,
                "retention": retention,
                "drop_fraction": drop,
            }

            # The primary bottleneck is where the largest number of
            # completed episodes is lost. Relative drop is only a
            # tie-breaker. This prevents a tiny 1 -> 0 tail transition
            # from hiding a much larger failure population such as
            # 8 touchdowns -> 2 stable supports.
            if (
                largest_drop is None
                or drop_count
                > largest_drop[
                    "drop_count"
                ]
                or (
                    drop_count
                    == largest_drop[
                        "drop_count"
                    ]
                    and drop
                    > largest_drop[
                        "drop_fraction"
                    ]
                )
            ):
                largest_drop = candidate

        previous_name = name
        previous_value = value

    if largest_drop is None:
        return {
            "status": "unavailable",
            "message": (
                "No comparable funnel transition "
                "was available."
            ),
        }

    source = FUNNEL_LABELS[
        largest_drop["from"]
    ]
    target = FUNNEL_LABELS[
        largest_drop["to"]
    ]

    largest_drop["status"] = "identified"
    largest_drop["message"] = (
        f"Largest completed-episode progression loss "
        f"is from {source} to {target}: "
        f"{largest_drop['from_count']} -> "
        f"{largest_drop['to_count']}."
    )

    return largest_drop


def _event_analysis(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    counts = _sum_named_counts(rows)

    return {
        "counts": counts,
        "nonzero": {
            name: value
            for name, value in sorted(
                counts.items()
            )
            if value > 0
        },
    }


def _stair_analysis(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    latest = _latest_named_rows(rows)

    selected = {}

    interesting_tokens = (
        "clearance",
        "distance_to_riser",
        "touchdown",
        "support",
        "force",
        "normal",
        "foot_speed",
        "horizontal_speed",
        "vertical_speed",
    )

    for name, row in sorted(
        latest.items()
    ):
        if not any(
            token in name
            for token in interesting_tokens
        ):
            continue

        selected[name] = {
            key: _number(row.get(key))
            for key in (
                "sample_count",
                "mean",
                "std",
                "min",
                "max",
            )
        }

    return {
        "signal_count": len(latest),
        "selected_signals": selected,
    }


def _training_analysis(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    latest = _latest_training_row(rows)

    wanted = (
        "iteration",
        "total_simulated_steps",
        "episode_return_mean",
        "episode_length_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "learning_rate",
        "simulation_fps",
        "action_rate_mean",
        "joint_jerk_mean",
        "mechanical_power_mean",
        "nan_count",
        "action_saturation_count",
        "joint_limit_violation_count",
        "velocity_limit_violation_count",
        "torque_limit_violation_count",
        "approach_teacher_loss",
        "approach_teacher_gate_mean",
        "approach_teacher_error_mean",
    )

    values = {}

    for name in wanted:
        if name in latest:
            values[name] = _number(
                latest[name]
            )

    return {
        "iterations_recorded": len(rows),
        "latest": values,
    }


def _physics_reward_alignment(
    training: dict[str, Any],
    episodes: dict[str, Any],
    events: dict[str, Any],
) -> dict[str, Any]:
    latest = training.get(
        "latest",
        {},
    )

    episode_return = latest.get(
        "episode_return_mean"
    )

    finished = episodes.get(
        "finished_episodes",
        0,
    )

    completion = episodes.get(
        "funnel",
        {},
    ).get(
        "episodes_completion_count",
        0,
    )

    event_nonzero = bool(
        events.get("nonzero")
    )

    if finished == 0:
        return {
            "status": "insufficient_episode_evidence",
            "message": (
                "Training return is available, but no completed "
                "episodes are present in the episode funnel. "
                "Do not interpret reward improvement as stair success."
            ),
            "episode_return_mean": episode_return,
            "any_stair_event_observed": event_nonzero,
        }

    if completion == 0:
        return {
            "status": "no_verified_completion",
            "message": (
                "Completed episodes exist but no verified stair "
                "completion was recorded. Reward should therefore "
                "not be treated as evidence of task success."
            ),
            "episode_return_mean": episode_return,
            "any_stair_event_observed": event_nonzero,
        }

    return {
        "status": "verified_completion_present",
        "message": (
            "At least one physically verified completion is present. "
            "Reward trends should still be interpreted alongside the "
            "episode funnel and motion-quality diagnostics."
        ),
        "episode_return_mean": episode_return,
        "any_stair_event_observed": event_nonzero,
    }


def build_summary(
    run_dir: Path,
) -> dict[str, Any]:
    training_rows = _read_csv(
        run_dir / "training_metrics.csv"
    )

    reward_rows = _read_csv(
        run_dir / "reward_diagnostics.csv"
    )

    stair_rows = _read_csv(
        run_dir / "stair_diagnostics.csv"
    )

    event_rows = _read_csv(
        run_dir / "stair_event_counts.csv"
    )

    episode_rows = _read_csv(
        run_dir / "stair_episode_counts.csv"
    )

    training = _training_analysis(
        training_rows
    )

    rewards = _reward_analysis(
        reward_rows
    )

    stairs = _stair_analysis(
        stair_rows
    )

    events = _event_analysis(
        event_rows
    )

    episodes = _episode_analysis(
        episode_rows
    )

    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "run_directory": str(
            run_dir.resolve()
        ),
        "files": {
            "training_metrics": len(
                training_rows
            ),
            "reward_diagnostics": len(
                reward_rows
            ),
            "stair_diagnostics": len(
                stair_rows
            ),
            "stair_event_counts": len(
                event_rows
            ),
            "stair_episode_counts": len(
                episode_rows
            ),
        },
        "training": training,
        "rewards": rewards,
        "stair_mechanics": stairs,
        "events": events,
        "episodes": episodes,
        "reward_physics_alignment": (
            _physics_reward_alignment(
                training,
                episodes,
                events,
            )
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


def render_markdown(
    summary: dict[str, Any],
) -> str:
    training = summary["training"]
    rewards = summary["rewards"]
    episodes = summary["episodes"]
    events = summary["events"]
    alignment = summary[
        "reward_physics_alignment"
    ]

    lines = [
        "# Stair Training Diagnostic Report",
        "",
        f"Run: `{summary['run_id']}`",
        "",
        "## Evidence coverage",
        "",
    ]

    for name, count in (
        summary["files"].items()
    ):
        lines.append(
            f"- `{name}`: {count} rows"
        )

    lines.extend(
        [
            "",
            "## Training health",
            "",
        ]
    )

    latest = training.get(
        "latest",
        {},
    )

    if not latest:
        lines.append(
            "No training metrics were available."
        )
    else:
        for name, value in latest.items():
            lines.append(
                f"- `{name}`: {_fmt(value)}"
            )

    lines.extend(
        [
            "",
            "## Reward diagnostics",
            "",
            (
                f"- Terms observed: "
                f"{rewards['term_count']}"
            ),
            (
                f"- Dead terms in latest iteration: "
                f"{len(rewards['dead_terms'])}"
            ),
        ]
    )

    if rewards["dead_terms"]:
        lines.append(
            "- Dead terms: "
            + ", ".join(
                f"`{name}`"
                for name in rewards[
                    "dead_terms"
                ]
            )
        )

    lines.append(
        "- Largest absolute weighted contributions:"
    )

    for item in rewards[
        "largest_weighted_terms"
    ]:
        lines.append(
            "  - "
            f"`{item['name']}`: "
            f"|weighted mean|="
            f"{_fmt(item['weighted_abs_mean'])}, "
            f"weighted mean="
            f"{_fmt(item['weighted_mean'])}, "
            f"active fraction="
            f"{_fmt(item['nonzero_fraction'])}"
        )

    lines.extend(
        [
            "",
            "## Stair event counts",
            "",
        ]
    )

    if not events["counts"]:
        lines.append(
            "No stair-event artifact was available."
        )
    else:
        for name, value in sorted(
            events["counts"].items()
        ):
            lines.append(
                f"- `{name}`: {value}"
            )

    lines.extend(
        [
            "",
            "## Completed-episode progression funnel",
            "",
        ]
    )

    funnel = episodes["funnel"]

    for name in FUNNEL:
        lines.append(
            f"- {FUNNEL_LABELS[name]}: "
            f"{funnel[name]}"
        )

    lines.extend(
        [
            "",
            "## Automatic bottleneck",
            "",
            episodes["bottleneck"][
                "message"
            ],
            "",
            "## Terminal classifications",
            "",
        ]
    )

    terminal = episodes[
        "terminal_nonzero"
    ]

    if not terminal:
        lines.append(
            "No non-zero terminal classification "
            "bins were recorded."
        )
    else:
        for name, value in terminal.items():
            lines.append(
                f"- `{name}`: {value}"
            )

    lines.extend(
        [
            "",
            "## Reward / physics alignment",
            "",
            alignment["message"],
            "",
            "## Interpretation rule",
            "",
            (
                "Training return is not treated as task success. "
                "Verified stair progression comes from the physical "
                "event detector and completed-episode funnel."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_directory",
        type=Path,
        help=(
            "Path to one experiments/g1_stairs/runs/<run_id> "
            "directory"
        ),
    )

    args = parser.parse_args()

    run_dir = args.run_directory.resolve()

    if not run_dir.is_dir():
        raise SystemExit(
            f"ERROR: run directory not found: {run_dir}"
        )

    summary = build_summary(
        run_dir
    )

    summary_path = (
        run_dir
        / "diagnostic_summary.json"
    )

    report_path = (
        run_dir
        / "diagnostic_report.md"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_path.write_text(
        render_markdown(summary),
        encoding="utf-8",
    )

    print(
        f"Wrote: {summary_path}"
    )
    print(
        f"Wrote: {report_path}"
    )


if __name__ == "__main__":
    main()
