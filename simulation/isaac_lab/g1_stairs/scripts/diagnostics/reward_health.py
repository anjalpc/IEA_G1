#!/usr/bin/env python3
"""Read-only mid-run health card for G1 stair specialist training.

The report intentionally combines reward activation with physical diagnostics.
It never imports Isaac Lab, never touches the trainer, and is safe to run while
CSV files are being appended by the active process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable


STATUS_ORDER = {
    "GOOD": 0,
    "ACTIVE": 0,
    "EMERGING": 0,
    "STABLE": 0,
    "WATCH": 1,
    "INACTIVE": 1,
    "BLOCKED": 2,
    "BAD": 2,
}


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []


def _series(
    rows: Iterable[dict[str, str]],
    name: str | None,
    field: str,
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for row in rows:
        if name is not None and row.get("name") != name:
            continue
        iteration = _finite_float(row.get("iteration"))
        value = _finite_float(row.get(field))
        if iteration is None or value is None:
            continue
        values.append((int(iteration), value))
    return values


def _latest(series: list[tuple[int, float]]):
    return series[-1][1] if series else None


def _window_delta(series: list[tuple[int, float]], window: int = 5):
    if len(series) < 4:
        return None
    width = min(window, len(series) // 2)
    if width < 2:
        return None
    old = mean(value for _, value in series[-2 * width : -width])
    new = mean(value for _, value in series[-width:])
    return old, new, new - old


def _trend(series, *, lower_is_better: bool):
    result = _window_delta(series)
    if result is None:
        return "n/a"
    old, new, delta = result
    scale = max(abs(old), abs(new), 0.01)
    if abs(delta) < 0.08 * scale:
        return "flat"
    improving = delta < 0 if lower_is_better else delta > 0
    return "improving" if improving else "worsening"


def _fmt(value, digits=3, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _status_from_threshold(value, *, good, bad, lower_is_better=True):
    if value is None:
        return "INACTIVE"
    if lower_is_better:
        if value <= good:
            return "GOOD"
        if value >= bad:
            return "BAD"
    else:
        if value >= good:
            return "GOOD"
        if value <= bad:
            return "BAD"
    return "WATCH"


def _reward(reward_rows, name, field="weighted_mean"):
    return _series(reward_rows, name, field)


def _diag(diag_rows, name, field="mean"):
    return _series(diag_rows, name, field)


def _metric(metric_rows, field):
    return _series(metric_rows, None, field)


def build_health(run_dir: Path) -> dict:
    reward_rows = _read_csv(run_dir / "reward_diagnostics.csv")
    diag_rows = _read_csv(run_dir / "stair_diagnostics.csv")
    metric_rows = _read_csv(run_dir / "training_metrics.csv")

    latest_iteration = None
    if metric_rows:
        latest_iteration = int(float(metric_rows[-1]["iteration"]))
    elif reward_rows:
        latest_iteration = int(float(reward_rows[-1]["iteration"]))

    anchor = _diag(diag_rows, "support_lock_anchor_error_m")
    if not anchor:
        anchor = _diag(diag_rows, "support_anchor_error_m")
    support_speed = _diag(diag_rows, "support_lock_planar_speed_mps")
    support_gate = _diag(diag_rows, "support_lock_gate")
    support_guard = _reward(reward_rows, "support_foot_anchor_guard")
    feet_slide = _reward(reward_rows, "feet_slide")

    swing_lat = _diag(diag_rows, "swing_lateral_abs_error_m")
    if not swing_lat:
        # Older runs only recorded signed lane error; abs(mean) is an
        # intentionally conservative fallback, not a replacement for v08's
        # absolute-error diagnostic.
        signed = _diag(diag_rows, "swing_lateral_error_m")
        swing_lat = [(it, abs(value)) for it, value in signed]
    recontact = _diag(diag_rows, "swing_lower_floor_recontact")
    swing_wp = _reward(reward_rows, "swing_waypoint_progress")
    swing_wp_nz = _reward(reward_rows, "swing_waypoint_progress", "nonzero_fraction")
    body = _reward(reward_rows, "swing_body_advance_progress")
    body_nz = _reward(reward_rows, "swing_body_advance_progress", "nonzero_fraction")
    foothold_nz = _reward(reward_rows, "upper_foothold_progress", "nonzero_fraction")
    transfer_nz = _reward(reward_rows, "body_transfer_progress", "nonzero_fraction")
    success_nz = _reward(reward_rows, "single_step_success", "nonzero_fraction")

    phase = {
        index: _latest(_diag(diag_rows, f"motion_phase_is_{index}"))
        for index in range(5)
    }
    phase_complete = _latest(_diag(diag_rows, "motion_phase_is_complete"))

    anchor_v = _latest(anchor)
    speed_v = _latest(support_speed)
    gate_v = _latest(support_gate)
    lateral_v = _latest(swing_lat)
    recontact_v = _latest(recontact)
    guard_v = _latest(support_guard)
    slide_v = _latest(feet_slide)
    swing_wp_v = _latest(swing_wp)
    swing_wp_nz_v = _latest(swing_wp_nz)
    body_v = _latest(body)
    body_nz_v = _latest(body_nz)
    foothold_v = _latest(foothold_nz) or 0.0
    transfer_v = _latest(transfer_nz) or 0.0
    success_v = _latest(success_nz) or 0.0

    anchor_status = _status_from_threshold(
        anchor_v, good=0.015, bad=0.060, lower_is_better=True
    )
    speed_status = _status_from_threshold(
        speed_v, good=0.020, bad=0.200, lower_is_better=True
    )
    support_status = max(
        (anchor_status, speed_status),
        key=lambda s: STATUS_ORDER.get(s, 1),
    )
    if gate_v is not None and gate_v < 0.01 and support_status == "GOOD":
        support_status = "INACTIVE"

    lateral_status = _status_from_threshold(
        lateral_v, good=0.030, bad=0.080, lower_is_better=True
    )
    recontact_status = _status_from_threshold(
        recontact_v, good=0.05, bad=0.20, lower_is_better=True
    )
    swing_status = max(
        (lateral_status, recontact_status),
        key=lambda s: STATUS_ORDER.get(s, 1),
    )
    if swing_wp_nz_v is not None and swing_wp_nz_v < 0.01:
        swing_status = "BLOCKED"

    if body_nz_v is None or body_nz_v < 0.001:
        body_status = "BLOCKED"
    elif body_nz_v < 0.01:
        body_status = "WATCH"
    else:
        body_status = "ACTIVE"

    foothold_status = "EMERGING" if foothold_v > 0 else "INACTIVE"
    transfer_status = "EMERGING" if transfer_v > 0 else "INACTIVE"

    phase3 = phase.get(3) or 0.0
    phase4 = phase.get(4) or 0.0
    phase_status = "WATCH"
    if phase3 > 0.70 and phase4 < 0.005 and foothold_v == 0.0:
        phase_status = "BLOCKED"
    elif phase4 > 0.01 or foothold_v > 0.0:
        phase_status = "ACTIVE"

    nan_v = _latest(_metric(metric_rows, "nan_count"))
    fall_v = _latest(_metric(metric_rows, "fall_rate"))
    dof_nz = _latest(_reward(reward_rows, "dof_pos_limits", "nonzero_fraction"))
    if (nan_v or 0) > 0:
        ppo_status = "BAD"
    elif (fall_v or 0) > 0.01 or (dof_nz or 0) > 0.40:
        ppo_status = "BAD"
    elif (fall_v or 0) > 0.002 or (dof_nz or 0) > 0.20:
        ppo_status = "WATCH"
    else:
        ppo_status = "STABLE"

    return {
        "run": run_dir.name,
        "iteration": latest_iteration,
        "support": {
            "status": support_status,
            "anchor_error_m": anchor_v,
            "anchor_trend": _trend(anchor, lower_is_better=True),
            "planar_speed_mps": speed_v,
            "speed_trend": _trend(support_speed, lower_is_better=True),
            "gate_fraction": gate_v,
            "guard_weighted_mean": guard_v,
            "guard_trend": _trend(support_guard, lower_is_better=False),
            "feet_slide_weighted_mean": slide_v,
        },
        "swing": {
            "status": swing_status,
            "waypoint_weighted_mean": swing_wp_v,
            "waypoint_nonzero_fraction": swing_wp_nz_v,
            "lateral_abs_error_m": lateral_v,
            "lateral_trend": _trend(swing_lat, lower_is_better=True),
            "lower_floor_recontact_fraction": recontact_v,
            "recontact_trend": _trend(recontact, lower_is_better=True),
        },
        "body_advance": {
            "status": body_status,
            "weighted_mean": body_v,
            "nonzero_fraction": body_nz_v,
            "trend": _trend(body_nz, lower_is_better=False),
        },
        "foothold": {
            "status": foothold_status,
            "nonzero_fraction": foothold_v,
        },
        "transfer": {
            "status": transfer_status,
            "nonzero_fraction": transfer_v,
            "success_nonzero_fraction": success_v,
        },
        "phases": {
            "status": phase_status,
            "phase_0": phase.get(0),
            "phase_1": phase.get(1),
            "phase_2": phase.get(2),
            "phase_3": phase.get(3),
            "phase_4": phase.get(4),
            "complete": phase_complete,
        },
        "ppo": {
            "status": ppo_status,
            "mean_return": _latest(_metric(metric_rows, "episode_return_mean")),
            "fall_rate": fall_v,
            "dof_limit_nonzero_fraction": dof_nz,
            "action_rate": _latest(_metric(metric_rows, "action_rate_mean")),
            "mechanical_power_w": _latest(_metric(metric_rows, "mechanical_power_mean")),
            "learning_rate": _latest(_metric(metric_rows, "learning_rate")),
            "entropy": _latest(_metric(metric_rows, "entropy")),
            "value_loss": _latest(_metric(metric_rows, "value_loss")),
            "policy_loss": _latest(_metric(metric_rows, "policy_loss")),
            "approximate_kl": _latest(_metric(metric_rows, "approximate_kl")),
            "clip_fraction": _latest(_metric(metric_rows, "clip_fraction")),
            "nan_count": nan_v,
        },
        "task": {
            "maximum_x_m": _latest(_metric(metric_rows, "maximum_x_any_m")),
            "maximum_step": _latest(_metric(metric_rows, "maximum_step_reached_any")),
            "platform_reached_rate": _latest(_metric(metric_rows, "platform_reached_rate")),
        },
    }


def _print_line(label, value, status=None, trend=None):
    suffix = ""
    if trend and trend != "n/a":
        suffix += f"  [{trend}]"
    if status:
        suffix += f"  {status}"
    print(f"  {label:<28} {value:<14}{suffix}")


def print_card(health: dict) -> None:
    print("========================================")
    print(" G1 STAIR REWARD HEALTH")
    print("========================================")
    print(f"Run:       {health['run']}")
    print(f"Iteration: {health['iteration'] if health['iteration'] is not None else 'n/a'}")

    s = health["support"]
    print("\nSUPPORT LOCK")
    _print_line("status", "", s["status"])
    _print_line("anchor error", _fmt(s["anchor_error_m"], 3, " m"), trend=s["anchor_trend"])
    _print_line("planar speed", _fmt(s["planar_speed_mps"], 3, " m/s"), trend=s["speed_trend"])
    _print_line("active gate", _fmt(s["gate_fraction"], 3))
    _print_line("anchor guard Wmean", _fmt(s["guard_weighted_mean"], 4), trend=s["guard_trend"])
    _print_line("feet slide Wmean", _fmt(s["feet_slide_weighted_mean"], 4))
    print("  contract: <=0.015 m / <=0.020 m/s free; >=0.060 m / >=0.200 m/s bad")

    s = health["swing"]
    print("\nSWING")
    _print_line("status", "", s["status"])
    _print_line("waypoint Wmean", _fmt(s["waypoint_weighted_mean"], 4))
    _print_line("waypoint active", _fmt(s["waypoint_nonzero_fraction"], 3))
    _print_line("lateral abs error", _fmt(s["lateral_abs_error_m"], 3, " m"), trend=s["lateral_trend"])
    _print_line("lower-floor recontact", _fmt(s["lower_floor_recontact_fraction"], 3), trend=s["recontact_trend"])

    s = health["body_advance"]
    print("\nBODY ADVANCE")
    _print_line("status", "", s["status"])
    _print_line("weighted mean", _fmt(s["weighted_mean"], 5))
    _print_line("active fraction", _fmt(s["nonzero_fraction"], 4), trend=s["trend"])

    f = health["foothold"]
    t = health["transfer"]
    print("\nFOOTHOLD / TRANSFER")
    _print_line("upper foothold", _fmt(f["nonzero_fraction"], 5), f["status"])
    _print_line("body transfer", _fmt(t["nonzero_fraction"], 5), t["status"])
    _print_line("single-step success", _fmt(t["success_nonzero_fraction"], 5))

    p = health["phases"]
    print("\nPHASE OCCUPANCY")
    _print_line("status", "", p["status"])
    values = [p[f"phase_{i}"] for i in range(5)] + [p["complete"]]
    labels = ["P0", "P1", "P2", "P3", "P4", "DONE"]
    phase_text = "  ".join(
        f"{label}:{_fmt(value, 3)}" for label, value in zip(labels, values)
    )
    print(f"  {phase_text}")

    p = health["ppo"]
    print("\nPPO / SAFETY")
    _print_line("status", "", p["status"])
    _print_line("mean return", _fmt(p["mean_return"], 3))
    _print_line("fall rate", _fmt(p["fall_rate"], 5))
    _print_line("soft-limit active", _fmt(p["dof_limit_nonzero_fraction"], 3))
    _print_line("action rate", _fmt(p["action_rate"], 3))
    _print_line("mechanical power", _fmt(p["mechanical_power_w"], 1, " W"))
    _print_line("learning rate", _fmt(p["learning_rate"], 7))
    _print_line("entropy", _fmt(p["entropy"], 3))
    _print_line("value loss", _fmt(p["value_loss"], 4))
    _print_line("policy loss", _fmt(p["policy_loss"], 4))
    _print_line("approx KL", _fmt(p["approximate_kl"], 5))
    _print_line("clip fraction", _fmt(p["clip_fraction"], 4))
    _print_line("NaNs", _fmt(p["nan_count"], 0))

    t = health["task"]
    print("\nTASK")
    _print_line("maximum X", _fmt(t["maximum_x_m"], 3, " m"))
    _print_line("maximum step", _fmt(t["maximum_step"], 0))
    _print_line("platform fraction", _fmt(t["platform_reached_rate"], 5))
    print("========================================")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"run directory does not exist: {args.run_dir}")

    health = build_health(args.run_dir)
    if args.json:
        print(json.dumps(health, indent=2, sort_keys=True))
    else:
        print_card(health)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
