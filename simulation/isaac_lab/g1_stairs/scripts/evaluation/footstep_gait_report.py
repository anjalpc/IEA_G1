#!/usr/bin/env python3
"""Gait quality from a GeometryTrace trace.jsonl (first episode only).

Pure Python, runs on the host. Compares a checkpoint against the Unitree
prior baseline: alternation, left/right timing and step-length symmetry,
speed tracking and lateral drift. Reset jumps are excluded by stopping at
the first terminated / time-out sample.

Usage:
  python3 footstep_gait_report.py CANDIDATE/trace.jsonl [REFERENCE/trace.jsonl]
"""
from __future__ import annotations

import json
import statistics
import sys

STEP_DT_S = 0.02
SETTLE_S = 2.0


def first_episode(rows):
    out = []
    for row in rows:
        if row.get("sample") == "initial":
            continue
        out.append(row)
        if row.get("terminated") or row.get("time_out"):
            break
    return out


def _median(values):
    return statistics.median(values) if values else None


def gait_metrics(rows):
    rows = first_episode(rows)
    if len(rows) < 2:
        raise ValueError("trace has fewer than two post-physics samples")
    audits = [row["footstep_landing_audit"] for row in rows]
    support = [a["supported_lr"] for a in audits]
    soles = [a["sole_centre_local"] for a in audits]
    touchdowns = [(k, foot) for k in range(1, len(rows)) for foot in (0, 1)
                  if support[k][foot] and not support[k - 1][foot]]
    touchdowns.sort()
    pairs = list(zip(touchdowns, touchdowns[1:]))
    alternating = [(a, b) for a, b in pairs if a[1] != b[1]]
    interval = {0: [], 1: []}   # keyed by the foot that lands second
    length = {0: [], 1: []}
    for (ka, fa), (kb, fb) in alternating:
        interval[fb].append((kb - ka) * STEP_DT_S)
        length[fb].append(soles[kb][fb][0] - soles[ka][fa][0])
    settle = int(SETTLE_S / STEP_DT_S)
    tail = rows[settle:] or rows
    vx = [r["root_lin_vel_w"][0] for r in tail]
    cmd = [r["command_base_velocity"][0] for r in tail]
    li, ri = _median(interval[0]), _median(interval[1])
    ll, rl = _median(length[0]), _median(length[1])
    final = rows[-1]
    origin = rows[0]["env_origin_w"]
    return {
        "duration_s": round(len(rows) * STEP_DT_S, 2),
        "ended_by": ("fall/terminated" if final.get("terminated")
                     else "time_out" if final.get("time_out") else "trace_end"),
        "touchdowns_left": sum(1 for _, f in touchdowns if f == 0),
        "touchdowns_right": sum(1 for _, f in touchdowns if f == 1),
        "alternation_ratio": round(len(alternating) / len(pairs), 3) if pairs else None,
        "interval_to_left_s": li,
        "interval_to_right_s": ri,
        "timing_symmetry": round(min(li, ri) / max(li, ri), 3) if li and ri else None,
        "step_length_left_m": round(ll, 3) if ll is not None else None,
        "step_length_right_m": round(rl, 3) if rl is not None else None,
        "length_symmetry": (round(min(abs(ll), abs(rl)) / max(abs(ll), abs(rl)), 3)
                            if ll and rl else None),
        "mean_command_vx": round(statistics.fmean(cmd), 3),
        "mean_root_vx": round(statistics.fmean(vx), 3),
        "lateral_drift_m": round((final["root_pos_w"][1] - origin[1])
                                 - (rows[0]["root_pos_w"][1] - origin[1]), 3),
        "planned_targets_completed": final.get("planned_footsteps", {}).get("completed"),
    }


def load(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    columns = [("candidate", gait_metrics(load(argv[1])))]
    if len(argv) == 3:
        columns.append(("reference", gait_metrics(load(argv[2]))))
    keys = list(columns[0][1])
    print(f"{'metric':28s}" + "".join(f"{name:>18s}" for name, _ in columns))
    for key in keys:
        print(f"{key:28s}" + "".join(f"{str(m[key]):>18s}" for _, m in columns))
    print(json.dumps({name: m for name, m in columns}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
