#!/usr/bin/env python3
"""Training status for `simctl status` (host side, pure Python + PyYAML).

Reads the run's training log (one block per "Learning iteration N/M") and
prints plain-language behaviour and learning signals with their trend over
the whole run, plus warnings when something is getting worse. Training
values include exploration noise; replays (trace_report.py) decide.

  python3 training_status.py RUN_DIR LOG [--all]

"now" and "run start" are means of 10 iterations (the first 50 of a run are
skipped: all robots restart together, so early episode statistics are biased).
Episode endings below 1% of episodes are never flagged. The trend compares the last 100
iterations with the 100 before; a change >= 15% the wrong way is flagged,
and so is a change >= 25% the wrong way since the start of the run.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import statistics as st
import sys

ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
VALUE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ /.-]*?):\s+(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*$")
SHAPING = re.compile(r"\[POLICY_SHAPING\] means beyond clip (-?[\d.]+)")
STEP_DT_S = 0.02
WINDOW = 100          # iterations compared: last WINDOW vs the WINDOW before
WORSE = 0.15          # relative change that raises a warning
RUN_WORSE = 0.25      # whole-run change (now vs run start) that raises a warning
WARMUP = 50           # first iterations of a run: all envs restart together, episode
                      # statistics need ~2 episode lengths (~22 iterations each) to settle
FLOOR = {"Episode_Termination/": 0.01}   # changes below 1% of episodes are not flagged
NOW = 10              # "now" and "run start" average this many iterations
SPARK = "▁▂▃▄▅▆▇█"

# (label, log key, better: "down"/"up"/None, use magnitude)
BEHAVIOUR = (
    ("Tipping over (per episode)", "Episode_Termination/bad_orientation", "down", False),
    ("Stuck on stairs", "Episode_Termination/stair_target_stall", "down", False),
    ("Reaches course end", "Episode_Termination/course_exit", "up", False),
    ("Planned steps hit", "Episode_Reward/footstep_acquisition", "up", False),
    ("Toe hits step edge", "Episode_Reward/foot_stumble", "down", True),
    ("Hard landings", "Episode_Reward/landing_impact", "down", True),
    ("Unwanted turning", "Episode_Reward/uncommanded_yaw", "down", True),
    ("Speed error m/s", "Metrics/base_velocity/error_vel_xy", "down", False),
    ("Episode length s", "Mean episode length", "up", False),
)
LEARNING = (
    ("Exploration noise", "Mean action noise std", None, False),
    ("Value loss", "Mean value_function loss", None, False),
    ("Symmetry loss", "Mean symmetry loss", "down", False),
    ("Outputs beyond clip", "policy_shaping_beyond_clip", "down", False),
)


def parse_log(path):
    """{iteration: {key: value}} from the training log."""
    blocks, current = {}, None
    with open(path, errors="replace") as stream:
        for line in stream:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            hit = ITERATION.search(line)
            if hit:
                current = blocks.setdefault(int(hit.group(1)), {})
                continue
            if current is None:
                continue
            hit = SHAPING.search(line)
            if hit:
                current["policy_shaping_beyond_clip"] = float(hit.group(1))
                continue
            hit = VALUE.match(line)
            if hit:
                current[hit.group(1).strip()] = float(hit.group(2))
    return blocks


def series(blocks, key, magnitude=False):
    out = [(it, b[key]) for it, b in sorted(blocks.items()) if key in b and math.isfinite(b[key])]
    if key == "Mean episode length":
        out = [(it, v * STEP_DT_S) for it, v in out]
    out = [(it, abs(v) if magnitude else v) for it, v in out]
    return out[WARMUP:] if len(out) > 2 * WARMUP else out


def sparkline(values, slots=12):
    if not values:
        return ""
    chunk = max(1, math.ceil(len(values) / slots))
    means = [st.fmean(values[i:i + chunk]) for i in range(0, len(values), chunk)]
    lo, hi = min(means), max(means)
    if hi - lo < 1e-12:
        return SPARK[3] * len(means)
    return "".join(SPARK[round((m - lo) / (hi - lo) * (len(SPARK) - 1))] for m in means)


def trend(values, better):
    """(text, worse) comparing the last WINDOW values with the WINDOW before."""
    if len(values) < 20:
        return "too early", False
    w = min(WINDOW, len(values) // 2)
    recent, before = st.fmean(values[-w:]), st.fmean(values[-2 * w:-w])
    scale = max(abs(before), abs(recent), 1e-9)
    change = (recent - before) / scale
    if abs(change) < 0.05 or better is None:
        return (f"steady ({change:+.0%})" if better else f"{change:+.0%}"), False
    good = (change < 0) == (better == "down")
    return f"{'better' if good else 'WORSE'} ({change:+.0%})", (not good and abs(change) >= WORSE)


def fmt(v):
    return "N/A" if v is None else (f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}")


def section(title, rows, blocks, warnings):
    shown = [(label, series(blocks, key, mag), better, key) for label, key, better, mag in rows]
    shown = [(label, s, better, key) for label, s, better, key in shown if s]
    if not shown:
        return
    w = min(WINDOW, max(len(s) for _, s, _, _ in shown) // 2)
    print(f"\n{title}")
    print(f"  {'':28s}{'now':>10s}{'run start':>11s}   {'whole run':12s}  last {w} it vs {w} before")
    mean = lambda v: st.fmean(v) if v else None
    for label, s, better, key in shown:
        values = [v for _, v in s]
        text, worse = trend(values, better)
        floor = next((f for prefix, f in FLOOR.items() if key.startswith(prefix)), 0.0)
        if max(abs(v) for v in values[-2 * w:]) < floor:
            worse = False
        # A small uptick after a large improvement is noise, not a regression:
        # the rise must be >= 10% of the range the metric covered this run.
        if worse and abs(mean(values[-w:]) - mean(values[-2 * w:-w])) < 0.1 * (max(values) - min(values)):
            worse = False
        if worse:
            warnings.append(f"{label} got worse over the last {w} iterations ({text.split()[-1]})")
        now, start = mean(values[-NOW:]), mean(values[:NOW])
        if better and not worse and len(values) >= 4 * NOW and max(abs(start), abs(now)) >= floor \
                and abs(start) > 1e-9:
            change = (now - start) / abs(start)
            if (change > RUN_WORSE) if better == "down" else (change < -RUN_WORSE):
                warnings.append(f"{label} is {change:+.0%} worse than at the start of the run")
        print(f"  {label:28s}{fmt(mean(values[-NOW:])):>10s}{fmt(mean(values[:NOW])):>11s}   {sparkline(values):12s}  {text}")


def curriculum(blocks):
    keys = sorted({k for b in blocks.values() for k in b if k.startswith("Curriculum/")})
    if not keys:
        return
    last = blocks[max(blocks)]
    print("\nCURRICULUM")
    for key in keys:
        values = [v for _, v in series(blocks, key)]
        print(f"  {key.split('/', 1)[1]:40s}{fmt(last.get(key)):>10s}   {sparkline(values)}")


def all_terms(blocks, run_dir):
    """Every reward term as weighted by the run (reward_components.csv), then
    the episode endings from the log."""
    import csv
    path = run_dir / "reward_components.csv"
    if path.is_file():
        with path.open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r.get("weighted_value") not in (None, "")]
        terms = {}
        for r in rows:
            terms.setdefault(r["name"], []).append((int(float(r["iteration"])), float(r["weighted_value"])))
        print(f"\nREWARD TERMS (weighted, as trained)")
        for name, s in sorted(terms.items(), key=lambda kv: -abs(sorted(kv[1])[-1][1])):
            values = [v for _, v in sorted(s)][WARMUP:]
            if values:
                print(f"  {name:36s}{fmt(st.fmean(values[-NOW:])):>10s}{fmt(st.fmean(values[:NOW])):>11s}   {sparkline(values)}")
    print("\nEPISODE ENDINGS (share of episodes)")
    for key in sorted(k for k in blocks[max(blocks)] if k.startswith("Episode_Termination/")):
        values = [v for _, v in series(blocks, key)]
        print(f"  {key.split('/', 1)[1]:36s}{fmt(st.fmean(values[-NOW:])):>10s}{fmt(st.fmean(values[:NOW])):>11s}   {sparkline(values)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if not args.log.is_file():
        print("\n  Training log not found yet.")
        return 0
    blocks = parse_log(args.log)
    if not blocks:
        print("\n  No training iterations logged yet.")
        return 0
    try:
        import yaml
        cfg = yaml.safe_load((args.run_dir / "resolved_config.yaml").read_text()) or {}
    except Exception:
        cfg = {}
    source = cfg.get("policy_initialization", {}).get("source", {}).get("path", "")
    heights = cfg.get("terrain", {}).get("riser_heights_m")
    purpose = cfg.get("experiment", {}).get("purpose", "")
    print()
    if purpose:
        print(f"Purpose: {purpose[:150]}{'...' if len(purpose) > 150 else ''}")
    if source or heights:
        start = "/".join(pathlib.Path(source).parts[-2:]) if source else "N/A"
        terrain = ", ".join("flat" if h == 0 else f"{h * 100:g} cm" for h in heights) if heights else "N/A"
        print(f"Start:   {start}   |   terrain: {terrain}")
    warnings = []
    section("BEHAVIOUR DURING TRAINING  (includes exploration noise; trace_report.py decides)",
            BEHAVIOUR, blocks, warnings)
    section("LEARNING", LEARNING, blocks, warnings)
    curriculum(blocks)
    if args.all:
        all_terms(blocks, args.run_dir)
    nan = [it for it, b in blocks.items() if any(not math.isfinite(v) for v in b.values())]
    if nan:
        warnings.insert(0, f"non-finite values in the log from iteration {min(nan)}")
    print("\nCHECK")
    for text in warnings or ["nothing getting worse"]:
        print(f"  {'!!' if warnings else 'ok'} {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
