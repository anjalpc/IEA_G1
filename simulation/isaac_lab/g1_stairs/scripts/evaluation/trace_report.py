#!/usr/bin/env python3
"""Pass/fail report for a trace batch (record_view.py --trace-only).

Pure Python, runs on the host. Reads <dir>/**/model_<ckpt>_h<cm>cm_seed<n>_side/
trace.jsonl, measures every episode, prints one block per checkpoint with a
line per terrain height and a gate verdict, e.g.

  python3 trace_report.py experiments/g1_stairs/outputs/evaluation/stair_p51
  python3 trace_report.py <dir> --seeds            # also one line per episode

Course (env_cfg, unchanged since patch 21): first riser at x 1.00 m, treads
0.25 m deep, 3 steps, platform from 1.75 m. Plan targets sit at tread centres.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from footstep_gait_report import first_episode  # noqa: E402

FIRST_RISER_M, TREAD_M, STEPS = 1.00, 0.25, 3
PLATFORM_M = FIRST_RISER_M + STEPS * TREAD_M
FOOT_FORCE = (6, 12)             # contact_net_forces_w rows of the two ankle roll links
STRIKE_N = 50.0                  # horizontal force on a toe within 6 cm of a riser
# Nose landing: a touchdown with the heel over the front edge of its tread and
# > STRIKE_N horizontal force within 4 frames (the sole lands on the nosing and
# is pushed back). The toe-strike rule and the training stumble term (horizontal
# > 4x vertical) both miss it: run 20261106_210725 model_15200, right foot onto
# tread 2, 1112 N horizontal / 555 N vertical, heel 1.1 cm over the edge.
FALL_TILT_DEG = 35.0
NAME = re.compile(r"model_(\d+)_h(\d+)cm_seed(\d+)")

# Gate. top: highest height in the batch; lower: other stair heights.
GATE = dict(top_clean=0.8, lower_clean=0.9, first_heel_cm=1.0, first_heel_share=0.9,
            flat_drift_degps=0.7, flat_lateral_cm=35.0, speed_ratio=(0.85, 1.20))


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _tilt_deg(q):
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def _slope(ts, ys):
    if len(ts) < 20:
        return None
    mt, my = st.fmean(ts), st.fmean(ys)
    den = sum((t - mt) ** 2 for t in ts)
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / den if den else None


def episode(rows, stairs):
    """Per-episode measurements (course frame, first episode only).

    stride: same foot, lift-off to touchdown (>= 6 cm); knee: mean stance
    angle; toe-out: foot yaw minus pelvis yaw in stance, outwards positive;
    landing: median peak vertical force over 6 steps after touchdown; fall:
    terminated with the base tilted > 35 deg; stall: other termination."""
    rows = [r for r in first_episode(rows) if "footstep_landing_audit" in r]
    o = rows[0]["env_origin_w"]
    ts, yaw, vx, cmd, knee = [], [], [], [], {0: [], 1: []}
    toe_out, land, steps = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}
    plat_t, plat_yaw = [], []
    misses = planned = strikes = nose = 0
    first_heel = None
    lift = {}
    unwrap = 0.0
    for k, r in enumerate(rows):
        a = r["footstep_landing_audit"]
        sup, bounds = a["supported_lr"], a["sole_bounds_local"]
        h = _yaw(r["root_quat_w"])
        if yaw:
            unwrap += (math.degrees(h) - unwrap + 180) % 360 - 180
        else:
            unwrap = math.degrees(h)
        if r["episode_time_s"] > 1.0:
            v = r["root_lin_vel_w"]
            ts.append(r["episode_time_s"]); yaw.append(unwrap)
            vx.append(v[0] * math.cos(h) + v[1] * math.sin(h))
            cmd.append(r["command_base_velocity"][0])
            mid = sum(bounds[f][0] + bounds[f][1] for f in (0, 1)) / 4
            if stairs and mid >= PLATFORM_M + 0.25:
                plat_t.append(r["episode_time_s"]); plat_yaw.append(unwrap)
        for f in (0, 1):
            heel, toe = bounds[f][0], bounds[f][1]
            centre = 0.5 * (heel + toe)
            force = r["contact_net_forces_w"][FOOT_FORCE[f]]
            if stairs and any(abs(toe - (FIRST_RISER_M + i * TREAD_M)) < 0.06 for i in range(STEPS)):
                strikes += math.hypot(force[0], force[1]) > STRIKE_N
            if sup[f]:
                knee[f].append(r["joint_pos_rad"][9 + f])
                d = (math.degrees(_yaw(r["foot_quat_w"][f]) - h) + 180) % 360 - 180
                toe_out[f].append(d if f == 0 else -d)
            if k == 0:
                continue
            was = rows[k - 1]["footstep_landing_audit"]["supported_lr"][f]
            if was and not sup[f]:
                lift[f] = centre
            if sup[f] and not was:
                land[f].append(max(abs(rr["contact_net_forces_w"][FOOT_FORCE[f]][2]) for rr in rows[k:k + 6]))
                if stairs and FIRST_RISER_M <= centre < PLATFORM_M:
                    edge = FIRST_RISER_M + TREAD_M * int((centre - FIRST_RISER_M) // TREAD_M)
                    push = max(math.hypot(*rr["contact_net_forces_w"][FOOT_FORCE[f]][:2]) for rr in rows[k:k + 4])
                    nose += heel < edge and push > STRIKE_N
                if f in lift and centre - lift[f] >= 0.06:
                    steps[f].append(centre - lift.pop(f))
                    if stairs and r["footstep_rolling_clock"]["on_stair_plan"] and centre < PLATFORM_M + 0.15:
                        planned += 1
                        misses += not a["raw_acquisition"]
                    if stairs and first_heel is None and FIRST_RISER_M <= centre < FIRST_RISER_M + TREAD_M:
                        first_heel = 100 * (heel - FIRST_RISER_M)
    last = rows[-1]
    tilt = _tilt_deg(last["root_quat_w"])
    ended = "time_out" if last.get("time_out") else ("fall" if last.get("terminated") and tilt > FALL_TILT_DEG
                                                      else "stall" if last.get("terminated") else "end")
    climbed = any(r["footstep_rolling_clock"].get("stairs_climbed") for r in rows)
    m = lambda v: st.fmean(v) if v else None
    med = lambda v: st.median(v) if v else None
    return dict(
        T=last["episode_time_s"], ended=ended, climbed=climbed if stairs else None,
        clean=(climbed and misses == 0 and ended not in ("fall", "stall")) if stairs else None,
        misses=misses if stairs else None, strikes=strikes if stairs else None, first_heel=first_heel,
        nose=nose if stairs else None,
        vx=m(vx), cmd=m(cmd), drift=_slope(ts, yaw), plat_drift=_slope(plat_t, plat_yaw) if stairs else None,
        lateral=100 * (last["root_pos_w"][1] - o[1]),
        stride=(m(steps[0]), m(steps[1])), knee=(m(knee[0]), m(knee[1])),
        toe_out=(m(toe_out[0]), m(toe_out[1])), land=(med(land[0]), med(land[1])))


def _f(v, fmt, na="N/A"):
    return na if v is None else format(v, fmt)


def _pair(p, fmt, scale=1.0):
    return "/".join(_f(None if v is None else v * scale, fmt) for v in p)


def _mean_pair(eps, key):
    return tuple(st.fmean([e[key][i] for e in eps if e[key][i] is not None] or [math.nan]) for i in (0, 1))


def judge(height, eps, top):
    """List of (check, ok) for one checkpoint/height."""
    n = len(eps)
    if height == 0:
        drift = max(abs(e["drift"] or 0) for e in eps)
        lat = st.fmean(abs(e["lateral"]) for e in eps)
        ratio = st.fmean(e["vx"] / e["cmd"] for e in eps if e["cmd"])
        lo, hi = GATE["speed_ratio"]
        return [(f"turn <= {GATE['flat_drift_degps']} deg/s", drift <= GATE["flat_drift_degps"]),
                (f"sideways <= {GATE['flat_lateral_cm']:.0f} cm", lat <= GATE["flat_lateral_cm"]),
                (f"speed {lo:.2f}-{hi:.2f} x cmd", lo <= ratio <= hi),
                ("no falls", all(e["ended"] != "fall" for e in eps))]
    need = GATE["top_clean"] if height == top else GATE["lower_clean"]
    clean = sum(bool(e["clean"]) for e in eps)
    checks = [(f"clean >= {need:.0%}", clean >= math.ceil(need * n - 1e-9)),
              ("no falls/stalls", all(e["ended"] not in ("fall", "stall") for e in eps))]
    if height == top:
        heels = [e["first_heel"] for e in eps if e["first_heel"] is not None]
        share = sum(h >= GATE["first_heel_cm"] for h in heels) / len(heels) if heels else 0.0
        checks.append((f"first heel >= {GATE['first_heel_cm']:.0f} cm in {GATE['first_heel_share']:.0%}",
                       share >= GATE["first_heel_share"]))
        checks.append(("no toe strikes", all(e["strikes"] == 0 for e in eps)))
        checks.append(("no nose landings", all(e["nose"] == 0 for e in eps)))
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--seeds", action="store_true", help="also print one line per episode")
    args = ap.parse_args(argv)
    groups = {}
    for path in sorted(pathlib.Path(args.directory).rglob("trace.jsonl")):
        hit = NAME.search(str(path.parent.name))
        if not hit:
            continue
        ckpt, cm, seed = (int(g) for g in hit.groups())
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        groups.setdefault(ckpt, {}).setdefault(cm, []).append((seed, episode(rows, cm > 0)))
    if not groups:
        print(f"no traces under {args.directory}")
        return 2
    top = max(cm for g in groups.values() for cm in g)
    verdict = True
    for ckpt in sorted(groups):
        print(f"\n=== checkpoint {ckpt} " + "=" * 60)
        print(f"{'terrain':8s}{'n':>3s} {'climbed':>8s}{'clean':>6s}{'falls':>6s}{'stalls':>7s}{'strikes':>8s}{'nose':>5s}"
              f"{'1st heel cm':>12s}{'speed/cmd':>11s}{'turn deg/s':>11s}{'side cm':>8s}{'plat turn':>10s}"
              f"  stride L/R cm knee L/R   toe-out L/R  landing L/R N")
        ok_all = True
        for cm in sorted(groups[ckpt]):
            eps = [e for _, e in sorted(groups[ckpt][cm])]
            n, stairs = len(eps), cm > 0
            cnt = lambda pred: sum(1 for e in eps if pred(e))
            heels = [e["first_heel"] for e in eps if e["first_heel"] is not None]
            plat = [e["plat_drift"] for e in eps if e["plat_drift"] is not None]
            ratio = lambda pred: f"{cnt(pred)}/{n}" if stairs else "-"
            print(f"{('flat' if not stairs else f'{cm} cm'):8s}{n:3d} "
                  f"{ratio(lambda e: e['climbed']):>8s}{ratio(lambda e: e['clean']):>6s}"
                  f"{cnt(lambda e: e['ended'] == 'fall'):>6d}{cnt(lambda e: e['ended'] == 'stall'):>7d}"
                  f"{str(sum(e['strikes'] for e in eps)) if stairs else '-':>8s}"
                  f"{str(sum(e['nose'] for e in eps)) if stairs else '-':>5s}"
                  f"{_f(st.median(heels) if heels else None, '+.1f') if stairs else '-':>12s}"
                  f"{st.fmean(e['vx'] for e in eps):6.2f}/{st.fmean(e['cmd'] for e in eps):.2f}"
                  f"{st.fmean(e['drift'] or 0 for e in eps):+11.2f}{st.fmean(e['lateral'] for e in eps):+8.0f}"
                  f"{_f(st.fmean(plat) if plat else None, '+.2f') if stairs else '-':>10s}"
                  f"   {_pair(_mean_pair(eps, 'stride'), '.0f', 100):>11s}  {_pair(_mean_pair(eps, 'knee'), '.2f'):>9s}"
                  f"  {_pair(_mean_pair(eps, 'toe_out'), '.1f'):>11s}  {_pair(_mean_pair(eps, 'land'), '.0f'):>11s}")
            checks = judge(cm, eps, top)
            ok = all(c for _, c in checks)
            ok_all &= ok
            print("        " + "  ".join(f"[{'PASS' if c else 'FAIL'}] {name}" for name, c in checks))
            if args.seeds:
                for seed, e in sorted(groups[ckpt][cm]):
                    print(f"          seed {seed:2d}: {e['ended']:8s} T {e['T']:4.1f}"
                          + (f" climbed {int(e['climbed'])} misses {e['misses']} strikes {e['strikes']} nose {e['nose']}"
                             f" 1st heel {_f(e['first_heel'], '+.1f')}" if stairs else "")
                          + f" speed {e['vx']:.2f} turn {_f(e['drift'], '+.2f')} side {e['lateral']:+.0f}")
        print(f"RESULT checkpoint {ckpt}: {'PASS' if ok_all else 'FAIL'}")
        verdict = ok_all
    print(f"\nLatest checkpoint ({max(groups)}): {'PASS' if verdict else 'FAIL'} "
          f"(top height {top} cm; gate {GATE})")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
