#!/usr/bin/env python3
"""Show imitation-stack status without touching the pure-RL process."""

from __future__ import annotations

import json
import sys
from pathlib import Path


MOTION_IMITATION_ROOT = Path(__file__).resolve().parents[2]
if str(MOTION_IMITATION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_IMITATION_ROOT))

from g1_stair_imitation.paths import REFERENCE_FILE, RUNS_ROOT  # noqa: E402
from g1_stair_imitation.reference import (  # noqa: E402
    load_g1_stair_imitation_reference,
)


def latest_run_dir() -> Path | None:
    if not RUNS_ROOT.is_dir():
        return None
    candidates = sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir())
    return candidates[-1] if candidates else None


def main() -> int:
    print("=" * 72)
    print(" G1 IMITATION STATUS")
    print("=" * 72)

    if REFERENCE_FILE.is_file():
        ref = load_g1_stair_imitation_reference(REFERENCE_FILE)
        print("Reference :", REFERENCE_FILE.name)
        print("Frames    :", ref.frame_count)
        print("Source    :", f"{ref.source_frame_start}->{ref.source_frame_end}")
        print("Taskspace :", f"{int(ref.taskspace_valid.sum())}/{ref.frame_count}")
        print("Pose DOF  :", int((ref.pose_mask > 0.0).sum()))
    else:
        print("Reference : MISSING")

    run_dir = latest_run_dir()
    print()
    if run_dir is None:
        print("Latest run: none")
        print("State     : scaffold/reference stage")
        return 0

    print("Latest run:", run_dir.name)
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        print("State     :", data.get("status", "unknown"))
        print("Stage     :", data.get("stage", "unknown"))
    else:
        print("State     : no manifest.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
