#!/usr/bin/env python3
"""Launch Unitree RL Lab train/play with the IEA stair task registered."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

# Direct execution adds scripts/, not the repository root, to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac_lab.g1_stairs.contracts import TASK_ID


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "play"}:
        raise SystemExit(
            "usage: run_isaac_stairs.py {train|play} [Unitree RL Lab arguments]"
        )
    mode = sys.argv.pop(1)
    root_value = os.environ.get("UNITREE_RL_LAB_ROOT")
    if not root_value:
        raise SystemExit("set UNITREE_RL_LAB_ROOT to the Unitree RL Lab checkout")
    if mode == "train":
        script = REPO_ROOT / "simulation" / "isaac_lab" / "g1_stairs" / "scripts" / "training" / "train_isaac_stairs.py"
    else:
        script = Path(root_value).resolve() / "scripts" / "rsl_rl" / "play.py"
    if not script.is_file():
        raise SystemExit(f"missing Unitree RL Lab launcher: {script}")
    # Registration imports Gymnasium/Isaac Lab, which only exist in the cloud
    # training environment. Delay it until local argument/path checks pass.
    from simulation.isaac_lab.g1_stairs import register as _register  # noqa: F401

    if "--task" not in sys.argv:
        sys.argv.extend(("--task", TASK_ID))

    # Unitree RL Lab launchers import sibling modules such as cli_args.
    # runpy does not automatically add the launcher's directory to
    # sys.path as direct script execution would.
    script_dir = str(script.parent)
    inserted_script_dir = script_dir not in sys.path

    if inserted_script_dir:
        sys.path.insert(0, script_dir)

    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        if inserted_script_dir:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass


if __name__ == "__main__":
    main()
