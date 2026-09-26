#!/usr/bin/env python3
"""Verify artifacts for this specific smoke; does not certify learned climbing."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.physical_step_config import parse_physical_step_config


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id, "run ID must be a single directory name")
    run = ROOT / "experiments/g1_stairs/runs" / args.run_id
    config_path = run / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    require(parse_physical_step_config(config) is not None, "not a physical 043 run")
    require(config["execution"]["maximum_iterations"] == 1, "not a one-iteration smoke")
    require(config["execution"]["environments"] == 32, "not a 32-environment smoke")
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    require(manifest["run"]["status"] == "completed", "process did not exit successfully")
    require(manifest["run"]["exit_code"] == 0, "nonzero process exit")
    require(manifest["git"]["dirty"] is False, "dirty source checkout")
    require(hashlib.sha256(config_path.read_bytes()).hexdigest() == manifest["configuration"]["resolved_sha256"],
            "resolved config hash mismatch")
    require((run / "training_complete.yaml").is_file(), "missing completion marker")
    summary = yaml.safe_load((run / "physical_step043_summary.yaml").read_text())
    require(summary["ppo_learn_returned"] is True, "PPO did not finish")
    require(summary["environment_steps"] == 32 * 24, "unexpected physical snapshot count")
    require(summary["calibrated_env_steps"] > 0, "no loaded lower-floor reference was acquired")
    event_counts = [summary[name] for name in ("first_support_count", "base_transfer_count", "both_feet_count", "completion_count")]
    require(all(a >= b >= 0 for a, b in zip(event_counts, event_counts[1:])), "transition counts out of order")
    with (run / "training_metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    require(len(metrics) == 1, "expected exactly one training row")
    require(float(metrics[0]["nan_count"]) == 0, "non-finite runtime values detected")
    extras = json.loads(metrics[0]["extra_json"])
    require(extras["step043_environment_steps"] == 768, "physical metrics not persisted")
    with (run / "reward_components.csv").open() as stream:
        rewards = list(csv.DictReader(stream))
    weights = active_reward_weights(config)
    require(len(rewards) == len(weights) and {r["name"] for r in rewards} == set(weights), "unexpected active reward set")
    for row in rewards:
        for key in ("raw_value", "weight", "weighted_value"):
            require(math.isfinite(float(row[key])), "non-finite reward component")
        require(float(row["weight"]) == weights[row["name"]], "reward weight differs from resolved config")
    checkpoint = (ROOT / "experiments/g1_stairs/logs/rsl_rl/unitree_iea_g1_29dof_stairs_v0"
                  / args.run_id / "model_0.pt")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "missing model_0.pt")
    print("PASS: clean one-iteration integration smoke; this is NOT evidence of learned climbing")
    print(yaml.safe_dump(summary, sort_keys=False))
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
