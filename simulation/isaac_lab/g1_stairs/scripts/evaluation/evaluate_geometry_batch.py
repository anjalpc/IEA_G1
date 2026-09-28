#!/usr/bin/env python3
"""Evaluate V03 using exactly the training outcome contract and fixed quotas."""

import argparse
import csv
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys


def dispatch_isolated_batches():
    """Run simulator workers separately; the coordinator never imports Isaac."""
    import signal
    import subprocess
    from collections import Counter

    worker_flag = "--_geometry-worker"
    if worker_flag in sys.argv:
        sys.argv.remove(worker_flag)
        return
    if "--help" in sys.argv or "-h" in sys.argv:
        return

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--heights", type=float, nargs="+", default=[0.0, 0.04])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes-per-env", type=int, default=1)
    parser.add_argument("--batch-timeout", type=float, default=300.0)
    options, remaining = parser.parse_known_args()

    if options.num_envs < 1 or options.episodes_per_env < 1:
        parser.error("environment and episode counts must be positive")
    if not 0 < options.batch_timeout < float("inf"):
        parser.error("--batch-timeout must be positive and finite")
    if (any(not 0 <= h <= 0.20 for h in options.heights)
            or len(set(options.heights)) != len(options.heights)):
        parser.error("heights must be distinct and in [0, 0.20]")
    if len(set(options.seeds)) != len(options.seeds):
        parser.error("seeds must be distinct")

    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive directory prevents mixing this invocation with an earlier sweep.
    batches = output / "batches"
    if (output / "summary.json").exists() or (output / "episodes.csv").exists():
        raise RuntimeError("Output already contains results; choose a new directory")
    batches.mkdir(exist_ok=False)
    all_rows, records = [], []
    first = None
    identity = None
    fieldnames = None

    for index, height in enumerate(options.heights):
        for seed in options.seeds:
            directory = batches / f"height_{index:02d}_seed_{seed}"
            directory.mkdir()
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), worker_flag,
                *remaining,
                "--heights", str(height), "--seeds", str(seed),
                "--num-envs", str(options.num_envs),
                "--episodes-per-env", str(options.episodes_per_env),
                "--output-dir", str(directory),
            ]
            print(f"[BATCH] Starting height={height} seed={seed}", flush=True)
            with (directory / "process.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=options.batch_timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    print(f"[BATCH] Interrupted/timed out; inspect {directory}", flush=True)
                    raise
            (directory / "process_exit_code.txt").write_text(str(returncode) + "\n")
            if returncode != 0:
                raise RuntimeError(
                    f"Worker exited {returncode}; inspect {directory / 'process.log'}"
                )

            summary = json.loads((directory / "summary.json").read_text())
            with (directory / "episodes.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            if fieldnames is None:
                fieldnames = fields
            elif fieldnames != fields:
                raise RuntimeError("Episode CSV columns differ between workers")

            expected = {
                (env_id, episode)
                for env_id in range(options.num_envs)
                for episode in range(options.episodes_per_env)
            }
            actual = {(int(r["env_id"]), int(r["episode"])) for r in rows}
            if actual != expected or len(rows) != len(expected):
                raise RuntimeError(f"Missing/duplicate episode quotas: {directory}")
            if any(float(r["height_m"]) != height or int(r["seed"]) != seed for r in rows):
                raise RuntimeError(f"Incorrect height/seed: {directory}")
            if (summary["seeds"] != [seed]
                    or summary["num_envs"] != options.num_envs
                    or summary["episodes_per_env"] != options.episodes_per_env
                    or len(summary["by_height"]) != 1):
                raise RuntimeError(f"Incorrect worker metadata: {directory}")
            result = summary["by_height"][0]
            outcomes = Counter(r["outcome"] for r in rows)
            if (result["height_m"] != height
                    or result["episodes"] != len(rows)
                    or result["successes"] != outcomes["success"]
                    or {k: v for k, v in result["outcomes"].items() if v} != dict(outcomes)):
                raise RuntimeError(f"Summary disagrees with episodes: {directory}")

            signature = (
                summary["model_sha256"], summary["config_sha256"],
                json.dumps(summary["episode_contract"], sort_keys=True),
                summary["unitree_baseline"],
                tuple(sorted(summary["forbidden_contact_bodies_observed"])),
            )
            if first is None:
                first, identity = summary, signature
            elif signature != identity:
                raise RuntimeError("Checkpoint/config/contract differs between workers")
            all_rows.extend(rows)
            records.append(str(directory.relative_to(output)))
            print(
                f"[BATCH] Complete height={height} seed={seed}: "
                f"{result['successes']}/{len(rows)}", flush=True,
            )

    combined = dict(first)
    combined["seeds"] = options.seeds
    combined["execution_mode"] = "isolated_process_per_height_seed"
    combined["worker_directories"] = records
    combined["by_height"] = []
    outcome_names = first["by_height"][0]["outcomes"]
    for height in options.heights:
        rows = [r for r in all_rows if float(r["height_m"]) == height]
        outcomes = Counter(r["outcome"] for r in rows)
        combined["by_height"].append({
            "height_m": height,
            "task": "flat_traversal" if height == 0 else "stair_traversal",
            "episodes": len(rows),
            "successes": outcomes["success"],
            "success_rate": outcomes["success"] / len(rows),
            "outcomes": {name: outcomes[name] for name in outcome_names},
        })
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    temporary = output / "summary.json.tmp"
    with temporary.open("x") as stream:
        json.dump(combined, stream, indent=2)
    temporary.replace(output / "summary.json")
    print(json.dumps(combined, indent=2), flush=True)
    print(f"[PASS] All batches validated: {output / 'summary.json'}", flush=True)
    raise SystemExit(0)


dispatch_isolated_batches()

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
from simulation.isaac_lab.g1_stairs.unitree_bootstrap import bootstrap_unitree_rl_lab
bootstrap_unitree_rl_lab(os.environ.get("UNITREE_RL_LAB_ROOT", "/workspace/unitree_rl_lab"))

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-IEA-G1-29dof-Stairs-v0")
parser.add_argument("--resolved-config", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--unitree-baseline", action="store_true")
parser.add_argument("--heights", type=float, nargs="+", default=[0.0, 0.04])
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes-per-env", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if bool(args.checkpoint) == args.unitree_baseline:
    parser.error("select exactly one of --checkpoint or --unitree-baseline")
if args.num_envs < 1 or args.episodes_per_env < 1:
    parser.error("environment and episode counts must be positive")
if any(not 0 <= h <= 0.20 for h in args.heights) or len(set(args.heights)) != len(args.heights):
    parser.error("heights must be distinct values in [0, 0.20]")
if len(set(args.seeds)) != len(args.seeds):
    parser.error("seeds must be distinct")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401
    from simulation.isaac_lab.g1_stairs.reward_runtime import load_resolved_config
    from simulation.isaac_lab.g1_stairs.geometry_episode import Result
    from simulation.isaac_lab.g1_stairs.geometry_episode_runtime import apply_geometry_episode_contract
    from simulation.isaac_lab.g1_stairs.policy_initialization import initialize_actor_from_unitree_onnx
    from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
        apply_resolved_replay_config, apply_resolved_replay_agent_config,
    )

    config_path = Path(args.resolved_config).resolve()
    resolved = load_resolved_config(config_path)
    if not resolved.get("episode_contract", {}).get("enabled", False):
        raise ValueError("batch evaluation requires an enabled V03 episode_contract")
    model = Path(args.checkpoint or resolved["policy_initialization"]["source"]["path"]).resolve()
    if args.unitree_baseline and resolved["policy_initialization"]["type"] != "unitree_velocity_onnx_transfer":
        raise ValueError("baseline requires the original Unitree ONNX source")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = ["height_m", "seed", "env_id", "episode", "outcome", "success",
              "duration_s", "supported_platform", "entered_stairs", "terminal_phase"]
    all_rows = []
    coverage = None
    def persist_completed_summary():
        summaries = []
        for height in args.heights:
            rows = [r for r in all_rows if r["height_m"] == height]
            successes = sum(r["success"] for r in rows)
            summaries.append({
                "height_m": height, "task": "flat_traversal" if height == 0 else "stair_traversal",
                "episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
                "outcomes": {r.name.lower(): sum(v["outcome"] == r.name.lower() for v in rows)
                             for r in Result if r != Result.RUNNING},
            })
        summary = {
            "schema_version": 1, "episode_contract": resolved["episode_contract"],
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "model": str(model), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "unitree_baseline": args.unitree_baseline, "seeds": args.seeds,
            "episodes_per_env": args.episodes_per_env, "num_envs": args.num_envs,
            "forbidden_contact_bodies_observed": coverage, "by_height": summaries,
        }
        with (output / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2)
        print(json.dumps(summary, indent=2))
        print("[INFO] Complete summary saved before cleanup.", flush=True)

    # Exclusive creation prevents accidentally mixing repeated evaluations.
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for height in args.heights:
            for seed in args.seeds:
                env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                                        use_fabric=True, entry_point_key="play_env_cfg_entry_point")
                apply_resolved_replay_config(env_cfg, config_path)
                cfg = env_cfg.scene.terrain.terrain_generator
                cfg.num_rows = 1
                cfg.num_cols = args.num_envs
                cfg.curriculum = False
                stairs = cfg.sub_terrains["three_step_ascent"]
                stairs.riser_heights = None
                stairs.riser_height_range = (height, height)
                env_cfg.scene.terrain.max_init_terrain_level = 0
                env_cfg.seed = seed
                env_cfg.commands.base_velocity.debug_vis = False
                apply_geometry_episode_contract(env_cfg, resolved, promote=False)
                args.seed = seed
                agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
                agent_cfg.seed = seed
                apply_resolved_replay_agent_config(agent_cfg, resolved)
                print(f"[INFO] Creating environment: height={height}, seed={seed}", flush=True)
                faulthandler.dump_traceback_later(30, repeat=True)
                try:
                    env = RslRlVecEnvWrapper(
                        gym.make(args.task, cfg=env_cfg),
                        clip_actions=agent_cfg.clip_actions,
                    )
                finally:
                    faulthandler.cancel_dump_traceback_later()
                print("[INFO] Environment creation finished", flush=True)
                try:
                    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                    if args.unitree_baseline:
                        initialize_actor_from_unitree_onnx(runner, model)
                    else:
                        runner.load(str(model), load_optimizer=False)
                    policy = runner.get_inference_policy(device=env.unwrapped.device)
                    obs = env.get_observations()
                    quotas = [0] * args.num_envs
                    limit = (env.unwrapped.max_episode_length + 1) * args.episodes_per_env
                    with torch.inference_mode():
                        for _ in range(limit):
                            obs, _, _, _ = env.step(policy(obs))
                            runtime = env.unwrapped._g1_geometry_episode
                            coverage = runtime.forbidden_names
                            completed = runtime.last_completed
                            if completed is None or completed["step"] != env.unwrapped.common_step_counter:
                                continue
                            ids = completed["env_ids"].cpu().tolist()
                            values = {k: v.cpu().tolist() for k, v in completed.items()
                                      if k not in ("step", "env_ids")}
                            for index, env_id in enumerate(ids):
                                if quotas[env_id] >= args.episodes_per_env:
                                    continue
                                reason = Result(values["reason"][index])
                                row = dict(
                                    height_m=height, seed=seed, env_id=env_id,
                                    episode=quotas[env_id], outcome=reason.name.lower(),
                                    success=reason == Result.SUCCESS,
                                    duration_s=values["episode_steps"][index] * env.unwrapped.step_dt,
                                    supported_platform=values["supported_platform"][index],
                                    entered_stairs=values["entered_stairs"][index],
                                    terminal_phase=("approach", "ascent", "platform")[values["phase"][index]],
                                )
                                writer.writerow(row)
                                all_rows.append(row)
                                quotas[env_id] += 1
                            if min(quotas) >= args.episodes_per_env:
                                break
                    if min(quotas) < args.episodes_per_env:
                        raise RuntimeError("incomplete evaluation quota; results are not a completed evaluation")
                    stream.flush()
                    print(f"[INFO] evaluated height={height} seed={seed}: {sum(quotas)} episodes", flush=True)
                    if height == args.heights[-1] and seed == args.seeds[-1]:
                        persist_completed_summary()
                finally:
                    print("[INFO] Starting env.close()", flush=True)
                    faulthandler.dump_traceback_later(30, repeat=True)
                    try:
                        env.close()
                    finally:
                        faulthandler.cancel_dump_traceback_later()
                    print("[INFO] Finished env.close()", flush=True)



if __name__ == "__main__":
    try:
        main()
    finally:
        print("[INFO] Starting simulation_app.close()", flush=True)
        faulthandler.dump_traceback_later(30, repeat=True)
        try:
            simulation_app.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
        print("[INFO] Finished simulation_app.close()", flush=True)
