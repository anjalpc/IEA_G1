#!/usr/bin/env python3
"""Evaluate V03 using exactly the training outcome contract and fixed quotas."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

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
                env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
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
                finally:
                    env.close()
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


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
