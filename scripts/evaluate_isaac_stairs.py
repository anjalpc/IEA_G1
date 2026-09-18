#!/usr/bin/env python3
"""Evaluate one G1 stair checkpoint with reward-independent outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNITREE_RL_ROOT = Path(
    os.environ.get("UNITREE_RL_LAB_ROOT", "/workspace/unitree_rl_lab")
)
RSL_RL_SCRIPTS = UNITREE_RL_ROOT / "scripts" / "rsl_rl"
if not RSL_RL_SCRIPTS.is_dir():
    raise RuntimeError(f"Unitree RSL-RL scripts missing: {RSL_RL_SCRIPTS}")
sys.path.insert(0, str(RSL_RL_SCRIPTS))

import cli_args  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser(
    description="Run one independent G1 stair outcome evaluation."
)
parser.add_argument("--task", default="Unitree-IEA-G1-29dof-Stairs-v0")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--max-steps", type=int, default=900)
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.checkpoint:
    parser.error("--checkpoint is required")
if args.max_steps < 1:
    parser.error("--max-steps must be positive")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def disable_native_terminations(env_cfg) -> tuple[str, ...]:
    """Let the independent evaluator own all terminal decisions."""
    disabled = []
    for name, value in vars(env_cfg.terminations).items():
        if name.startswith("_") or value is None:
            continue
        setattr(env_cfg.terminations, name, None)
        disabled.append(name)
    if not disabled:
        raise RuntimeError("no native termination terms were found")
    return tuple(sorted(disabled))


def main() -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import unitree_rl_lab.tasks  # noqa: F401
    import isaac_lab.g1_stairs.register  # noqa: F401

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from experiments.evaluation.isaac_lab_adapter import (
        capture_isaac_lab_state,
        to_step_sample,
    )
    from experiments.evaluation.metrics import RunMetricsWriter
    from experiments.evaluation.stair_outcomes import StairOutcomeEvaluator

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(retrieve_file_path(args.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    env_cfg.commands.base_velocity.debug_vis = False
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    disabled_terminations = disable_native_terminations(env_cfg)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    evaluator = StairOutcomeEvaluator()
    writer = RunMetricsWriter(output_dir, output_dir.name)
    result = evaluator.update(
        to_step_sample(capture_isaac_lab_state(env.unwrapped, time_s=0.0))
    )
    steps = 0

    while result is None and steps < args.max_steps and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        steps += 1
        timed_out = steps >= args.max_steps
        state = capture_isaac_lab_state(
            env.unwrapped,
            time_s=steps * float(env.unwrapped.step_dt),
            timed_out=timed_out,
        )
        result = evaluator.update(to_step_sample(state))

    if result is None:
        raise RuntimeError("simulation stopped before an outcome was produced")

    writer.record_episode(
        result,
        checkpoint=str(checkpoint),
        evaluation_seed=args.seed,
        episode_index=0,
    )
    summary = writer.write_summary()
    metadata = {
        "schema": "iea_g1_stair_evaluation_v1",
        "task": args.task,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "completed_steps": steps,
        "step_dt": float(env.unwrapped.step_dt),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "disabled_native_terminations": list(disabled_terminations),
        "outcome": result.outcome.value,
        "success": result.success,
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    env.close()

    print("=== INDEPENDENT STAIR EVALUATION ===")
    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_sha256={metadata['checkpoint_sha256']}")
    print(f"disabled_native_terminations={disabled_terminations}")
    print(f"completed_steps={steps}")
    print(f"outcome={result.outcome.value}")
    print(f"success={result.success}")
    print(f"episode_csv={writer.episode_path}")
    print(f"summary_json={writer.summary_path}")
    print(f"summary_episode_count={summary['episode_count']}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
