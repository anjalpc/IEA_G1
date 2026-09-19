#!/usr/bin/env python3
"""Train the registered G1 stair task with structured metrics."""

from __future__ import annotations

import argparse
from datetime import datetime
import inspect
import os
from pathlib import Path
import shutil
import sys

from isaaclab.app import AppLauncher


UNITREE_RL_ROOT = Path(os.environ["UNITREE_RL_LAB_ROOT"])
RSL_RL_SCRIPTS = UNITREE_RL_ROOT / "scripts" / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPTS))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int)
parser.add_argument("--task", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from simulation.isaac_lab.g1_stairs.evaluation.isaac_training_metrics import (
        InstrumentedRslRlVecEnvWrapper,
        MetricsOnPolicyRunnerMixin,
    )

    metrics_dir_value = os.environ.get("IEA_G1_METRICS_DIR")
    run_id = os.environ.get("IEA_G1_RUN_ID")
    if not metrics_dir_value or not run_id:
        raise RuntimeError(
            "IEA_G1_METRICS_DIR and IEA_G1_RUN_ID are required"
        )

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_dir.resolve()
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = InstrumentedRslRlVecEnvWrapper(env)

    class MetricsOnPolicyRunner(MetricsOnPolicyRunnerMixin, OnPolicyRunner):
        pass

    runner = MetricsOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(log_dir),
        device=agent_cfg.device,
    )
    runner.configure_metrics(Path(metrics_dir_value), run_id)
    runner.add_git_repo_to_log(__file__)

    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, str(log_dir))
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        params_dir / Path(inspect.getfile(env_cfg.__class__)).name,
    )

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
