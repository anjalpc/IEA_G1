#!/usr/bin/env python3
"""Train the registered G1 stair task with structured metrics."""

from __future__ import annotations

import argparse
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

    from simulation.isaac_lab.g1_stairs.policy_initialization import (
        initialize_actor_from_unitree_onnx,
    )

    metrics_dir_value = os.environ.get("IEA_G1_METRICS_DIR")
    run_id = os.environ.get("IEA_G1_RUN_ID")
    resolved_config_value = os.environ.get("IEA_G1_RESOLVED_CONFIG")
    if not metrics_dir_value or not run_id or not resolved_config_value:
        raise RuntimeError(
            "IEA_G1_METRICS_DIR, IEA_G1_RUN_ID, and "
            "IEA_G1_RESOLVED_CONFIG are required"
        )

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = args.seed

    from simulation.isaac_lab.g1_stairs.reward_runtime import (
        apply_resolved_rewards,
        load_resolved_config,
    )

    resolved_config = load_resolved_config(Path(resolved_config_value))
    terrain_cfg = resolved_config.get("terrain", {})
    if "riser_height_m" in terrain_cfg:
        riser_height_m = float(terrain_cfg["riser_height_m"])

        if not 0.0 < riser_height_m <= 0.20:
            raise ValueError(
                "terrain.riser_height_m must be in (0.0, 0.20]"
            )

        stair_cfg = (
            env_cfg.scene.terrain.terrain_generator
            .sub_terrains["three_step_ascent"]
        )
        stair_cfg.riser_height_range = (
            riser_height_m,
            riser_height_m,
        )

        print(
            "[INFO] Stair riser_height_m:",
            riser_height_m,
            flush=True,
        )

    if "step_count" in terrain_cfg:
        step_count = int(terrain_cfg["step_count"])

        if not 1 <= step_count <= 3:
            raise ValueError(
                "terrain.step_count must be between 1 and 3"
            )

        stair_cfg = (
            env_cfg.scene.terrain.terrain_generator
            .sub_terrains["three_step_ascent"]
        )
        stair_cfg.step_count = step_count

        print(
            "[INFO] Stair step_count:",
            step_count,
            flush=True,
        )

    command_cfg = resolved_config.get("command", {})
    if "forward_velocity_mps" in command_cfg:
        forward_velocity_mps = float(
            command_cfg["forward_velocity_mps"]
        )

        if not 0.0 < forward_velocity_mps <= 0.50:
            raise ValueError(
                "command.forward_velocity_mps must be in (0.0, 0.50]"
            )

        env_cfg.commands.base_velocity.ranges.lin_vel_x = (
            forward_velocity_mps,
            forward_velocity_mps,
        )

        print(
            "[INFO] Forward command m/s:",
            forward_velocity_mps,
            flush=True,
        )

    active_rewards = apply_resolved_rewards(env_cfg, resolved_config)
    print("[INFO] Active reward stage:", resolved_config["reward_experiment"]["active_stage"], flush=True)
    print("[INFO] Active rewards:", active_rewards, flush=True)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    ppo_cfg = resolved_config.get("ppo", {})

    if "initial_action_noise_std" in ppo_cfg:
        agent_cfg.policy.init_noise_std = float(
            ppo_cfg["initial_action_noise_std"]
        )

    if "learning_rate" in ppo_cfg:
        agent_cfg.algorithm.learning_rate = float(
            ppo_cfg["learning_rate"]
        )

    if "learning_rate_schedule" in ppo_cfg:
        agent_cfg.algorithm.schedule = str(
            ppo_cfg["learning_rate_schedule"]
        )

    if "desired_kl" in ppo_cfg:
        agent_cfg.algorithm.desired_kl = float(
            ppo_cfg["desired_kl"]
        )

    print(
        "[INFO] PPO init_noise_std:",
        agent_cfg.policy.init_noise_std,
        flush=True,
    )
    print(
        "[INFO] PPO learning_rate:",
        agent_cfg.algorithm.learning_rate,
        flush=True,
    )
    print(
        "[INFO] PPO learning_rate_schedule:",
        agent_cfg.algorithm.schedule,
        flush=True,
    )
    print(
        "[INFO] PPO desired_kl:",
        agent_cfg.algorithm.desired_kl,
        flush=True,
    )

    log_root = (
        Path("experiments")
        / "g1_stairs"
        / "logs"
        / "rsl_rl"
        / agent_cfg.experiment_name
    )
    # Use the recorded experiment run ID so TensorBoard, metrics,
    # manifests, and checkpoints share one unambiguous identifier.
    log_dir = log_root / run_id
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

    initialization_cfg = resolved_config.get(
        "policy_initialization",
        {},
    )
    initialization_type = initialization_cfg.get("type")

    if initialization_type == "unitree_velocity_onnx_transfer":
        initialize_actor_from_unitree_onnx(
            runner=runner,
            onnx_path=Path(
                initialization_cfg["source"]["path"]
            ),
        )
    elif initialization_type not in (None, "random"):
        raise RuntimeError(
            "Unsupported policy initialization: "
            f"{initialization_type}"
        )

    print("[TRACE] before configure_metrics", flush=True)
    runner.configure_metrics(Path(metrics_dir_value), run_id)
    print("[TRACE] after configure_metrics", flush=True)

    print("[TRACE] before add_git_repo_to_log", flush=True)
    runner.add_git_repo_to_log(__file__)
    print("[TRACE] after add_git_repo_to_log", flush=True)

    params_dir = log_dir / "params"

    print("[TRACE] before params mkdir", flush=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    print("[TRACE] after params mkdir", flush=True)

    print("[TRACE] before dump env yaml", flush=True)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    print("[TRACE] after dump env yaml", flush=True)

    print("[TRACE] before dump agent yaml", flush=True)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    print("[TRACE] after dump agent yaml", flush=True)

    print("[TRACE] before export_deploy_cfg", flush=True)
    export_deploy_cfg(env.unwrapped, str(log_dir))
    print("[TRACE] after export_deploy_cfg", flush=True)

    print("[TRACE] before shutil copy", flush=True)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        params_dir / Path(inspect.getfile(env_cfg.__class__)).name,
    )
    print("[TRACE] after shutil copy", flush=True)

    print("[TRACE] ENTERING runner.learn", flush=True)
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    env.close()

    completion_marker = Path(metrics_dir_value) / "training_complete.yaml"
    completion_marker.write_text(
        f"status: completed\nrun_id: {run_id}\n",
        encoding="utf-8",
    )
    print(
        f"[INFO] Training completion marker: {completion_marker}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
