#!/usr/bin/env python3
"""Train the registered G1 stair task with structured metrics."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import shutil
import sys
import yaml

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
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module
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
    from simulation.isaac_lab.g1_stairs.halfway_reset import (
        configure_halfway_reset,
        parse_halfway_reset_config,
        validate_halfway_reset,
    )

    halfway_cfg = parse_halfway_reset_config(resolved_config)
    if halfway_cfg is not None:
        configure_halfway_reset(env_cfg, halfway_cfg)
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

    if halfway_cfg is not None:
        print("[021] before gym.make", flush=True)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if halfway_cfg is not None:
        print("[021] environment constructed", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    halfway_summary = None
    if halfway_cfg is not None:
        env.reset()
        print("[021] first reset complete", flush=True)
        for _ in range(halfway_cfg.settle_steps):
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(env.unwrapped.physics_dt)
        halfway_summary = validate_halfway_reset(
            env.unwrapped,
            halfway_cfg,
        )
        halfway_summary["ppo_iteration_completed"] = False
        summary_path = Path(metrics_dir_value) / "halfway_reset_smoke.yaml"
        summary_path.write_text(
            yaml.safe_dump(halfway_summary, sort_keys=False),
            encoding="utf-8",
        )
        print("=== 021 HALFWAY RESET SMOKE ===", flush=True)
        for name, value in halfway_summary.items():
            print(f"{name}: {value}", flush=True)
        print("[021] halfway reset validated", flush=True)
        print("PASS", flush=True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = InstrumentedRslRlVecEnvWrapper(env)

    class MetricsOnPolicyRunner(MetricsOnPolicyRunnerMixin, OnPolicyRunner):
        def save(self, path: str, infos=None):
            print(
                f"[POST_UPDATE] checkpoint_save ENTER path={path}",
                flush=True,
            )
            result = super().save(path, infos=infos)
            print(
                f"[POST_UPDATE] checkpoint_save EXIT path={path}",
                flush=True,
            )
            return result

    runner = MetricsOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(log_dir),
        device=agent_cfg.device,
    )

    original_store_code_state = on_policy_runner_module.store_code_state

    def traced_store_code_state(log_dir, repositories):
        print(
            "[POST_UPDATE] store_code_state ENTER "
            f"repositories={repositories}",
            flush=True,
        )
        result = original_store_code_state(log_dir, repositories)
        print("[POST_UPDATE] store_code_state EXIT", flush=True)
        return result

    on_policy_runner_module.store_code_state = traced_store_code_state

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

    if halfway_summary is not None:
        halfway_summary["ppo_iteration_completed"] = True
        summary_path.write_text(
            yaml.safe_dump(halfway_summary, sort_keys=False),
            encoding="utf-8",
        )

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
