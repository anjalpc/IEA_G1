#!/usr/bin/env python3
"""Verify C7 near-riser snapshot restore preserves actor observation and next action."""

from __future__ import annotations

import argparse
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
parser.add_argument("--riser-height", type=float, required=True)
parser.add_argument("--snapshot-time", type=float, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--atol", type=float, default=1.0e-6)
parser.add_argument("--rtol", type=float, default=1.0e-6)
parser.add_argument("--output-json", type=str, default=None)
parser.add_argument("--snapshot-output", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.checkpoint is None:
    parser.error("--checkpoint is required")
if not 0.0 <= args.riser_height <= 0.20:
    parser.error("--riser-height must be in [0, 0.20]")
if args.snapshot_time <= 0.0:
    parser.error("--snapshot-time must be positive")
if args.atol < 0.0 or args.rtol < 0.0:
    parser.error("--atol and --rtol must be non-negative")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _max_abs(a, b) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from simulation.isaac_lab.g1_stairs.balance_residual import _policy_observation
    from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
        apply_resolved_replay_agent_config,
        apply_resolved_replay_config,
    )
    from simulation.isaac_lab.g1_stairs.geometry_episode_runtime import apply_geometry_episode_contract
    from simulation.isaac_lab.g1_stairs.reward_runtime import load_resolved_config
    from simulation.isaac_lab.g1_stairs.runtime_snapshot import (
        capture_physics_runtime,
        capture_temporal_runtime,
        restore_physics_runtime,
        restore_temporal_runtime,
        save_single_env_snapshot_artifact,
    )

    config_path = Path(args.resolved_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    resolved = load_resolved_config(config_path)
    if not resolved.get("episode_contract", {}).get("enabled", False):
        raise ValueError("continuity diagnostic requires enabled episode_contract")

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    apply_resolved_replay_config(env_cfg, config_path)
    terrain = env_cfg.scene.terrain.terrain_generator
    terrain.num_rows = 1
    terrain.num_cols = 1
    terrain.curriculum = False
    stairs = terrain.sub_terrains["three_step_ascent"]
    stairs.riser_heights = None
    stairs.riser_height_range = (args.riser_height, args.riser_height)
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.seed = args.seed
    env_cfg.commands.base_velocity.debug_vis = False
    apply_geometry_episode_contract(env_cfg, resolved, promote=False)

    args.num_envs = 1
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed
    apply_resolved_replay_agent_config(agent_cfg, resolved)

    env = RslRlVecEnvWrapper(
        gym.make(args.task, cfg=env_cfg),
        clip_actions=agent_cfg.clip_actions,
    )
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        raw = env.unwrapped

        step_dt = float(raw.step_dt)
        target_steps = round(args.snapshot_time / step_dt)
        represented_time = target_steps * step_dt
        if abs(represented_time - args.snapshot_time) > 1.0e-9:
            raise ValueError(
                f"snapshot time {args.snapshot_time} is not representable at step_dt={step_dt}"
            )

        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(target_steps):
                obs, _, _, _ = env.step(policy(obs))

            episode_time = float(raw.episode_length_buf[0].item()) * step_dt
            if abs(episode_time - args.snapshot_time) > 1.0e-9:
                raise RuntimeError(
                    f"episode did not reach requested snapshot continuously: "
                    f"expected={args.snapshot_time}, observed={episode_time}"
                )

            gait = raw.action_manager.get_term("gait_frequency")
            committed = getattr(gait, "approach_committed", None)
            if not isinstance(committed, torch.Tensor):
                raise RuntimeError("gait_frequency.approach_committed is unavailable")
            if bool(committed[0].item()):
                raise RuntimeError("snapshot tick is already approach-committed")
            if bool(gait.active[0].item()):
                raise RuntimeError("snapshot tick already has an active gait reference")
            if bool(gait.transition_motion_miss[0].item()):
                raise RuntimeError("snapshot tick already has a transition motion miss")
            if int(gait.approach_preferred_foot[0].item()) != -1:
                raise RuntimeError("snapshot tick already owns an approach foot")

            snapshot_obs = _policy_observation(obs).detach().clone()
            snapshot_action = policy(obs).detach().clone()
            if snapshot_obs.shape != (1, 492):
                raise RuntimeError(f"unexpected C7 actor observation shape: {tuple(snapshot_obs.shape)}")
            if snapshot_action.shape != (1, 30):
                raise RuntimeError(f"unexpected C7 action shape: {tuple(snapshot_action.shape)}")

            physics = capture_physics_runtime(raw)
            temporal = capture_temporal_runtime(raw)
            episode_length = raw.episode_length_buf.detach().clone()
            local_root_x = float(
                physics.root_link_state[0, 0].item() - raw.scene.env_origins[0, 0].item()
            )
            if args.snapshot_output is not None:
                save_single_env_snapshot_artifact(args.snapshot_output, raw, env_id=0)

            # V03 intentionally rejects non-terminal episode resets. For this
            # continuity diagnostic only, mark the episode as an initial reset
            # before invoking the normal destructive reset path. The captured
            # logical episode clock is restored before observations are read.
            raw.episode_length_buf.zero_()
            env.reset()
            restore_physics_runtime(raw, physics)
            restore_temporal_runtime(raw, temporal)
            raw.episode_length_buf.copy_(episode_length)
            raw.sim.forward()
            if raw.sim.has_rtx_sensors() and raw.cfg.rerender_on_reset:
                raw.sim.render()

            restored_container = env.get_observations()
            restored_obs = _policy_observation(restored_container).detach().clone()
            restored_action = policy(restored_container).detach().clone()

        result = {
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "resolved_config": str(config_path),
            "seed": args.seed,
            "height_m": args.riser_height,
            "snapshot_time_s": args.snapshot_time,
            "snapshot_step": target_steps,
            "local_root_x_m": local_root_x,
            "actor_obs_shape": list(snapshot_obs.shape),
            "action_shape": list(snapshot_action.shape),
            "obs_max_abs_error": _max_abs(snapshot_obs, restored_obs),
            "action_max_abs_error": _max_abs(snapshot_action, restored_action),
            "atol": args.atol,
            "rtol": args.rtol,
        }

        try:
            torch.testing.assert_close(
                restored_obs,
                snapshot_obs,
                atol=args.atol,
                rtol=args.rtol,
            )
            torch.testing.assert_close(
                restored_action,
                snapshot_action,
                atol=args.atol,
                rtol=args.rtol,
            )
        except AssertionError:
            result["status"] = "FAIL"
            print(json.dumps(result, indent=2), flush=True)
            if args.output_json is not None:
                path = Path(args.output_json)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n")
            raise

        print(json.dumps(result, indent=2), flush=True)
        if args.output_json is not None:
            path = Path(args.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
