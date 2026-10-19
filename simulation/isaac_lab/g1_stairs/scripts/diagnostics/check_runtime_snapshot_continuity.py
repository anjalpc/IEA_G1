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
parser.add_argument("--verify-portable-startup", action="store_true", help="Legacy two-environment validation")
parser.add_argument("--verify-portable-startup-only", action="store_true", help="Inspect one fresh snapshot-reset Isaac environment without trajectory replay")
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
        load_single_env_snapshot_artifact,
        capture_physics_runtime,
        capture_temporal_runtime,
        restore_physics_runtime,
        restore_temporal_runtime,
        save_single_env_snapshot_artifact,
    )

    config_path = Path(args.resolved_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if (args.verify_portable_startup or args.verify_portable_startup_only) and not args.snapshot_output:
        raise ValueError("--verify-portable-startup requires --snapshot-output")
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

    if args.verify_portable_startup_only:
        # One simulation environment, matching the real training reset subclass.
        # Do not construct or close another Isaac environment in this process.
        artifact_path = Path(args.snapshot_output).resolve()
        artifact = load_single_env_snapshot_artifact(artifact_path)
        env_cfg.near_riser_snapshot_path = str(artifact_path)
        print("[C7] Constructing one portable-startup environment", flush=True)
        startup_env = RslRlVecEnvWrapper(
            gym.make(args.task, cfg=env_cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        try:
            raw = startup_env.unwrapped
            expected_step = int(artifact["episode_length"][0].item())
            observed_step = int(raw.episode_length_buf[0].item())
            if observed_step != expected_step:
                raise AssertionError(f"episode clock mismatch: {observed_step} != {expected_step}")
            source_x = float(artifact["physics"]["root_link_state"][0, 0].item() - artifact["source_env_origin"][0, 0].item())
            observed_x = float(raw.scene["robot"].data.root_link_state_w[0, 0].item() - raw.scene.env_origins[0, 0].item())
            if abs(observed_x - source_x) > args.atol + args.rtol * abs(source_x):
                raise AssertionError(f"root X mismatch: {observed_x} != {source_x}")
            history_errors = {}
            for full_name, saved in artifact["temporal"]["observation_history"].items():
                group, term = full_name.split("/", 1)
                actual = raw.observation_manager._group_obs_term_history_buffer[group][term].buffer
                expected = saved["ordered"].to(device=actual.device)
                history_errors[full_name] = _max_abs(actual, expected)
                torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
            gait = raw.action_manager.get_term("gait_frequency")
            for name, expected in artifact["temporal"]["gait"].items():
                actual = getattr(gait, name)
                torch.testing.assert_close(actual, expected.to(device=actual.device), atol=args.atol, rtol=args.rtol)
            print("[C7] Startup state and history verified; loading actor", flush=True)
            runner = OnPolicyRunner(startup_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=raw.device)
            with torch.inference_mode():
                observations = startup_env.get_observations()
                actor_obs = _policy_observation(observations)
                action = policy(observations)
            if actor_obs.shape != (1, 492) or action.shape != (1, 30):
                raise AssertionError(f"actor shapes: {tuple(actor_obs.shape)}, {tuple(action.shape)}")
            if not torch.isfinite(actor_obs).all() or not torch.isfinite(action).all():
                raise AssertionError("actor observation/action contains nonfinite values")
            print(json.dumps({
                "status": "PASS_SINGLE_ENV_STARTUP_STATE",
                "episode_length": observed_step,
                "local_root_x_m": observed_x,
                "history_max_abs_error": max(history_errors.values(), default=0.0),
                "actor_obs_shape": list(actor_obs.shape),
                "action_shape": list(action.shape),
                "limitations": "No independent original full-observation/action oracle; no repeated or partial reset validation",
            }, indent=2), flush=True)
        finally:
            startup_env.close()
        return

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

        if not args.verify_portable_startup:
            print(json.dumps(result, indent=2), flush=True)
            if args.output_json is not None:
                path = Path(args.output_json)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()

    if not args.verify_portable_startup:
        return

    # A separate environment construction exercises the same gym registration,
    # initial manager reset, and observation append used by PPO startup.
    artifact_path = Path(args.snapshot_output).resolve()
    artifact = load_single_env_snapshot_artifact(artifact_path)
    if int(artifact["episode_length"].item()) != target_steps:
        raise RuntimeError("portable artifact episode clock differs from captured step")

    from copy import deepcopy
    portable_cfg = deepcopy(env_cfg)
    portable_cfg.near_riser_snapshot_path = str(artifact_path)
    portable_env = RslRlVecEnvWrapper(
        gym.make(args.task, cfg=portable_cfg),
        clip_actions=agent_cfg.clip_actions,
    )
    try:
        portable_raw = portable_env.unwrapped
        if int(portable_raw.episode_length_buf[0].item()) != target_steps:
            raise AssertionError("fresh-start snapshot episode clock mismatch")
        portable_x = float(
            portable_raw.scene["robot"].data.root_link_state_w[0, 0].item()
            - portable_raw.scene.env_origins[0, 0].item()
        )
        if abs(portable_x - local_root_x) > args.atol + args.rtol * abs(local_root_x):
            raise AssertionError(f"fresh-start local root X mismatch: {portable_x} != {local_root_x}")
        portable_runner = OnPolicyRunner(
            portable_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        portable_runner.load(str(checkpoint), load_optimizer=False)
        portable_policy = portable_runner.get_inference_policy(device=portable_raw.device)
        with torch.inference_mode():
            portable_obs = _policy_observation(portable_env.get_observations()).detach().clone()
            portable_action = portable_policy(portable_env.get_observations()).detach().clone()
        result["portable_startup"] = {
            "episode_length": int(portable_raw.episode_length_buf[0].item()),
            "local_root_x_m": portable_x,
            "obs_max_abs_error": _max_abs(snapshot_obs, portable_obs),
            "action_max_abs_error": _max_abs(snapshot_action, portable_action),
        }
        torch.testing.assert_close(portable_obs, snapshot_obs, atol=args.atol, rtol=args.rtol)
        torch.testing.assert_close(portable_action, snapshot_action, atol=args.atol, rtol=args.rtol)
        print(json.dumps(result, indent=2), flush=True)
        if args.output_json is not None:
            path = Path(args.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        portable_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
