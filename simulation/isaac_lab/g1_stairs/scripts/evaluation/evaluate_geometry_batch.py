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
                json.dumps(summary["action_sampling"], sort_keys=True),
                summary["wrist_target_scale"],
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
            "upper_platform_contacts": sum(
                str(r.get("upper_platform_contact", "")).lower() == "true"
                for r in rows
            ),
            "supported_platforms": sum(
                str(r.get("supported_platform", "")).lower() == "true"
                for r in rows
            ),
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
parser.add_argument("--sample-actions", action="store_true")
parser.add_argument("--arm-noise-scale", type=float, default=1.0)
parser.add_argument("--wrist-target-scale", type=float, default=1.0)
parser.add_argument("--trace", action="store_true", help="Record one environment and one episode before automatic reset")
parser.add_argument("--heights", type=float, nargs="+", default=[0.0, 0.04])
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes-per-env", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.0 <= args.wrist_target_scale <= 1.0:
    parser.error("--wrist-target-scale must be in [0, 1]")
if args.wrist_target_scale != 1.0 and (args.sample_actions or args.unitree_baseline):
    parser.error("Wrist intervention requires deterministic checkpoint evaluation")
if not 0.0 < args.arm_noise_scale <= 1.0:
    parser.error("--arm-noise-scale must be in (0, 1]")
if args.arm_noise_scale != 1.0 and not args.sample_actions:
    parser.error("--arm-noise-scale requires --sample-actions")
if args.sample_actions and args.unitree_baseline:
    parser.error("sampled evaluation requires an RSL-RL checkpoint")
if args.trace and (args.num_envs != 1 or args.episodes_per_env != 1):
    parser.error("--trace requires --num-envs 1 --episodes-per-env 1")
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
    # Qualification always uses strict first-contact termination, even for D2 checkpoints.
    resolved["episode_contract"]["training_recovery_s"] = 0.0
    if not resolved.get("episode_contract", {}).get("enabled", False):
        raise ValueError("batch evaluation requires an enabled V03 episode_contract")
    model = Path(args.checkpoint or resolved["policy_initialization"]["source"]["path"]).resolve()
    if args.unitree_baseline and resolved["policy_initialization"]["type"] != "unitree_velocity_onnx_transfer":
        raise ValueError("baseline requires the original Unitree ONNX source")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = ["height_m", "seed", "env_id", "episode", "outcome", "success",
              "duration_s", "upper_platform_contact", "supported_platform",
              "entered_stairs", "terminal_phase"]
    all_rows = []
    coverage = None
    sampling = None
    def persist_completed_summary():
        summaries = []
        for height in args.heights:
            rows = [r for r in all_rows if r["height_m"] == height]
            successes = sum(r["success"] for r in rows)
            summaries.append({
                "height_m": height, "task": "flat_traversal" if height == 0 else "stair_traversal",
                "episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
                "upper_platform_contacts": sum(v["upper_platform_contact"] for v in rows),
                "supported_platforms": sum(v["supported_platform"] for v in rows),
                "outcomes": {r.name.lower(): sum(v["outcome"] == r.name.lower() for v in rows)
                             for r in Result if r != Result.RUNNING},
            })
        summary = {
            "schema_version": 1, "episode_contract": resolved["episode_contract"],
            "action_sampling": sampling,
            "wrist_target_scale": args.wrist_target_scale,
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
                from simulation.isaac_lab.g1_stairs.runtime_snapshot import (
                    apply_near_riser_snapshot_reset_config,
                )
                snapshot_reset = apply_near_riser_snapshot_reset_config(
                    env_cfg, resolved, project_root=REPO_ROOT,
                )
                if snapshot_reset["enabled"]:
                    print("[INFO] Evaluation near-riser snapshot:", snapshot_reset, flush=True)
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
                tracer = None
                try:
                    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                    if args.unitree_baseline:
                        initialize_actor_from_unitree_onnx(runner, model)
                    else:
                        runner.load(str(model), load_optimizer=False)
                    policy = runner.get_inference_policy(device=env.unwrapped.device)
                    sampling = {
                        "mode": "sampled" if args.sample_actions else "deterministic",
                        "arm_noise_scale": args.arm_noise_scale,
                    }
                    if args.sample_actions:
                        if resolved["ppo"].get("observation_normalization", False):
                            raise ValueError("Sampling diagnostic requires unnormalized observations")
                        actor = runner.alg.policy
                        term = env.unwrapped.action_manager.get_term("JointPositionAction")
                        names = list(term._joint_names)
                        if names != list(resolved["actions"]["joint_order"]):
                            raise ValueError("Runtime action order differs from recorded joint order")
                        arm_ids = [
                            i for i, name in enumerate(names)
                            if any(token in name for token in ("_shoulder_", "_elbow_", "_wrist_"))
                        ]
                        expected_actions = len(names) + int(
                            getattr(env.unwrapped.cfg, "gait_phase_enabled", False)
                        )
                        if len(arm_ids) != 14 or actor.std.shape != (expected_actions,):
                            raise ValueError("Unexpected arm selection or policy std shape")
                        if not bool(torch.isfinite(actor.std).all() and (actor.std > 0).all()):
                            raise ValueError("Policy std must be finite and positive")
                        original_std = actor.std.detach().clone()
                        expected_std = original_std.clone()
                        expected_std[arm_ids] *= args.arm_noise_scale
                        with torch.no_grad():
                            actor.std.copy_(expected_std)
                        sampling.update({
                            "joint_names": names,
                            "original_std": original_std.cpu().tolist(),
                            "effective_std": actor.std.detach().cpu().tolist(),
                        })

                        # Check native distribution without consuming evaluation RNG.
                        device = torch.device(env.unwrapped.device)
                        devices = [device.index if device.index is not None
                                   else torch.cuda.current_device()] if device.type == "cuda" else []
                        check_obs = env.get_observations()
                        with torch.random.fork_rng(devices=devices), torch.inference_mode():
                            expected_mean = policy(check_obs).clone()
                            actor.act(check_obs)
                            torch.testing.assert_close(actor.action_mean, expected_mean)
                            torch.testing.assert_close(
                                actor.action_std,
                                expected_std.expand_as(actor.action_std),
                            )
                        policy = actor.act
                        print("[PASS] Native sampled mean/std and action ordering verified.", flush=True)
                    if args.wrist_target_scale != 1.0:
                        term = env.unwrapped.action_manager.get_term("JointPositionAction")
                        names = list(term._joint_names)
                        if names != list(resolved["actions"]["joint_order"]):
                            raise ValueError("Runtime action ordering differs from config")
                        wrists = [i for i, name in enumerate(names) if "_wrist_" in name]
                        if len(wrists) != 6:
                            raise ValueError("Expected exactly six wrist actions")
                        if term.cfg.clip is not None or not term.cfg.use_default_offset:
                            raise ValueError("Wrist diagnostic requires unclipped default-offset targets")

                        robot = env.unwrapped.scene["robot"]
                        nominal = robot.data.default_joint_pos[:, term._joint_ids]
                        torch.testing.assert_close(term._offset, nominal)
                        old_scale = torch.as_tensor(
                            term._scale, device=term.raw_actions.device,
                            dtype=term.raw_actions.dtype,
                        ).expand_as(term.raw_actions).clone()
                        new_scale = old_scale.clone()
                        new_scale[:, wrists] *= args.wrist_target_scale
                        others = [i for i in range(len(names)) if i not in wrists]

                        # Exercise the actual controller, then restore its buffers.
                        with torch.inference_mode():
                            raw_before = term.raw_actions.clone()
                            processed_before = term.processed_actions.clone()
                            probe = torch.linspace(
                                -0.8, 0.8, len(names), device=nominal.device,
                                dtype=nominal.dtype,
                            ).expand_as(term.raw_actions)
                            term.process_actions(probe)
                            original_targets = term.processed_actions.clone()
                            term._scale = new_scale
                            term.process_actions(probe)
                            torch.testing.assert_close(
                                term.processed_actions[:, wrists] - nominal[:, wrists],
                                args.wrist_target_scale *
                                (original_targets[:, wrists] - nominal[:, wrists]),
                            )
                            assert torch.equal(
                                term.processed_actions[:, others],
                                original_targets[:, others],
                            ), "Non-wrist targets changed"
                            term._raw_actions.copy_(raw_before)
                            term._processed_actions.copy_(processed_before)
                        print("[PASS] Wrist offsets scaled; non-wrist targets unchanged.", flush=True)

                    obs = env.get_observations()
                    if args.trace:
                        from geometry_trace import GeometryTrace
                        tracer = GeometryTrace(env.unwrapped, output)
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
                                    upper_platform_contact=values["upper_platform_contact"][index],
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
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    sys.stderr.flush()
                    raise
                finally:
                    if tracer is not None:
                        tracer.close()
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
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        print("[INFO] Starting simulation_app.close()", flush=True)
        faulthandler.dump_traceback_later(30, repeat=True)
        try:
            simulation_app.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
        print("[INFO] Finished simulation_app.close()", flush=True)
