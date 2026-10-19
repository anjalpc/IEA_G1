#!/usr/bin/env python3
"""Diagnostic: isolate an env-1 near-riser reset from env-0 state (no PPO)."""
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resolved-config", required=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import torch
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401
    from simulation.isaac_lab.g1_stairs.reward_runtime import load_resolved_config
    from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import apply_resolved_replay_config
    from simulation.isaac_lab.g1_stairs.geometry_episode_runtime import apply_geometry_episode_contract
    from simulation.isaac_lab.g1_stairs.runtime_snapshot import (
        apply_near_riser_snapshot_reset_config, load_single_env_snapshot_artifact,
        _reward_term_object,
    )

    config_path = Path(args.resolved_config).resolve()
    resolved = load_resolved_config(config_path)
    cfg = parse_env_cfg(
        "Unitree-IEA-G1-29dof-Stairs-v0", device=args.device,
        num_envs=2, use_fabric=True, entry_point_key="play_env_cfg_entry_point",
    )
    apply_resolved_replay_config(cfg, config_path)
    cfg.scene.num_envs = 2
    terrain = cfg.scene.terrain.terrain_generator
    terrain.num_rows = 1
    terrain.num_cols = 1
    terrain.curriculum = False
    stairs = terrain.sub_terrains["three_step_ascent"]
    stairs.riser_heights = None
    stairs.riser_height_range = (0.04, 0.04)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.seed = args.seed
    cfg.commands.base_velocity.debug_vis = False
    apply_geometry_episode_contract(cfg, resolved, promote=False)
    snapshot_info = apply_near_riser_snapshot_reset_config(cfg, resolved, project_root=REPO_ROOT)
    if not snapshot_info["enabled"]:
        raise RuntimeError("resolved config did not enable snapshot reset")
    artifact = load_single_env_snapshot_artifact(snapshot_info["artifact_path"])
    print("[C7] Creating 2-env real Isaac snapshot-reset environment", flush=True)
    env = gym.make("Unitree-IEA-G1-29dof-Stairs-v0", cfg=cfg)
    try:
        observations, _ = env.reset(seed=args.seed)
        raw = env.unwrapped
        if raw.num_envs != 2:
            raise RuntimeError(f"Expected 2 environments; got {raw.num_envs}")
        if tuple(raw.action_manager.action.shape) != (2, 30):
            raise RuntimeError("Unexpected action dimensions")
        expected_step = int(artifact["episode_length"][0].item())
        if raw.episode_length_buf.tolist() != [expected_step, expected_step]:
            raise RuntimeError("Initial snapshot episode clock mismatch")
        print("[C7] Both envs started from snapshot", flush=True)

        # Complete one simulation/policy step without PPO, using a zero action.
        env.step(torch.zeros((2, 30), device=raw.device))
        print("[C7] One real Isaac step completed", flush=True)

        robot = raw.scene["robot"]
        gait = raw.action_manager.get_term("gait_frequency")
        acq = getattr(raw, "_g1_v04_acquisition", None)
        motion = getattr(raw, "_g1_coordinated_motion", None)
        geometry = getattr(raw, "_g1_geometry_episode", None)
        pref = _reward_term_object(raw, "preferred_transition_milestones")
        step_contract = _reward_term_object(raw, "step_over_transition_contract")
        sources = {
            "robot": robot.data,
            "gait": gait,
            "acquisition": acq,
            "motion": motion,
            "geometry.tracker": None if geometry is None else geometry.tracker,
            "geometry.strict_gait": None if geometry is None else geometry.strict_gait,
            "preferred": pref,
            "step_contract": step_contract,
            "action_manager": raw.action_manager,
        }
        # Snapshot all batched tensor fields, including ones not present in the artifact.
        def observe_env0():
            state = {"episode_length": raw.episode_length_buf[0].detach().clone()}
            for prefix, obj in sources.items():
                if obj is None:
                    continue
                for name, value in vars(obj).items():
                    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 2:
                        state[f"{prefix}.{name}"] = value[0].detach().clone()
            for group, terms in raw.observation_manager._group_obs_term_history_buffer.items():
                for term, buffer in terms.items():
                    if buffer._buffer is not None:
                        state[f"history.{group}/{term}"] = buffer._buffer[:, 0].detach().clone()
                    state[f"history_pushes.{group}/{term}"] = buffer._num_pushes[0].detach().clone()
            for name, term in raw.command_manager._terms.items():
                for field, value in vars(term).items():
                    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 2:
                        state[f"command.{name}.{field}"] = value[0].detach().clone()
            return state

        def cache_markers():
            return {
                "gait.last_finished_step": gait.last_finished_step,
                "acquisition.last_step": None if acq is None else acq.last_step,
                "motion.last_step": None if motion is None else motion.last_step,
                "geometry.last_step": None if geometry is None else geometry.last_step,
            }

        # Observe the real reset call made from ManagerBasedRLEnv.step(),
        # so the comparison is not contaminated by normal physics advancement.
        results = []
        original_reset_idx = raw._reset_idx
        reset_calls = []
        derived_torque_fields = {"robot.computed_torque", "robot.applied_torque"}

        def audited_reset_idx(env_ids):
            ids = torch.as_tensor(env_ids, device=raw.device, dtype=torch.long)
            reset_calls.append(ids.detach().cpu().tolist())
            if ids.tolist() != [1]:
                raise RuntimeError(f"FAIL_AUTO_RESET: expected only env 1, got {ids.tolist()}")
            before = observe_env0()
            markers_before = cache_markers()
            try:
                original_reset_idx(ids)
            except BaseException:
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                os._exit(70)
            after = observe_env0()
            changed = [key for key in before if not torch.equal(before[key], after[key])]
            changed_physical_or_persistent = [
                key for key in changed if key not in derived_torque_fields
            ]
            result = {
                "attempt": len(results) + 1,
                "reset_ids": ids.detach().cpu().tolist(),
                "untouched_tensor_count": len(before),
                "changed_untouched_tensors": changed,
                "changed_non_torque_tensors": changed_physical_or_persistent,
                "shared_cache_markers_before": markers_before,
                "shared_cache_markers_after": cache_markers(),
                "restored_episode_length": int(raw.episode_length_buf[1].item()),
            }
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
            if changed_physical_or_persistent:
                raise RuntimeError("FAIL_AUTO_RESET: untouched non-torque tensors changed")
            if result["restored_episode_length"] != expected_step:
                raise RuntimeError("FAIL_AUTO_RESET: restored episode clock incorrect")

        raw._reset_idx = audited_reset_idx
        try:
            for attempt in (1, 2):
                # Boundary-only acceleration: Isaac's real step must compute
                # termination/timeout and perform the reset automatically.
                # Do not forge tracker.reason or call _reset_idx manually.
                raw.episode_length_buf[1] = raw.max_episode_length - 1
                before_count = len(reset_calls)
                print(f"[C7] Automatic partial reset #{attempt} through env.step", flush=True)
                _, _, terminated, truncated, _ = env.step(
                    torch.zeros((2, 30), device=raw.device)
                )
                if len(reset_calls) != before_count + 1:
                    raise RuntimeError("FAIL_AUTO_RESET: env.step did not make exactly one reset")
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    raise RuntimeError("FAIL_AUTO_RESET: untouched env 0 terminated")
                if not (bool(terminated[1].item()) or bool(truncated[1].item())):
                    raise RuntimeError("FAIL_AUTO_RESET: env 1 did not terminate")
                print(
                    json.dumps({
                        "attempt": attempt,
                        "env1_terminated": bool(terminated[1].item()),
                        "env1_timeout": bool(truncated[1].item()),
                        "cache_markers_after_step": cache_markers(),
                        "env1_episode_length_after_step": int(raw.episode_length_buf[1].item()),
                    }, indent=2), flush=True,
                )
                # Run a genuine non-timeout step after each reset.  This is a
                # bounded cache-consumption check, not a trajectory oracle.
                resets_before = len(reset_calls)
                _, _, follow_term, follow_timeout, _ = env.step(
                    torch.zeros((2, 30), device=raw.device)
                )
                if len(reset_calls) != resets_before:
                    raise RuntimeError("FAIL_POST_RESET: unexpected automatic reset")
                if bool(follow_term.any().item()) or bool(follow_timeout.any().item()):
                    raise RuntimeError("FAIL_POST_RESET: unexpected terminal outcome")
                current_step = int(raw.common_step_counter)
                follow_markers = cache_markers()
                for key in ("gait.last_finished_step", "acquisition.last_step",
                            "motion.last_step", "geometry.last_step"):
                    if follow_markers[key] != current_step:
                        raise RuntimeError(
                            f"FAIL_POST_RESET: {key}={follow_markers[key]} "
                            f"expected {current_step}"
                        )
                if int(raw.episode_length_buf[1].item()) != expected_step + 1:
                    raise RuntimeError("FAIL_POST_RESET: snapshot clock did not advance")
                print(json.dumps({
                    "attempt": attempt,
                    "post_reset_common_step": current_step,
                    "post_reset_cache_markers": follow_markers,
                    "post_reset_episode_length": int(raw.episode_length_buf[1].item()),
                    "post_reset_nonterminal": True,
                }, indent=2), flush=True)
        finally:
            raw._reset_idx = original_reset_idx
        print("PASS_TWO_ENV_POST_RESET_CACHE_CONSUMPTION", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
