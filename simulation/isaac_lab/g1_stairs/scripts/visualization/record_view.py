#!/usr/bin/env python3

"""Record one deterministic fixed-camera checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


UNITREE_RL_ROOT = Path(
    os.environ.get(
        "UNITREE_RL_LAB_ROOT",
        "/workspace/unitree_rl_lab",
    )
)

RSL_RL_SCRIPTS = UNITREE_RL_ROOT / "scripts" / "rsl_rl"

if not RSL_RL_SCRIPTS.is_dir():
    raise RuntimeError(
        f"Unitree RSL-RL scripts missing: {RSL_RL_SCRIPTS}"
    )

sys.path.insert(0, str(RSL_RL_SCRIPTS))

import cli_args  # noqa: E402


# Coordinates are relative to env_0 because origin_type="env".
# These are evaluation-camera settings, not physical measurements.
VIEWS = {
    # Wide lateral view. Covers robot start, all three stair edges,
    # and most of the upper platform.
    "side": {
        "eye": (1.40, -3.20, 1.35),
        "lookat": (1.40, 0.00, 0.60),
        "description": (
            "Wide fixed side view covering robot approach, "
            "three stair edges, and upper platform."
        ),
    },

    # View from behind the robot along the direction of travel.
    # This makes robot-to-stair alignment directly visible.
    "front": {
        "eye": (-4.00, 0.00, 1.60),
        "lookat": (1.10, 0.00, 0.60),
        "description": (
            "Fixed longitudinal view from behind the robot, "
            "showing alignment with the staircase."
        ),
    },

    # Oblique overview of the start pose and first-riser approach.
    "approach": {
        "eye": (-1.20, -3.00, 1.70),
        "lookat": (1.20, 0.00, 0.60),
        "description": (
            "Oblique fixed view showing start pose, "
            "approach trajectory, and first stair entry."
        ),
    },

    # Low side camera around the first and second stair edges.
    "contact": {
        "eye": (0.15, -2.00, 0.75),
        "lookat": (0.75, 0.00, 0.30),
        "description": (
            "Low fixed view centered on both feet and "
            "the first two stair contact edges."
        ),
    },
}

# Camera presets are expressed around the nominal robot start.
NOMINAL_ROBOT_START = (0.0, 0.0, 0.8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def git_diff_sha(path: Path) -> str | None:
    try:
        payload = subprocess.check_output(
            ["git", "-C", str(path), "diff", "--binary"],
        )
    except Exception:
        return None

    return hashlib.sha256(payload).hexdigest()


parser = argparse.ArgumentParser(
    description="Phase 11 deterministic G1 checkpoint recorder."
)

parser.add_argument(
    "--task",
    default="Unitree-IEA-G1-29dof-Stairs-v0",
)

parser.add_argument(
    "--view",
    required=True,
    choices=sorted(VIEWS),
)

parser.add_argument(
    "--output_dir",
    required=True,
)

parser.add_argument(
    "--video_length",
    type=int,
    default=400,
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
)

parser.add_argument(
    "--policy_onnx",
    default=None,
)

parser.add_argument(
    "--project_commit",
    required=True,
)

parser.add_argument(
    "--isaaclab_commit",
    required=True,
)

parser.add_argument(
    "--unitree_rl_commit",
    required=True,
)

parser.add_argument(
    "--unitree_diff_sha256",
    required=True,
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

if bool(args.checkpoint) == bool(args.policy_onnx):
    parser.error(
        "Exactly one of --checkpoint or --policy_onnx is required"
    )

if args.num_envs != 1:
    parser.error("Phase 11 requires --num_envs 1")

# Same behavior used by pinned Unitree play.py when --video is selected.
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch


    from rsl_rl.runners import OnPolicyRunner

    import unitree_rl_lab.tasks  # noqa: F401

    # Register thesis task.
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from simulation.isaac_lab.g1_stairs.policy_initialization import (
        initialize_actor_from_unitree_onnx,
    )

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    policy_onnx = None

    if args.checkpoint:
        checkpoint = Path(
            retrieve_file_path(args.checkpoint)
        ).resolve()

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    else:
        policy_onnx = Path(args.policy_onnx).resolve()

        if not policy_onnx.is_file():
            raise FileNotFoundError(policy_onnx)

    view = VIEWS[args.view]

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Deterministic single-course evaluation environment.
    env_cfg.seed = args.seed

    # Do not include command-direction markers in evaluation recordings.
    env_cfg.commands.base_velocity.debug_vis = False

    # Evaluation uses exactly one terrain tile. Training may use a grid,
    # but neighboring courses must never enter the evaluation recordings.
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    # Fixed world-coordinate evaluation camera.
    #
    # The nominal robot begins around world x=0 and faces +X.
    # Stair edges are approximately:
    #   first riser    x=1.00 m
    #   second riser   x=1.25 m
    #   third riser    x=1.50 m
    #   platform start x=1.75 m
    #   platform end   x=4.75 m
    env_cfg.viewer.eye = view["eye"]
    env_cfg.viewer.lookat = view["lookat"]
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = None

    agent_cfg = cli_args.parse_rsl_rl_cfg(
        args.task,
        args,
    )

    agent_cfg.seed = args.seed

    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="rgb_array",
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(raw_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        disable_logger=True,
    )

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )

    initialization_metadata = None

    if checkpoint is not None:
        runner.load(str(checkpoint))
    else:
        initialization_metadata = initialize_actor_from_unitree_onnx(
            runner=runner,
            onnx_path=policy_onnx,
        )

    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    metadata = {
        "schema": "iea_g1_phase11_view_v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "view": args.view,
        "view_description": view["description"],
        "viewer_eye": list(view["eye"]),
        "viewer_lookat": list(view["lookat"]),
        "viewer_origin_type": "world",
        "viewer_env_index": 0,
        "seed": args.seed,
        "num_envs": 1,
        "video_length_steps": args.video_length,
        "checkpoint": (
            str(checkpoint)
            if checkpoint is not None
            else None
        ),
        "checkpoint_sha256": (
            sha256(checkpoint)
            if checkpoint is not None
            else None
        ),
        "policy_initialization": initialization_metadata,
        "project_commit": args.project_commit,
        "isaaclab_commit": args.isaaclab_commit,
        "unitree_rl_commit": args.unitree_rl_commit,
        "unitree_integration_diff_sha256": args.unitree_diff_sha256,
        "step_dt": float(env.unwrapped.step_dt),
    }

    metadata_file = output_dir / "metadata.json"

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=== PHASE 11 VIEW CONTRACT ===")
    print(f"view={args.view}")
    print(f"eye={view['eye']}")
    print(f"lookat={view['lookat']}")
    print("origin_type=world")
    print("env_index=0")
    print(f"seed={args.seed}")
    if checkpoint is not None:
        print(f"checkpoint={checkpoint}")
        print(
            "checkpoint_sha256="
            f"{metadata['checkpoint_sha256']}"
        )
    else:
        print(f"policy_onnx={policy_onnx}")
        print(
            "policy_initialization="
            f"{initialization_metadata}"
        )
    print(
        "unitree_integration_diff_sha256="
        f"{metadata['unitree_integration_diff_sha256']}"
    )
    print(f"step_dt={metadata['step_dt']}")

    obs = env.get_observations()

    # Apply the evaluation camera only after the environment has completed
    # its initialization/reset path. This is the authoritative camera pose.
    robot = env.unwrapped.scene["robot"]

    robot_root = (
        robot.data.root_pos_w[0]
        .detach()
        .cpu()
        .tolist()
    )

    spawn_offset = tuple(
        robot_root[i] - NOMINAL_ROBOT_START[i]
        for i in range(3)
    )

    resolved_eye = tuple(
        view["eye"][i] + spawn_offset[i]
        for i in range(3)
    )

    resolved_lookat = tuple(
        view["lookat"][i] + spawn_offset[i]
        for i in range(3)
    )

    env.unwrapped.sim.set_camera_view(
        eye=resolved_eye,
        target=resolved_lookat,
    )

    metadata["robot_root_world"] = list(robot_root)
    metadata["camera_spawn_offset"] = list(spawn_offset)
    metadata["viewer_eye_resolved"] = list(resolved_eye)
    metadata["viewer_lookat_resolved"] = list(resolved_lookat)

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"robot_root_world={robot_root}")
    print(f"spawn_offset={spawn_offset}")
    print(f"resolved_eye={resolved_eye}")
    print(f"resolved_lookat={resolved_lookat}")

    steps = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        steps += 1

        if steps >= args.video_length:
            break

    env.close()

    metadata["completed"] = True
    metadata["completed_steps"] = steps

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"VALID: recorded {args.view} view "
        f"for {steps} simulation steps"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
