#!/usr/bin/env python3

"""Record one deterministic fixed-camera checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import yaml
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
    "--show-colliders",
    action="store_true",
    help="Show PhysX collision geometry in the rendered recording.",
)
parser.add_argument(
    "--showcase",
    action="store_true",
    help="Presentation video, not an evaluation: --num_envs robots (<= 64) "
    "on their own stair courses of the --riser_height_override height, "
    "scattered over a field, starting at random times, a moving camera "
    "(showcase_camera.py) at 1920x1080. No traces or symmetry checks.",
)
parser.add_argument(
    "--showcase_spacing_m",
    type=float,
    default=2.2,
    help="Showcase: distance between neighbouring courses. Each staircase "
    "keeps its training width (1.2 m) with plain floor between them.",
)
parser.add_argument(
    "--showcase_rows",
    type=int,
    default=1,
    help="Showcase: rows of courses (robots per row = num_envs / rows).",
)
parser.add_argument(
    "--showcase_jitter_x",
    type=float,
    default=3.0,
    help="Showcase: each course (with its robot start) moves forward by a "
    "random 0..this many metres, so the field is irregular.",
)
parser.add_argument(
    "--showcase_jitter_y",
    type=float,
    default=None,
    help="Showcase: random sideways offset +-this (default: all the room "
    "left in the tile minus 0.05 m).",
)
parser.add_argument(
    "--showcase_start_spread_s",
    type=float,
    default=4.0,
    help="Showcase: each robot stands for a random 0..this many seconds "
    "before it sets off.",
)
parser.add_argument(
    "--showcase_stop_x",
    type=float,
    default=3.8,
    help="Showcase: a robot this far past its start (on the platform) gets a "
    "zero command and stands, instead of finishing the course and resetting.",
)
parser.add_argument(
    "--markers",
    action="store_true",
    help="Draw footstep targets and foot contact-force bars into the video "
    "(footstep tasks; visual only, ignored with --trace-only).",
)

parser.add_argument(
    "--symmetry",
    choices=("off", "check", "average"),
    default="off",
    help="Footstep tasks: 'check' measures the policy's left/right "
    "equivariance error |pi(M s) - M pi(s)| on the visited states and writes "
    "mirror_check.json; 'average' also drives the robot with the exactly "
    "symmetric policy (clip(pi(s)) + M clip(pi(M s))) / 2 (evaluation only).",
)
parser.add_argument(
    "--teacher-symmetry",
    choices=("off", "average"),
    default="off",
    help="Residual footstep tasks: 'average' replaces the frozen Unitree "
    "teacher by (T(s) + M T(M s)) / 2 for this replay (runs trained with "
    "balance_residual_control.teacher_symmetry: average already use it). "
    "With --symmetry the reported teacher error is then that of the "
    "averaged teacher.",
)
parser.add_argument(
    "--trace",
    action="store_true",
    help="Record the existing read-only geometry/reward trace with the video.",
)

parser.add_argument(
    "--trace-only",
    action="store_true",
    help="Record the geometry/reward trace without rendering or writing video.",
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
    help="Output directory. With --trace-only and --num_envs N > 1 it must "
    "contain {env}: one directory per robot, {env} = 1..N.",
)

parser.add_argument(
    "--resolved_config",
    default=None,
    help="Resolved experiment YAML used for the training run.",
)

parser.add_argument(
    "--riser_height_override",
    type=float,
    default=None,
    help="Optional fixed evaluation riser height in [0.0, 0.20] m.",
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

if args.showcase:
    if not 1 <= args.num_envs <= 64:
        parser.error("--showcase needs 1 <= --num_envs <= 64")
    if args.trace or args.trace_only or args.symmetry != "off":
        parser.error("--showcase records video only (no --trace, --symmetry)")
    if args.riser_height_override is None:
        parser.error("--showcase needs --riser_height_override")
    if args.showcase_rows < 1 or args.num_envs % args.showcase_rows:
        parser.error("--showcase_rows must divide --num_envs")
elif args.num_envs != 1:
    # Trace batch: N robots on N identical courses in one replay, one trace
    # directory each (one Isaac start-up instead of N; the A10G is GPU-bound
    # with parallel single-robot replays).
    if not (args.trace_only and 1 < args.num_envs <= 32 and "{env}" in args.output_dir):
        parser.error("--num_envs > 1 needs --trace-only, at most 32 robots and "
                     "{env} in --output_dir (Phase 11 videos use --num_envs 1)")

if args.trace_only:
    args.trace = True

# Camera rendering is unnecessary for trace-only replay.
args.enable_cameras = not args.trace_only

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

if args.show_colliders:
    import carb

    settings = carb.settings.get_settings()
    settings.set_int(
        "/persistent/physics/visualizationDisplayColliders",
        2,
    )
    print("[VIEW] Physics collider visualization enabled")


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

    batch = args.num_envs > 1 and not args.showcase
    output_dirs = ([Path(args.output_dir.format(env=k + 1)).resolve() for k in range(args.num_envs)]
                   if batch else [Path(args.output_dir).resolve()])
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    output_dir = output_dirs[0]

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
        num_envs=args.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Deterministic single-course evaluation environment.
    env_cfg.seed = args.seed

    # Do not include command-direction markers in evaluation recordings.
    # Showcase: Isaac Lab's velocity arrows above each robot (green command,
    # blue actual velocity).
    env_cfg.commands.base_velocity.debug_vis = bool(args.showcase)

    # Evaluation: one course tile per robot (one robot unless a trace batch
    # or showcase), placed explicitly after gym.make.
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_generator.num_rows = args.showcase_rows if args.showcase else 1
    # Showcase: one course per robot (placed explicitly after gym.make).
    terrain_generator.num_cols = (args.num_envs // args.showcase_rows if args.showcase
                                  else args.num_envs)
    if args.showcase:
        stair_width = float(terrain_generator.size[1])
        spacing = max(float(args.showcase_spacing_m), stair_width)
        jitter_x = max(float(args.showcase_jitter_x), 0.0)
        jitter_y = (max(0.5 * (spacing - stair_width) - 0.05, 0.0)
                    if args.showcase_jitter_y is None else float(args.showcase_jitter_y))
        terrain_generator.size = (terrain_generator.size[0] + jitter_x, spacing)
        terrain_generator.seed = args.seed
        for sub in terrain_generator.sub_terrains.values():
            sub.size = terrain_generator.size
            if hasattr(sub, "stair_width"):
                sub.stair_width = stair_width
                sub.showcase_jitter = (jitter_x, jitter_y)
                sub.showcase_seed = args.seed

    # Reproduce the exact terrain and command used by the training run.
    resolved_config = None
    replay = None
    applied_riser_height_m = None

    if args.resolved_config:
        resolved_config_path = Path(
            args.resolved_config
        ).resolve()

        if not resolved_config_path.is_file():
            raise FileNotFoundError(
                resolved_config_path
            )

        resolved_config = yaml.safe_load(
            resolved_config_path.read_text(
                encoding="utf-8"
            )
        )

        training_heights = (
            resolved_config.get("terrain", {}).get("riser_heights_m")
        )
        if (
            args.riser_height_override is None
            and isinstance(training_heights, (list, tuple))
            and len(training_heights) > 1
        ):
            raise ValueError(
                "recording a multi-height run requires "
                "--riser_height_override"
            )

        from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
            apply_resolved_replay_config,
            apply_resolved_replay_agent_config,
        )

        replay = apply_resolved_replay_config(
            env_cfg,
            resolved_config_path,
        )

        applied_riser_height_m = float(replay.riser_height_m)

        if args.riser_height_override is not None:
            height = float(args.riser_height_override)

            if not 0.0 <= height <= 0.20:
                raise ValueError(
                    "--riser_height_override must be in [0.0, 0.20]"
                )

            # Match evaluate_geometry_batch.py fixed-height semantics:
            # disable discrete/curriculum height selection and force one height.
            stairs = terrain_generator.sub_terrains["three_step_ascent"]
            stairs.riser_heights = None
            stairs.riser_height_range = (height, height)
            terrain_generator.curriculum = False
            env_cfg.scene.terrain.max_init_terrain_level = 0

            applied_riser_height_m = height

            print(
                "[INFO] Recording fixed riser override:",
                height,
                flush=True,
            )

        print(
            "[INFO] Recording replay config:",
            f"riser={applied_riser_height_m}",
            f"tread={replay.tread_depth_m}",
            f"steps={replay.step_count}",
            f"command={replay.forward_velocity_mps}",
            flush=True,
        )

    if resolved_config is not None:
        from simulation.isaac_lab.g1_stairs.runtime_contract import (
            reject_archived_pre_specialist_runtime,
        )

        reject_archived_pre_specialist_runtime(resolved_config)

        # Reconstruct reference-conditioned observations and residual
        # action semantics used by experiments 029+.
        from simulation.isaac_lab.g1_stairs.reference_tracking_runtime import (
            apply_reference_tracking_config,
        )

        reference_tracking_cfg = apply_reference_tracking_config(
            env_cfg,
            resolved_config,
        )

        if reference_tracking_cfg:
            print(
                "[INFO] Recording reference tracking:",
                reference_tracking_cfg,
                flush=True,
            )

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
    if args.showcase:
        env_cfg.viewer.resolution = (1920, 1080)

    agent_cfg = cli_args.parse_rsl_rl_cfg(
        args.task,
        args,
    )

    agent_cfg.seed = args.seed

    if resolved_config is not None:
        apply_resolved_replay_agent_config(
            agent_cfg,
            resolved_config,
        )

    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None if args.trace_only else "rgb_array",
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args.showcase or batch:
        # One robot per course: env i -> row i % rows, column i // rows
        # (Isaac Lab draws rows at random, so robots could share a course).
        rows = args.showcase_rows if args.showcase else 1
        terrain = env.unwrapped.scene.terrain
        ids = torch.arange(args.num_envs, device=terrain.env_origins.device)
        terrain.terrain_levels[:] = ids % rows
        terrain.terrain_types[:] = ids // rows
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

    if not args.trace_only:
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

    if resolved_config is not None:
        from simulation.isaac_lab.g1_stairs.balance_residual import (
            wrap_balance_residual_from_config,
        )
        env = wrap_balance_residual_from_config(env, resolved_config)

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
        "num_envs": args.num_envs,
        "showcase": bool(args.showcase),
        "video_length_steps": args.video_length,
        "riser_height_m": applied_riser_height_m,
        "tread_depth_m": (
            float(replay.tread_depth_m)
            if replay is not None
            else None
        ),
        "step_count": (
            int(replay.step_count)
            if replay is not None
            else None
        ),
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
        "trace": bool(args.trace),
        "trace_only": bool(args.trace_only),
    }

    def write_metadata():
        for k, directory in enumerate(output_dirs):
            entry = dict(metadata, env_index=k, robot=k + 1) if batch else metadata
            (directory / "metadata.json").write_text(
                json.dumps(entry, indent=2, sort_keys=True) + "\n")

    write_metadata()

    print("=== PHASE 11 VIEW CONTRACT ===")
    print(f"view={args.view}")
    print(f"eye={view['eye']}")
    print(f"lookat={view['lookat']}")
    print("origin_type=world")
    print("env_index=0")
    print(f"seed={args.seed}")
    print(f"riser_height_m={metadata['riser_height_m']}")
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

    trace = None

    if args.trace:
        from simulation.isaac_lab.g1_stairs.scripts.evaluation.geometry_trace import (
            GeometryTrace,
        )

        trace = GeometryTrace(
            env.unwrapped,
            output_dirs,
        )

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

    if not args.trace_only:
        env.unwrapped.sim.set_camera_view(
            eye=resolved_eye,
            target=resolved_lookat,
        )

    metadata["robot_root_world"] = list(robot_root)
    metadata["camera_spawn_offset"] = list(spawn_offset)
    metadata["viewer_eye_resolved"] = list(resolved_eye)
    metadata["viewer_lookat_resolved"] = list(resolved_lookat)

    write_metadata()

    print(f"robot_root_world={robot_root}")
    print(f"spawn_offset={spawn_offset}")
    print(f"resolved_eye={resolved_eye}")
    print(f"resolved_lookat={resolved_lookat}")

    showcase = None
    if args.showcase:
        from simulation.isaac_lab.g1_stairs.showcase_camera import (
            camera_pose, field_geometry, start_delay_steps,
        )
        from simulation.isaac_lab.g1_stairs.footstep_isaac_adapter import course_root_pose
        origins = env.unwrapped.scene.env_origins
        centre_y, half_width, spread = field_geometry(origins[:, :2].tolist())
        showcase = {"centre_y": centre_y, "half_width": half_width, "spread": spread,
                    "ground_z": float(origins[:, 2].mean()),
                    "mean_x": float(robot.data.root_pos_w[:, 0].mean()),
                    # close-up robot: the one nearest the camera side
                    "hero": int(torch.argmin(origins[:, 1]))}
        metadata["showcase_field"] = {"centre_y": centre_y, "half_width": half_width,
                                      "spread": spread, "hero_env": showcase["hero"],
                                      "origins": origins[:, :2].tolist()}

        # Random start: a held robot keeps a zero command (it stands, the
        # gait clock stays idle; the stall timer only runs on the stair
        # plan). A robot past --showcase_stop_x (on the platform) also gets
        # a zero command and stands. The stair steering rewrites the
        # command every step, so the hold is re-applied right after it.
        from simulation.isaac_lab.g1_stairs import footstep_live_progress as flp
        hold = torch.tensor(start_delay_steps(args.num_envs, args.showcase_start_spread_s,
                                              float(env.unwrapped.step_dt), args.seed),
                            device=env.unwrapped.device)
        showcase["hold"] = hold
        metadata["showcase_hold_steps"] = hold.tolist()
        steering = flp.steer_stair_command

        def apply_hold(raw_env):
            arrived = course_root_pose(raw_env)[0][:, 0] > args.showcase_stop_x
            held = (showcase["hold"] > steps) | arrived
            if bool(held.any()):
                raw_env.command_manager.get_command("base_velocity")[held] = 0.0

        def steer_then_hold(raw_env, *a, **kw):
            active = steering(raw_env, *a, **kw)
            apply_hold(raw_env)
            return active

        flp.steer_stair_command = steer_then_hold

    def move_showcase_camera():
        # Follow the robots' mean x (smoothed, so resets do not jerk the view).
        x = float(robot.data.root_pos_w[:, 0].mean())
        showcase["mean_x"] += 0.05 * (x - showcase["mean_x"])
        hero = robot.data.root_pos_w[showcase["hero"]]
        eye, lookat = camera_pose(
            steps / max(args.video_length - 1, 1),
            (showcase["mean_x"], showcase["centre_y"], showcase["half_width"], showcase["ground_z"]),
            (float(hero[0]), float(hero[1])), showcase["spread"])
        env.unwrapped.sim.set_camera_view(eye=eye, target=lookat)

    steps = 0
    if showcase is not None:
        apply_hold(env.unwrapped)
        obs = env.get_observations()      # first step already sees the hold
        move_showcase_camera()

    markers = None
    if args.markers and not args.trace_only:
        from simulation.isaac_lab.g1_stairs.footstep_markers import FootstepMarkers
        markers = FootstepMarkers(env.unwrapped)

    if args.teacher_symmetry == "average":
        from simulation.isaac_lab.g1_stairs.balance_residual import MirrorAveragedTeacher
        if getattr(env, "_teacher", None) is None or not getattr(env, "_teacher_prefix", False):
            raise SystemExit("--teacher-symmetry needs the frozen Unitree ONNX residual wrapper")
        if env.teacher_metadata.get("teacher_symmetry") != "average":
            env._teacher = MirrorAveragedTeacher(env._teacher)
        print("TEACHER SYMMETRY: average", flush=True)

    mirror = None
    if args.symmetry != "off":
        from simulation.isaac_lab.g1_stairs import footstep_symmetry as fsym
        from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS
        mirror = {"err_sum": None, "clip_err_sum": None, "sat_sum": None, "sq_sum": None, "n": 0}
        # The residual wrapper clips the policy output before execution, so
        # equivariance and averaging are judged on the clipped residual.
        residual_clip = float(getattr(env, "_residual_clip", float("inf")))

        def acc(key, value):
            mirror[key] = value if mirror.get(key) is None else mirror[key] + value

        # The executed action is teacher + clip(residual) * scale, so the
        # frozen Unitree teacher is checked the same way: teacher(M s)
        # against M teacher(s), and the executed action as a whole.
        from simulation.isaac_lab.g1_stairs.balance_residual import teacher_observation
        teacher_net = getattr(env, "_teacher", None)
        residual_scale = getattr(env, "_scale", None)

        def teacher_of(o):
            return teacher_net(teacher_observation(
                o["policy"], int(env.teacher_metadata["observation_dim"]),
                env._teacher_prefix))

        def mirrored_obs(o):
            m = o.clone()
            m["policy"] = fsym.mirror_policy_obs(o["policy"])
            return m

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            if mirror is not None:
                actions_m = fsym.mirror_actions(policy(mirrored_obs(obs)))
                clipped = actions.clamp(-residual_clip, residual_clip)
                clipped_m = actions_m.clamp(-residual_clip, residual_clip)
                # Per robot: [num_envs, joints].
                err = actions_m - actions
                acc("err_sum", err.abs())
                acc("clip_err_sum", (clipped_m - clipped).abs())
                acc("sat_sum", (actions.abs() > residual_clip).float())
                acc("raw_sum", actions)
                acc("raw_abs_sum", actions.abs())
                acc("signed_sum", err)
                if teacher_net is not None and residual_scale is not None:
                    teacher = teacher_of(obs)
                    teacher_m = fsym.mirror_actions(teacher_of(mirrored_obs(obs)))
                    t_err = teacher_m - teacher
                    acc("teacher_abs_sum", t_err.abs())
                    acc("teacher_signed_sum", t_err)
                    executed = teacher + clipped * residual_scale
                    executed_m = teacher_m + clipped_m * residual_scale
                    e_err = executed_m - executed
                    acc("executed_abs_sum", e_err.abs())
                    acc("executed_signed_sum", e_err)
                acc("sq_sum", (err * err).mean(dim=-1))
                mirror["n"] += 1
                if args.symmetry == "average":
                    actions = 0.5 * (clipped + clipped_m)
            obs, _, _, _ = env.step(actions)
            if markers is not None:
                markers.update(env.unwrapped)

        steps += 1
        if showcase is not None:
            move_showcase_camera()

        if steps >= args.video_length:
            break

    if trace is not None:
        trace.close()

    env.close()

    metadata["completed"] = True
    metadata["completed_steps"] = steps
    if mirror is not None and mirror["n"]:
        n = mirror["n"]
        per_joint = lambda key, k: dict(zip(BODY_JOINTS, (mirror[key][k] / n).tolist()))
        for k, directory in enumerate(output_dirs):
            report = {
                "mode": args.symmetry,
                "steps": n,
                "rms_policy_action": float(mirror["sq_sum"][k] / n) ** 0.5,
                "mean_abs_by_joint": per_joint("err_sum", k),
                "residual_clip": residual_clip,
                "clipped_mean_abs_by_joint": per_joint("clip_err_sum", k),
                "saturated_fraction_by_joint": per_joint("sat_sum", k),
                # signed: mean of M pi(M s) - pi(s); a steady bias shows a
                # one-sided preference, not just noise.
                "signed_mean_by_joint": per_joint("signed_sum", k),
            }
            for key in ("teacher_abs", "teacher_signed", "executed_abs", "executed_signed", "raw", "raw_abs"):
                if mirror.get(key + "_sum") is not None:
                    report[key + "_by_joint"] = per_joint(key + "_sum", k)
            (directory / "mirror_check.json").write_text(json.dumps(report, indent=2) + "\n")
        if "teacher_abs_by_joint" in report:
            t_top = sorted(report["teacher_abs_by_joint"].items(), key=lambda kv: -kv[1])[:4]
            print("MIRROR teacher: " + ", ".join(f"{k} {v:.3f}" for k, v in t_top))
        top = sorted(report["mean_abs_by_joint"].items(), key=lambda kv: -kv[1])[:6]
        print(f"MIRROR {args.symmetry}: rms {report['rms_policy_action']:.3f}; "
              + ", ".join(f"{k} {v:.2f}" for k, v in top))
    metadata["symmetry"] = args.symmetry
    metadata["teacher_symmetry"] = args.teacher_symmetry

    write_metadata()

    print(
        f"VALID: recorded {args.view} view "
        f"for {steps} simulation steps"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
