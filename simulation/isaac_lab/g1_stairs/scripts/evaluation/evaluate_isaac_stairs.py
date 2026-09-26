#!/usr/bin/env python3
"""Evaluate one G1 stair policy with reward-independent outcomes."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNITREE_RL_ROOT = Path(
    os.environ.get("UNITREE_RL_LAB_ROOT", "/workspace/unitree_rl_lab")
)
from simulation.isaac_lab.g1_stairs.unitree_bootstrap import (  # noqa: E402
    bootstrap_unitree_rl_lab,
)

bootstrap_unitree_rl_lab(UNITREE_RL_ROOT)

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
parser.add_argument("--policy_onnx", default=None)
parser.add_argument(
    "--resolved-config",
    default=None,
    help="Replay terrain and command settings from a recorded experiment.",
)
parser.add_argument(
    "--validate-step-up-detector",
    action="store_true",
    help="Run physical single-step detector validation during evaluation.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if bool(args.checkpoint) == bool(args.policy_onnx):
    parser.error(
        "Exactly one of --checkpoint or --policy_onnx is required"
    )
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
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from simulation.isaac_lab.g1_stairs.evaluation.isaac_lab_adapter import (
        capture_isaac_lab_state,
        to_step_sample,
    )
    from simulation.isaac_lab.g1_stairs.evaluation.metrics import RunMetricsWriter
    from simulation.isaac_lab.g1_stairs.evaluation.stair_outcomes import StairOutcomeEvaluator
    from simulation.isaac_lab.g1_stairs.policy_initialization import (
        initialize_actor_from_unitree_onnx,
    )
    from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
        StepUpCheckpointValidator,
        apply_resolved_replay_agent_config,
        apply_resolved_replay_config,
        stair_outcome_config_from_replay,
    )

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

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    env_cfg.commands.base_velocity.debug_vis = False

    replay_config = None
    reference_tracking_cfg = None
    resolved_config = None

    if args.resolved_config is not None:
        resolved_config_path = Path(args.resolved_config).resolve()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(resolved_config_path)

        replay_config = apply_resolved_replay_config(
            env_cfg,
            resolved_config_path,
        )

        # Reconstruct reference-conditioned observations and residual
        # action semantics used by experiments 029+.
        from simulation.isaac_lab.g1_stairs.reward_runtime import (
            load_resolved_config,
        )
        from simulation.isaac_lab.g1_stairs.reference_tracking_runtime import (
            apply_reference_tracking_config,
        )

        resolved_config = load_resolved_config(
            resolved_config_path
        )

        reference_tracking_cfg = apply_reference_tracking_config(
            env_cfg,
            resolved_config,
        )

        if reference_tracking_cfg:
            print(
                "[INFO] Evaluation reference tracking:",
                reference_tracking_cfg,
                flush=True,
            )

        print(
            "[INFO] Evaluation replay config: "
            f"riser={replay_config.riser_height_m} "
            f"tread={replay_config.tread_depth_m} "
            f"steps={replay_config.step_count} "
            f"command={replay_config.forward_velocity_mps}",
            flush=True,
        )

    if args.validate_step_up_detector and replay_config is None:
        raise RuntimeError(
            "--validate-step-up-detector requires --resolved-config"
        )

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    disabled_terminations = disable_native_terminations(env_cfg)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed

    if resolved_config is not None:
        apply_resolved_replay_agent_config(
            agent_cfg,
            resolved_config,
        )

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

    initialization_metadata = None

    if checkpoint is not None:
        runner.load(str(checkpoint))
    else:
        initialization_metadata = initialize_actor_from_unitree_onnx(
            runner=runner,
            onnx_path=policy_onnx,
        )

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    if replay_config is None:
        evaluator = StairOutcomeEvaluator()
    else:
        evaluator = StairOutcomeEvaluator(
            config=stair_outcome_config_from_replay(replay_config)
        )
    writer = RunMetricsWriter(output_dir, output_dir.name)

    robot = env.unwrapped.scene["robot"]
    joint_names = tuple(robot.joint_names)
    body_names = tuple(robot.body_names)

    contact_sensor = env.unwrapped.scene.sensors[
        "contact_forces"
    ]
    contact_body_names = tuple(
        contact_sensor.body_names
    )

    from simulation.isaac_lab.g1_stairs.contracts import (
        BODY_JOINTS,
    )

    right_ankle_joint_name = (
        "right_ankle_pitch_joint"
    )

    try:
        right_ankle_joint_index = joint_names.index(
            right_ankle_joint_name
        )
        right_ankle_action_index = tuple(
            BODY_JOINTS
        ).index(
            right_ankle_joint_name
        )
    except ValueError as exc:
        raise RuntimeError(
            "right ankle pitch joint is missing"
        ) from exc

    reference_motion = None
    reference_start_frame = 0
    reference_end_frame = 0
    residual_scale_rad = None
    residual_raw_clip = None

    if (
        reference_tracking_cfg
        and reference_tracking_cfg.get(
            "enabled",
            False,
        )
    ):
        from simulation.isaac_lab.g1_stairs.reference_motion import (
            load_reference_motion,
        )

        metadata_path = Path(
            reference_tracking_cfg["metadata"]
        )

        if not metadata_path.is_absolute():
            metadata_path = (
                REPO_ROOT
                / metadata_path
            )

        reference_motion = load_reference_motion(
            metadata_path
        )

        reference_start_frame = int(
            reference_tracking_cfg[
                "start_frame"
            ]
        )

        reference_end_frame = int(
            reference_tracking_cfg[
                "end_frame"
            ]
        )

        if reference_tracking_cfg.get(
            "residual_control_enabled",
            False,
        ):
            residual_scale_rad = float(
                reference_tracking_cfg[
                    "residual_scale_rad"
                ]
            )

            residual_raw_clip = float(
                reference_tracking_cfg[
                    "raw_action_clip"
                ]
            )

    try:
        waist_pitch_index = joint_names.index("waist_pitch_joint")
    except ValueError as exc:
        raise RuntimeError(
            "waist_pitch_joint is missing from robot joint_names"
        ) from exc

    print(
        "[DIAG] waist_pitch default =",
        float(
            robot.data.default_joint_pos[
                0,
                waist_pitch_index,
            ].item()
        ),
    )

    print(
        "[DIAG] waist_pitch actual at reset =",
        float(
            robot.data.joint_pos[
                0,
                waist_pitch_index,
            ].item()
        ),
    )

    required_feet = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )

    try:
        foot_indices = {
            name: body_names.index(name)
            for name in required_feet
        }

        foot_contact_indices = {
            name: contact_body_names.index(name)
            for name in required_feet
        }

    except ValueError as exc:
        raise RuntimeError(
            "required ankle body is missing from robot "
            "or contact sensor body names"
        ) from exc

    foot_collision_spheres = (
        (-0.05, 0.025, -0.03),
        (-0.05, -0.025, -0.03),
        (0.12, 0.03, -0.03),
        (0.12, -0.03, -0.03),
    )
    foot_collision_radius_m = 0.005

    def rotated_x_wxyz(quaternion, vector):
        w, x, y, z = quaternion
        vx, vy, vz = vector

        return (
            (1.0 - 2.0 * (y * y + z * z)) * vx
            + 2.0 * (x * y - w * z) * vy
            + 2.0 * (x * z + w * y) * vz
        )

    def posture_sample(
        sample,
        *,
        policy_actions=None,
        reference_frame=None,
    ):
        env_origin = env.unwrapped.scene.env_origins[0]

        feet = {}

        for name, body_index in foot_indices.items():
            position = robot.data.body_pos_w[
                0,
                body_index,
            ]

            quaternion = tuple(
                float(value)
                for value in robot.data.body_quat_w[
                    0,
                    body_index,
                ].tolist()
            )

            collision_centres_x = []

            for sphere_offset in foot_collision_spheres:
                world_x = (
                    float(position[0].item())
                    + rotated_x_wxyz(
                        quaternion,
                        sphere_offset,
                    )
                    - float(env_origin[0].item())
                    + evaluator.config.start_x_m
                )

                collision_centres_x.append(world_x)

            contact_force = (
                contact_sensor.data.net_forces_w[
                    0,
                    foot_contact_indices[name],
                ]
            )

            feet[name] = {
                "course_x_m": float(
                    position[0].item()
                    - env_origin[0].item()
                    + evaluator.config.start_x_m
                ),
                "height_m": float(
                    position[2].item()
                    - env_origin[2].item()
                ),
                "contact_force_n": float(
                    torch.linalg.vector_norm(
                        contact_force
                    ).item()
                ),
                "contact_force_z_n": float(
                    contact_force[2].item()
                ),
                "collision_sphere_centres_x_m": (
                    collision_centres_x
                ),
                "collision_forward_x_m": (
                    max(collision_centres_x)
                    + foot_collision_radius_m
                ),
            }

        diagnostic = {
            "time_s": float(sample.time_s),
            "course_x_m": float(sample.base_x_m),
            "root_height_m": float(sample.base_height_m),
            "root_roll_rad": float(sample.roll_rad),
            "root_pitch_rad": float(sample.pitch_rad),
            "root_angular_velocity_y_radps": float(
                robot.data.root_ang_vel_w[
                    0,
                    1,
                ].item()
            ),
            "waist_pitch_rad": float(
                robot.data.joint_pos[
                    0,
                    waist_pitch_index,
                ].item()
            ),
            "contacts": sorted(sample.contacts),
            "feet": feet,
        }

        right_ankle = {
            "actual_position_rad": float(
                robot.data.joint_pos[
                    0,
                    right_ankle_joint_index,
                ].item()
            ),
            "actual_velocity_radps": float(
                robot.data.joint_vel[
                    0,
                    right_ankle_joint_index,
                ].item()
            ),
        }

        if (
            policy_actions is not None
            and reference_motion is not None
            and reference_frame is not None
        ):
            action_value = float(
                policy_actions[
                    0,
                    right_ankle_action_index,
                ].item()
            )

            frame = max(
                reference_start_frame,
                min(
                    reference_end_frame,
                    int(reference_frame),
                ),
            )

            q_ref = float(
                reference_motion.dof_pos[
                    frame,
                    right_ankle_action_index,
                ]
            )

            right_ankle[
                "reference_frame"
            ] = frame

            right_ankle[
                "reference_position_rad"
            ] = q_ref

            right_ankle[
                "policy_action_raw"
            ] = action_value

            if (
                residual_scale_rad is not None
                and residual_raw_clip is not None
            ):
                bounded_action = max(
                    -residual_raw_clip,
                    min(
                        residual_raw_clip,
                        action_value,
                    ),
                )

                residual_rad = (
                    bounded_action
                    * residual_scale_rad
                )

                target_unclamped = (
                    q_ref
                    + residual_rad
                )

                limits = (
                    robot.data.soft_joint_pos_limits[
                        0,
                        right_ankle_joint_index,
                    ]
                )

                target_clamped = max(
                    float(limits[0].item()),
                    min(
                        float(limits[1].item()),
                        target_unclamped,
                    ),
                )

                right_ankle[
                    "policy_action_bounded"
                ] = bounded_action

                right_ankle[
                    "residual_rad"
                ] = residual_rad

                right_ankle[
                    "target_unclamped_rad"
                ] = target_unclamped

                right_ankle[
                    "target_command_rad"
                ] = target_clamped

                right_ankle[
                    "target_error_rad"
                ] = (
                    target_clamped
                    - right_ankle[
                        "actual_position_rad"
                    ]
                )

        diagnostic[
            "right_ankle_pitch"
        ] = right_ankle

        return diagnostic

    initial_state = capture_isaac_lab_state(
        env.unwrapped,
        time_s=0.0,
    )
    initial_sample = to_step_sample(
        initial_state,
        config=evaluator.config,
    )

    initial_diagnostic = posture_sample(initial_sample)

    posture = {
        "start": initial_diagnostic,
        "mid_approach": None,
        "pre_riser": None,
        "terminal": None,
    }

    pre_terminal_trace = deque(
        [initial_diagnostic],
        maxlen=31,
    )

    mid_approach_target_x_m = 0.5 * (
        evaluator.config.start_x_m
        + evaluator.config.first_riser_x_m
    )
    mid_approach_error = float("inf")
    pre_riser_max_x = float("-inf")

    step_up_validator = None

    if args.validate_step_up_detector:
        step_up_validator = StepUpCheckpointValidator(
            env.unwrapped,
            replay=replay_config,
            dt=float(env.unwrapped.step_dt),
        )

        # Capture the initialized pose as part of force/base calibration.
        step_up_validator.update()

    result = evaluator.update(initial_sample)
    steps = 0

    while result is None and steps < args.max_steps and simulation_app.is_running():
        reference_frame_used = (
            reference_start_frame
            + steps
        )

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
        sample = to_step_sample(
            state,
            config=evaluator.config,
        )
        diagnostic_sample = posture_sample(
            sample,
            policy_actions=actions,
            reference_frame=reference_frame_used,
        )

        if step_up_validator is not None:
            step_up_validator.update()

        pre_terminal_trace.append(
            diagnostic_sample
        )

        distance_to_mid = abs(
            sample.base_x_m - mid_approach_target_x_m
        )

        if distance_to_mid < mid_approach_error:
            mid_approach_error = distance_to_mid
            posture["mid_approach"] = diagnostic_sample

        if (
            sample.base_x_m < evaluator.config.first_riser_x_m
            and sample.base_x_m > pre_riser_max_x
        ):
            pre_riser_max_x = sample.base_x_m
            posture["pre_riser"] = diagnostic_sample

        result = evaluator.update(sample)

        if result is not None:
            posture["terminal"] = diagnostic_sample

    if result is None:
        raise RuntimeError("simulation stopped before an outcome was produced")

    policy_source = (
        str(checkpoint)
        if checkpoint is not None
        else f"onnx:{policy_onnx}"
    )

    writer.record_episode(
        result,
        checkpoint=policy_source,
        evaluation_seed=args.seed,
        episode_index=0,
    )
    summary = writer.write_summary()

    checkpoint_sha256 = (
        sha256(checkpoint)
        if checkpoint is not None
        else None
    )

    policy_onnx_sha256 = (
        sha256(policy_onnx)
        if policy_onnx is not None
        else None
    )

    metadata = {
        "schema": "iea_g1_stair_evaluation_v1",
        "task": args.task,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "completed_steps": steps,
        "step_dt": float(env.unwrapped.step_dt),
        "checkpoint": (
            str(checkpoint)
            if checkpoint is not None
            else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_onnx": (
            str(policy_onnx)
            if policy_onnx is not None
            else None
        ),
        "policy_onnx_sha256": policy_onnx_sha256,
        "policy_initialization": initialization_metadata,
        "disabled_native_terminations": list(disabled_terminations),
        "outcome": result.outcome.value,
        "success": result.success,
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    posture_diagnostics = {
        "schema": "iea_g1_stair_posture_diagnostics_v1",
        "coordinate_system": "stair_course",
        "definitions": {
            "start": "first valid sample after environment initialization",
            "mid_approach": (
                "sample closest to the midpoint between evaluator start "
                "and first-riser X"
            ),
            "pre_riser": (
                "maximum course X strictly before the configured first "
                "riser"
            ),
            "terminal": (
                "sample that produced the reward-independent evaluator outcome"
            ),
        },
        "start_x_m": evaluator.config.start_x_m,
        "first_riser_x_m": evaluator.config.first_riser_x_m,
        "mid_approach_target_x_m": mid_approach_target_x_m,
        "foot_collision_model": {
            "source": (
                "g1_29dof_rev_1_0.urdf "
                "ankle_roll_link collision spheres"
            ),
            "sphere_radius_m": foot_collision_radius_m,
            "local_centres_m": [
                list(value)
                for value in foot_collision_spheres
            ],
        },
        "samples": posture,
        "pre_terminal_trace": list(
            pre_terminal_trace
        ),
    }

    (
        output_dir / "posture_diagnostics.json"
    ).write_text(
        json.dumps(
            posture_diagnostics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    step_up_summary = None

    if step_up_validator is not None:
        step_up_summary = step_up_validator.summary()

        (
            output_dir / "step_up_detector_validation.json"
        ).write_text(
            json.dumps(
                step_up_summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    env.close()

    print("=== INDEPENDENT STAIR EVALUATION ===")

    if checkpoint is not None:
        print(f"checkpoint={checkpoint}")
        print(
            "checkpoint_sha256="
            f"{metadata['checkpoint_sha256']}"
        )
    else:
        print(f"policy_onnx={policy_onnx}")
        print(
            "policy_onnx_sha256="
            f"{metadata['policy_onnx_sha256']}"
        )
        print(
            "policy_initialization="
            f"{initialization_metadata}"
        )
    print(f"disabled_native_terminations={disabled_terminations}")
    print(f"completed_steps={steps}")
    print(f"outcome={result.outcome.value}")
    print(f"success={result.success}")
    print(f"episode_csv={writer.episode_path}")
    print(f"summary_json={writer.summary_path}")
    print(f"summary_episode_count={summary['episode_count']}")

    if step_up_summary is not None:
        print("")
        print("=== STEP-UP DETECTOR VALIDATION ===")
        print(f"checkpoint={policy_source}")
        print(
            "stance_force_median_n="
            f"{step_up_summary['stance_force_median_n']:.6f}"
        )
        print(
            "support_threshold_n="
            f"{step_up_summary['support_threshold_n']:.6f}"
        )
        print(
            "lower_base_height_m="
            f"{step_up_summary['lower_base_height_m']:.6f}"
        )
        print(
            "max_left_fz_n="
            f"{step_up_summary['max_left_fz_n']:.6f}"
        )
        print(
            "max_right_fz_n="
            f"{step_up_summary['max_right_fz_n']:.6f}"
        )
        print(
            "first_support_events="
            f"{step_up_summary['first_support_events']}"
        )
        print(
            "base_above_events="
            f"{step_up_summary['base_above_events']}"
        )
        print(
            "both_feet_supported_events="
            f"{step_up_summary['both_feet_supported_events']}"
        )
        print(
            "completed_events="
            f"{step_up_summary['completed_events']}"
        )
        print(
            "maximum_state="
            f"{step_up_summary['maximum_state']}"
        )
        print(
            "forbidden_contact_steps="
            f"{step_up_summary['forbidden_contact_steps']}"
        )
        print(
            "PASS/FAIL="
            + ("PASS" if step_up_summary["passed"] else "FAIL")
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac/Omniverse shutdown can block. Emit the original evaluator
        # failure before closing the application so it cannot be hidden.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
