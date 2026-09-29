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
        initialize_actor_from_rsl_rl_checkpoint,
        initialize_actor_from_unitree_onnx,
        initialize_zero_residual_actor,
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

    from simulation.isaac_lab.g1_stairs.runtime_contract import (
        reject_archived_pre_specialist_runtime,
    )

    reject_archived_pre_specialist_runtime(resolved_config)

    from simulation.isaac_lab.g1_stairs.specialist_runtime import (
        apply_specialist_runtime,
    )

    specialist_summary = apply_specialist_runtime(
        env_cfg,
        resolved_config,
    )
    if specialist_summary is not None:
        print(
            "[INFO] Specialist stair runtime:",
            specialist_summary,
            flush=True,
        )

    observations_cfg = resolved_config.get(
        "observations",
        {},
    )

    if observations_cfg.get(
        "actor_height_scan",
        True,
    ) is False:
        env_cfg.observations.policy.height_scan = None

        print(
            "[INFO] Actor height_scan: disabled",
            flush=True,
        )

    terrain_cfg = resolved_config.get("terrain", {})
    if "riser_heights_m" in terrain_cfg:
        values = terrain_cfg["riser_heights_m"]

        if (
            not isinstance(values, (list, tuple))
            or len(values) < 2
        ):
            raise ValueError(
                "terrain.riser_heights_m must contain at least two heights"
            )

        riser_heights_m = tuple(float(value) for value in values)
        if any(
            value < 0.0 or value > 0.20
            for value in riser_heights_m
        ):
            raise ValueError(
                "terrain.riser_heights_m values must be in [0.0, 0.20]"
            )

        if any(
            right <= left
            for left, right in zip(
                riser_heights_m,
                riser_heights_m[1:],
            )
        ):
            raise ValueError(
                "terrain.riser_heights_m must be strictly increasing"
            )

        terrain_generator = (
            env_cfg.scene.terrain.terrain_generator
        )
        stair_cfg = (
            terrain_generator
            .sub_terrains["three_step_ascent"]
        )

        terrain_generator.num_rows = len(riser_heights_m)
        stair_cfg.riser_heights = riser_heights_m
        stair_cfg.difficulty_bins = len(riser_heights_m)
        terrain_generator.curriculum = True

        curriculum_cfg = terrain_cfg.get("curriculum", {})
        if curriculum_cfg is None:
            curriculum_cfg = {}
        if not isinstance(curriculum_cfg, dict):
            raise ValueError("terrain.curriculum must be a mapping")

        curriculum_enabled = bool(
            curriculum_cfg.get("enabled", False)
        )
        if curriculum_enabled:
            initial_max_level = int(
                curriculum_cfg.get("initial_max_level", 1)
            )
            if not 0 <= initial_max_level < len(riser_heights_m):
                raise ValueError(
                    "terrain.curriculum.initial_max_level must index "
                    "terrain.riser_heights_m"
                )
            env_cfg.scene.terrain.max_init_terrain_level = (
                initial_max_level
            )

            from isaaclab.managers import CurriculumTermCfg
            from simulation.isaac_lab.g1_stairs import (
                terrain_curriculum,
            )

            env_cfg.curriculum.terrain_levels = CurriculumTermCfg(
                func=terrain_curriculum.stair_platform_levels,
                params={
                    "platform_margin_m": float(
                        curriculum_cfg.get(
                            "platform_margin_m", 0.10
                        )
                    ),
                    "minimum_upright_gravity_z": float(
                        curriculum_cfg.get(
                            "minimum_upright_gravity_z", 0.75
                        )
                    ),
                    "minimum_base_clearance_m": float(
                        curriculum_cfg.get(
                            "minimum_base_clearance_m", 0.55
                        )
                    ),
                },
            )
        else:
            env_cfg.scene.terrain.max_init_terrain_level = (
                len(riser_heights_m) - 1
            )

        print(
            "[INFO] Exact training riser heights m:",
            riser_heights_m,
            flush=True,
        )
        if curriculum_enabled:
            print(
                "[INFO] Stair terrain curriculum:",
                {
                    "initial_max_level": initial_max_level,
                    **env_cfg.curriculum.terrain_levels.params,
                },
                flush=True,
            )
        print(
            "[INFO] Stair difficulty_bins:",
            stair_cfg.difficulty_bins,
            flush=True,
        )

    elif "riser_height_range_m" in terrain_cfg:
        values = terrain_cfg[
            "riser_height_range_m"
        ]

        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
        ):
            raise ValueError(
                "terrain.riser_height_range_m "
                "must contain [min, max]"
            )

        riser_min_m = float(values[0])
        riser_max_m = float(values[1])

        if not (
            0.0
            < riser_min_m
            <= riser_max_m
            <= 0.20
        ):
            raise ValueError(
                "terrain.riser_height_range_m must satisfy "
                "0 < min <= max <= 0.20"
            )

        terrain_generator = (
            env_cfg.scene.terrain.terrain_generator
        )

        stair_cfg = (
            terrain_generator
            .sub_terrains["three_step_ascent"]
        )

        stair_cfg.riser_heights = None
        stair_cfg.riser_height_range = (
            riser_min_m,
            riser_max_m,
        )

        stair_cfg.difficulty_bins = int(
            terrain_generator.num_rows
        )

        terrain_generator.curriculum = True

        env_cfg.scene.terrain.max_init_terrain_level = (
            terrain_generator.num_rows - 1
        )

        print(
            "[INFO] Stair riser_height_range_m:",
            (riser_min_m, riser_max_m),
            flush=True,
        )

        print(
            "[INFO] Stair difficulty_bins:",
            stair_cfg.difficulty_bins,
            flush=True,
        )

    elif "riser_height_m" in terrain_cfg:
        riser_height_m = float(terrain_cfg["riser_height_m"])

        if not 0.0 < riser_height_m <= 0.20:
            raise ValueError(
                "terrain.riser_height_m must be in (0.0, 0.20]"
            )

        stair_cfg = (
            env_cfg.scene.terrain.terrain_generator
            .sub_terrains["three_step_ascent"]
        )
        stair_cfg.riser_heights = None
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

    if "tread_depth_m" in terrain_cfg:
        tread_depth_m = float(
            terrain_cfg["tread_depth_m"]
        )

        if not 0.15 <= tread_depth_m <= 0.40:
            raise ValueError(
                "terrain.tread_depth_m must be "
                "in [0.15, 0.40]"
            )

        stair_cfg = (
            env_cfg.scene.terrain.terrain_generator
            .sub_terrains["three_step_ascent"]
        )

        stair_cfg.tread_depth = (
            tread_depth_m
        )

        print(
            "[INFO] Stair tread_depth_m:",
            tread_depth_m,
            flush=True,
        )

    reset_cfg = resolved_config.get("reset_randomization", {})
    if reset_cfg is None:
        reset_cfg = {}
    if not isinstance(reset_cfg, dict):
        raise ValueError("reset_randomization must be a mapping")

    if bool(reset_cfg.get("enabled", False)):
        distance_range = reset_cfg.get(
            "distance_before_first_riser_m",
            (0.30, 1.00),
        )
        lateral_range = reset_cfg.get(
            "lateral_offset_m",
            (-0.05, 0.05),
        )
        yaw_range = reset_cfg.get(
            "yaw_offset_rad",
            (-0.0872664626, 0.0872664626),
        )

        for label, values in (
            ("distance_before_first_riser_m", distance_range),
            ("lateral_offset_m", lateral_range),
            ("yaw_offset_rad", yaw_range),
        ):
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 2
            ):
                raise ValueError(
                    f"reset_randomization.{label} must contain [min, max]"
                )

        distance_min, distance_max = (
            float(distance_range[0]),
            float(distance_range[1]),
        )
        lateral_min, lateral_max = (
            float(lateral_range[0]),
            float(lateral_range[1]),
        )
        yaw_min, yaw_max = (
            float(yaw_range[0]),
            float(yaw_range[1]),
        )

        stair_cfg = (
            env_cfg.scene.terrain.terrain_generator
            .sub_terrains["three_step_ascent"]
        )
        nominal_distance = float(stair_cfg.start_clearance)

        if not (0.0 < distance_min <= distance_max <= nominal_distance):
            raise ValueError(
                "reset_randomization distance range must satisfy "
                "0 < min <= max <= stair start_clearance"
            )
        if lateral_min > lateral_max:
            raise ValueError(
                "reset_randomization lateral range must be ordered"
            )
        if yaw_min > yaw_max:
            raise ValueError(
                "reset_randomization yaw range must be ordered"
            )

        # reset_root_state_uniform applies these as offsets from each terrain
        # origin.  The origin is nominally start_clearance metres before the
        # first riser, so positive X shortens that approach distance.
        x_offset_min = nominal_distance - distance_max
        x_offset_max = nominal_distance - distance_min

        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (x_offset_min, x_offset_max),
            "y": (lateral_min, lateral_max),
            "yaw": (yaw_min, yaw_max),
        }

        print(
            "[INFO] Stair reset randomization:",
            {
                "distance_before_first_riser_m": (
                    distance_min,
                    distance_max,
                ),
                "x_offset_m": (x_offset_min, x_offset_max),
                "lateral_offset_m": (lateral_min, lateral_max),
                "yaw_offset_rad": (yaw_min, yaw_max),
            },
            flush=True,
        )

    command_cfg = resolved_config.get("command", {})
    if "forward_velocity_mps" in command_cfg:
        forward_velocity_mps = float(
            command_cfg["forward_velocity_mps"]
        )

        if not 0.0 <= forward_velocity_mps <= 0.50:
            raise ValueError(
                "command.forward_velocity_mps must be in [0.0, 0.50]"
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

    from simulation.isaac_lab.g1_stairs.command_runtime import (
        apply_stair_heading_command,
    )

    heading_summary = apply_stair_heading_command(
        env_cfg,
        resolved_config,
    )
    if heading_summary is not None:
        print(
            "[INFO] Stair heading command:",
            heading_summary,
            flush=True,
        )


    stair_schedule_cfg = command_cfg.get(
        "stair_velocity_schedule",
        {},
    )

    if stair_schedule_cfg.get("enabled", False):
        from simulation.isaac_lab.g1_stairs import (
            terrain_observations,
        )

        schedule_params = {
            "approach_velocity_mps": float(
                stair_schedule_cfg[
                    "approach_velocity_mps"
                ]
            ),
            "stair_velocity_mps": float(
                stair_schedule_cfg[
                    "stair_velocity_mps"
                ]
            ),
            "slowdown_start_distance_m": float(
                stair_schedule_cfg[
                    "slowdown_start_distance_m"
                ]
            ),
            "slowdown_end_distance_m": float(
                stair_schedule_cfg[
                    "slowdown_end_distance_m"
                ]
            ),
            "platform_velocity_mps": float(
                stair_schedule_cfg[
                    "platform_velocity_mps"
                ]
            ),
            "platform_accel_distance_m": float(
                stair_schedule_cfg[
                    "platform_accel_distance_m"
                ]
            ),
            "centerline_gain_s": float(
                stair_schedule_cfg.get(
                    "centerline_gain_s",
                    0.0,
                )
            ),
            "max_lateral_velocity_mps": float(
                stair_schedule_cfg.get(
                    "max_lateral_velocity_mps",
                    0.0,
                )
            ),
            "heading_gain_s": float(
                stair_schedule_cfg.get(
                    "heading_gain_s",
                    0.0,
                )
            ),
            "max_yaw_rate_radps": float(
                stair_schedule_cfg.get(
                    "max_yaw_rate_radps",
                    0.0,
                )
            ),
            "stop_after_ascent_start": bool(
                stair_schedule_cfg.get(
                    "stop_after_ascent_start",
                    False,
                )
            ),
        }

        # Preserve the original three command channels and their five-frame
        # history. Only their values become geometry-conditioned.
        observation_groups = [
            env_cfg.observations.policy,
            env_cfg.observations.critic,
        ]
        for observation_group in observation_groups:
            observation_group.velocity_commands.func = (
                terrain_observations.stair_velocity_command
            )
            observation_group.velocity_commands.params = dict(
                schedule_params
            )

        env_cfg.rewards.track_stair_velocity_world.params.update(
            schedule_params
        )

        print(
            "[INFO] Stair velocity schedule:",
            schedule_params,
            flush=True,
        )

    from simulation.isaac_lab.g1_stairs.reference_tracking_runtime import (
        apply_reference_tracking_config,
    )

    reference_tracking_cfg = apply_reference_tracking_config(
        env_cfg,
        resolved_config,
    )

    if reference_tracking_cfg:
        print(
            "[INFO] Reference tracking:",
            reference_tracking_cfg,
            flush=True,
        )

    active_rewards = apply_resolved_rewards(env_cfg, resolved_config)
    print("[INFO] Active reward stage:", resolved_config["reward_experiment"]["active_stage"], flush=True)
    print("[INFO] Active rewards:", active_rewards, flush=True)
    if resolved_config.get("reward_parameters"):
        print(
            "[INFO] Reward parameters:",
            resolved_config["reward_parameters"],
            flush=True,
        )

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    ppo_cfg = resolved_config.get("ppo", {})

    checkpointing_cfg = resolved_config.get("checkpointing", {})
    if checkpointing_cfg is None:
        checkpointing_cfg = {}
    if not isinstance(checkpointing_cfg, dict):
        raise ValueError("checkpointing must be a mapping")

    checkpoint_interval = int(
        checkpointing_cfg.get(
            "interval_iterations",
            agent_cfg.save_interval,
        )
    )
    if checkpoint_interval <= 0:
        raise ValueError(
            "checkpointing.interval_iterations must be > 0"
        )
    agent_cfg.save_interval = checkpoint_interval

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

    if "num_mini_batches" in ppo_cfg:
        num_mini_batches = int(
            ppo_cfg["num_mini_batches"]
        )
        if num_mini_batches <= 0:
            raise ValueError(
                "ppo.num_mini_batches must be > 0"
            )
        agent_cfg.algorithm.num_mini_batches = num_mini_batches

    if "action_clip" in ppo_cfg:
        action_clip = float(ppo_cfg["action_clip"])

        if action_clip <= 0.0:
            raise ValueError(
                "ppo.action_clip must be > 0"
            )

        agent_cfg.clip_actions = action_clip

    print(
        "[INFO] Checkpoint interval:",
        agent_cfg.save_interval,
        flush=True,
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
    print(
        "[INFO] PPO num_mini_batches:",
        agent_cfg.algorithm.num_mini_batches,
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

    from simulation.isaac_lab.g1_stairs.geometry_episode_runtime import apply_geometry_episode_contract
    apply_geometry_episode_contract(env_cfg, resolved_config)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if (
        "riser_heights_m" in terrain_cfg
        or "riser_height_range_m" in terrain_cfg
    ):
        from simulation.isaac_lab.g1_stairs.terrain_state import (
            riser_height_per_env,
        )

        terrain_levels = (
            env.unwrapped.scene.terrain.terrain_levels
        )

        unique_levels, level_counts = (
            terrain_levels.unique(
                return_counts=True
            )
        )

        exact_heights = (
            riser_height_per_env(
                env.unwrapped
            ).unique()
        )

        print(
            "[INFO] Terrain levels/counts:",
            list(
                zip(
                    unique_levels.detach().cpu().tolist(),
                    level_counts.detach().cpu().tolist(),
                )
            ),
            flush=True,
        )

        print(
            "[INFO] Exact riser heights present:",
            exact_heights.detach().cpu().tolist(),
            flush=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    from simulation.isaac_lab.g1_stairs.balance_residual import (
        wrap_balance_residual_from_config,
    )

    env = wrap_balance_residual_from_config(
        env,
        resolved_config,
    )

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
    elif initialization_type == "rsl_rl_checkpoint_actor_transfer":
        initialize_actor_from_rsl_rl_checkpoint(
            runner=runner,
            checkpoint_path=Path(
                initialization_cfg["source"]["path"]
            ),
        )
    elif initialization_type == "rsl_rl_training_resume":
        if resolved_config.get("ppo", {}).get("observation_normalization", False):
            raise ValueError("This resume path requires observation_normalization=false")
        from simulation.isaac_lab.g1_stairs.training_resume import resume_training
        resume_training(
            runner,
            initialization_cfg["source"]["path"],
            initialization_cfg["source"]["sha256"],
        )
    elif initialization_type == "zero_residual_actor":
        initialize_zero_residual_actor(runner)
    elif initialization_type not in (None, "random"):
        raise RuntimeError(
            "Unsupported policy initialization: "
            f"{initialization_type}"
        )

    runner.configure_metrics(
        Path(metrics_dir_value),
        run_id,
        checkpointing=checkpointing_cfg,
    )
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
    except BaseException as exc:
        import traceback

        print(
            f"[FATAL] train_isaac_stairs.py: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
