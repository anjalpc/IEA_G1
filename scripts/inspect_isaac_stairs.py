#!/usr/bin/env python3
"""Render the nominal G1 staircase without requiring a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one mode-5 G1 on the exact nominal staircase. "
            "Physics is frozen unless --simulate is supplied."
        )
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Advance physics using zero actions; the robot may fall.",
    )
    parser.add_argument(
        "--audit-evaluation-inputs",
        action="store_true",
        help="Print the live robot/contact fields required by the evaluator and exit.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Kit-dependent imports must happen after AppLauncher starts Isaac Sim.
    import gymnasium as gym
    import torch

    from isaac_lab.g1_stairs.contracts import (
        NOMINAL_RISER_HEIGHT_M,
        TASK_ID,
    )
    from isaac_lab.g1_stairs.env_cfg import StairPlayEnvCfg
    from isaac_lab.g1_stairs.register import register

    register()

    cfg = StairPlayEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 6.0

    terrain_generator = cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise RuntimeError("stair terrain generator is missing")

    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    stair_cfg = terrain_generator.sub_terrains["three_step_ascent"]
    expected_risers = (
        NOMINAL_RISER_HEIGHT_M,
        NOMINAL_RISER_HEIGHT_M,
    )

    if tuple(stair_cfg.riser_height_range) != expected_risers:
        raise RuntimeError(
            "inspection must use exact 0.20 m risers; "
            f"got {stair_cfg.riser_height_range}"
        )

    cfg.viewer.eye = (7.0, -6.0, 4.0)
    cfg.viewer.lookat = (2.8, 1.0, 0.5)

    print("=== NOMINAL STAIR INSPECTION CONTRACT ===")
    print(f"task_id={TASK_ID}")
    print("robot=G1 EDU U7 mode_machine 5")
    print("policy_dofs=29")
    print("riser_height_m=0.20")
    print("tread_depth_m=0.25")
    print("step_count=3")
    print("total_rise_m=0.60")
    print("platform_length_m=3.00")
    print("start_position_m=(1.00, 0.00, 0.80)")
    print(f"simulate={args.simulate}")

    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()

    try:
        if args.audit_evaluation_inputs:
            robot = env.unwrapped.scene["robot"]

            print("=== EVALUATION INPUT AUDIT ===", flush=True)
            print(f"body_names={list(robot.body_names)}", flush=True)
            print(f"joint_names={list(robot.joint_names)}", flush=True)

            print("--- ARTICULATION TENSORS ---", flush=True)

            for field in (
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "joint_pos",
                "joint_vel",
                "joint_acc",
                "applied_torque",
                "computed_torque",
                "joint_effort_target",
                "joint_effort_limits",
                "joint_pos_limits",
                "soft_joint_pos_limits",
                "body_incoming_joint_wrench_b",
            ):
                try:
                    value = getattr(robot.data, field)
                    print(
                        f"{field}: "
                        f"type={type(value).__name__}, "
                        f"shape={getattr(value, 'shape', None)}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"{field}: ERROR "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

            print("--- SCENE SENSORS ---", flush=True)

            try:
                sensors = env.unwrapped.scene.sensors
                print(
                    f"sensor_names={list(sensors.keys())}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"sensor_registry: ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                sensors = {}

            for name, sensor in sensors.items():
                print(f"sensor={name}", flush=True)
                print(
                    f"sensor_type={type(sensor).__name__}",
                    flush=True,
                )
                print(
                    f"sensor_body_names="
                    f"{list(getattr(sensor, 'body_names', []))}",
                    flush=True,
                )

                data = getattr(sensor, "data", None)

                if data is None:
                    print("sensor_data=None", flush=True)
                    continue

                for field in (
                    "net_forces_w",
                    "net_forces_w_history",
                    "force_matrix_w",
                    "force_matrix_w_history",
                ):
                    try:
                        value = getattr(data, field)
                        print(
                            f"{field}: "
                            f"type={type(value).__name__}, "
                            f"shape={getattr(value, 'shape', None)}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"{field}: ERROR "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )

            print("=== END EVALUATION INPUT AUDIT ===", flush=True)
            return

        if args.simulate:
            action_dim = env.unwrapped.action_manager.total_action_dim
            actions = torch.zeros(
                (env.unwrapped.num_envs, action_dim),
                device=env.unwrapped.device,
            )

            while simulation_app.is_running():
                with torch.inference_mode():
                    env.step(actions)
        else:
            print(
                "Physics is frozen. Close Isaac Sim after inspection."
            )

            while simulation_app.is_running():
                env.unwrapped.sim.render()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
