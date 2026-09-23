#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[5]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNITREE_RL_ROOT = Path(
    os.environ.get(
        "UNITREE_RL_LAB_ROOT",
        "/workspace/unitree_rl_lab",
    )
)

from simulation.isaac_lab.g1_stairs.unitree_bootstrap import (
    bootstrap_unitree_rl_lab,
)

bootstrap_unitree_rl_lab(UNITREE_RL_ROOT)

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(
    description="Read-only G1 first-riser mechanics diagnostic."
)

parser.add_argument(
    "--task",
    default="Unitree-IEA-G1-29dof-Stairs-v0",
)

parser.add_argument(
    "--resolved-config",
    required=True,
)

parser.add_argument(
    "--output-dir",
    required=True,
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=600,
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def quat_pitch_wxyz(q) -> float:
    w, x, y, z = (
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )

    sin_pitch = 2.0 * (w * y - z * x)

    sin_pitch = max(
        -1.0,
        min(1.0, sin_pitch),
    )

    return math.asin(sin_pitch)


def distance(a, b) -> float:
    return math.sqrt(
        sum(
            (
                float(a[i])
                - float(b[i])
            ) ** 2
            for i in range(3)
        )
    )


def main() -> None:
    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner

    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from isaaclab.envs import (
        DirectMARLEnv,
        multi_agent_to_single_agent,
    )

    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
    )

    from unitree_rl_lab.utils.parser_cfg import (
        parse_env_cfg,
    )

    from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
        apply_resolved_replay_config,
    )

    from simulation.isaac_lab.g1_stairs.reward_runtime import (
        load_resolved_config,
    )

    from simulation.isaac_lab.g1_stairs.reference_tracking_runtime import (
        apply_reference_tracking_config,
    )

    from simulation.isaac_lab.g1_stairs.terrain_observations import (
        stair_velocity_command,
    )

    from simulation.isaac_lab.g1_stairs.terrain_state import (
        stair_cfg,
        riser_height_per_env,
    )

    checkpoint = Path(
        args.checkpoint
    ).resolve()

    resolved_path = Path(
        args.resolved_config
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
        )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            resolved_path
        )

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )

    env_cfg.seed = args.seed
    env_cfg.commands.base_velocity.debug_vis = False

    replay = apply_resolved_replay_config(
        env_cfg,
        resolved_path,
    )

    resolved = load_resolved_config(
        resolved_path
    )

    apply_reference_tracking_config(
        env_cfg,
        resolved,
    )

    terrain_generator = (
        env_cfg.scene.terrain.terrain_generator
    )

    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    agent_cfg = cli_args.parse_rsl_rl_cfg(
        args.task,
        args,
    )

    agent_cfg.seed = args.seed

    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None,
    )

    if isinstance(
        env.unwrapped,
        DirectMARLEnv,
    ):
        env = multi_agent_to_single_agent(
            env
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

    runner.load(
        str(checkpoint)
    )

    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    obs = env.get_observations()

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    sensor = base_env.scene.sensors[
        "contact_forces"
    ]

    joint_names = tuple(
        robot.joint_names
    )

    body_names = tuple(
        robot.body_names
    )

    sensor_names = tuple(
        sensor.body_names
    )

    required_joints = (
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
    )

    required_bodies = (
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    )

    missing_joints = [
        name
        for name in required_joints
        if name not in joint_names
    ]

    missing_bodies = [
        name
        for name in required_bodies
        if name not in body_names
    ]

    if missing_joints:
        raise RuntimeError(
            "Missing joints: "
            + ", ".join(missing_joints)
        )

    if missing_bodies:
        raise RuntimeError(
            "Missing bodies: "
            + ", ".join(missing_bodies)
        )

    joint_index = {
        name: joint_names.index(name)
        for name in required_joints
    }

    body_index = {
        name: body_names.index(name)
        for name in required_bodies
    }

    ankle_sensor_index = {}

    for side in ("left", "right"):
        ankle = (
            f"{side}_ankle_roll_link"
        )

        if ankle not in sensor_names:
            raise RuntimeError(
                f"{ankle} missing from contact sensor"
            )

        ankle_sensor_index[side] = (
            sensor_names.index(ankle)
        )

    cfg = stair_cfg(base_env)

    first_riser_x = float(
        cfg.start_clearance
    )

    tread_depth = float(
        cfg.tread_depth
    )

    step_count = int(
        cfg.step_count
    )

    riser_height = float(
        riser_height_per_env(
            base_env
        )[0].item()
    )

    env_origin = (
        base_env.scene.env_origins[0]
    )

    print(
        "[MECH] checkpoint =",
        checkpoint,
        flush=True,
    )

    print(
        "[MECH] resolved_config =",
        resolved_path,
        flush=True,
    )

    print(
        "[MECH] first_riser_x_local =",
        first_riser_x,
        flush=True,
    )

    print(
        "[MECH] riser_height_m =",
        riser_height,
        flush=True,
    )

    print(
        "[MECH] tread_depth_m =",
        tread_depth,
        flush=True,
    )

    print(
        "[MECH] step_count =",
        step_count,
        flush=True,
    )

    rows = []

    def local_position(name):
        p = robot.data.body_pos_w[
            0,
            body_index[name],
        ]

        return (
            float(
                p[0].item()
                - env_origin[0].item()
            ),
            float(
                p[1].item()
                - env_origin[1].item()
            ),
            float(
                p[2].item()
                - env_origin[2].item()
            ),
        )

    def body_vz(name):
        return float(
            robot.data.body_lin_vel_w[
                0,
                body_index[name],
                2,
            ].item()
        )

    def contact(side):
        idx = ankle_sensor_index[
            side
        ]

        return int(
            float(
                sensor.data.current_contact_time[
                    0,
                    idx,
                ].item()
            )
            > 0.0
        )

    def next_riser_distance(
        root_x: float,
    ) -> float:
        edges = [
            first_riser_x
            + i * tread_depth
            for i in range(
                step_count
            )
        ]

        ahead = [
            edge - root_x
            for edge in edges
            if edge >= root_x
        ]

        if not ahead:
            return 0.0

        return min(ahead)

    near_riser_seen = False

    for step in range(
        args.max_steps
    ):
        with torch.inference_mode():
            actions = policy(obs)

            obs, _, _, _ = env.step(
                actions
            )

        root_pos = (
            robot.data.root_pos_w[0]
        )

        root_x = float(
            root_pos[0].item()
            - env_origin[0].item()
        )

        root_z = float(
            root_pos[2].item()
            - env_origin[2].item()
        )

        root_vx = float(
            robot.data.root_lin_vel_w[
                0,
                0,
            ].item()
        )

        pitch = quat_pitch_wxyz(
            robot.data.root_quat_w[
                0
            ]
        )

        distance_to_riser = (
            next_riser_distance(
                root_x
            )
        )

        target_vx = float(
            stair_velocity_command(
                base_env
            )[0, 0].item()
        )

        left_hip = local_position(
            "left_hip_pitch_link"
        )

        left_knee = local_position(
            "left_knee_link"
        )

        left_ankle = local_position(
            "left_ankle_roll_link"
        )

        right_hip = local_position(
            "right_hip_pitch_link"
        )

        right_knee = local_position(
            "right_knee_link"
        )

        right_ankle = local_position(
            "right_ankle_roll_link"
        )

        row = {
            "step": step,
            "time_s": (
                step
                * float(
                    base_env.step_dt
                )
            ),
            "root_x_m": root_x,
            "distance_to_next_riser_m": (
                distance_to_riser
            ),
            "target_vx_mps": target_vx,
            "root_vx_mps": root_vx,
            "root_height_m": root_z,
            "root_pitch_rad": pitch,

            "left_hip_pitch_rad": float(
                robot.data.joint_pos[
                    0,
                    joint_index[
                        "left_hip_pitch_joint"
                    ],
                ].item()
            ),
            "left_knee_rad": float(
                robot.data.joint_pos[
                    0,
                    joint_index[
                        "left_knee_joint"
                    ],
                ].item()
            ),
            "left_ankle_pitch_rad": float(
                robot.data.joint_pos[
                    0,
                    joint_index[
                        "left_ankle_pitch_joint"
                    ],
                ].item()
            ),
            "left_hip_z_m": left_hip[2],
            "left_knee_z_m": left_knee[2],
            "left_ankle_x_m": left_ankle[0],
            "left_ankle_z_m": left_ankle[2],
            "left_ankle_vz_mps": body_vz(
                "left_ankle_roll_link"
            ),
            "left_leg_length_m": distance(
                left_hip,
                left_ankle,
            ),
            "left_contact": contact(
                "left"
            ),

            "right_hip_pitch_rad": float(
                robot.data.joint_pos[
                    0,
                    joint_index[
                        "right_hip_pitch_joint"
                    ],
                ].item()
            ),
            "right_knee_rad": float(
                robot.data.joint_pos[
                    0,
                    joint_index[
                        "right_knee_joint"
                    ],
                ].item()
            ),
            "right_ankle_pitch_rad": float(
                robot.data.joint_pos[
                    0,
                    joint_index[
                        "right_ankle_pitch_joint"
                    ],
                ].item()
            ),
            "right_hip_z_m": right_hip[2],
            "right_knee_z_m": right_knee[2],
            "right_ankle_x_m": right_ankle[0],
            "right_ankle_z_m": right_ankle[2],
            "right_ankle_vz_mps": body_vz(
                "right_ankle_roll_link"
            ),
            "right_leg_length_m": distance(
                right_hip,
                right_ankle,
            ),
            "right_contact": contact(
                "right"
            ),
        }

        rows.append(row)

        if (
            0.0
            <= distance_to_riser
            <= 0.30
        ):
            near_riser_seen = True

        # Stop after the first clear failure rather than
        # allowing an environment reset to contaminate
        # the first-riser trace.
        if near_riser_seen and (
            root_z < 0.55
            or abs(pitch) > 0.80
        ):
            print(
                "[MECH] first failure detected at step",
                step,
                flush=True,
            )

            break

    csv_path = (
        output_dir
        / "mechanics_trace.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    def phase_rows(
        minimum,
        maximum,
    ):
        return [
            row
            for row in rows
            if minimum
            <= row[
                "distance_to_next_riser_m"
            ]
            < maximum
        ]

    phases = {
        "far_approach": phase_rows(
            0.60,
            10.0,
        ),
        "slowdown": phase_rows(
            0.30,
            0.60,
        ),
        "pre_riser": phase_rows(
            0.02,
            0.30,
        ),
        "riser_edge": phase_rows(
            0.0,
            0.02,
        ),
    }

    def summarize_phase(
        values,
    ):
        if not values:
            return None

        result = {
            "samples": len(values),
            "root_vx_mean": sum(
                row["root_vx_mps"]
                for row in values
            ) / len(values),
            "target_vx_mean": sum(
                row["target_vx_mps"]
                for row in values
            ) / len(values),
            "root_pitch_abs_max_rad": max(
                abs(
                    row[
                        "root_pitch_rad"
                    ]
                )
                for row in values
            ),
        }

        for side in (
            "left",
            "right",
        ):
            result[
                f"{side}_ankle_z_max_m"
            ] = max(
                row[
                    f"{side}_ankle_z_m"
                ]
                for row in values
            )

            result[
                f"{side}_ankle_vz_max_mps"
            ] = max(
                row[
                    f"{side}_ankle_vz_mps"
                ]
                for row in values
            )

            result[
                f"{side}_leg_length_min_m"
            ] = min(
                row[
                    f"{side}_leg_length_m"
                ]
                for row in values
            )

            result[
                f"{side}_hip_pitch_range_rad"
            ] = (
                max(
                    row[
                        f"{side}_hip_pitch_rad"
                    ]
                    for row in values
                )
                - min(
                    row[
                        f"{side}_hip_pitch_rad"
                    ]
                    for row in values
                )
            )

            result[
                f"{side}_knee_range_rad"
            ] = (
                max(
                    row[
                        f"{side}_knee_rad"
                    ]
                    for row in values
                )
                - min(
                    row[
                        f"{side}_knee_rad"
                    ]
                    for row in values
                )
            )

            result[
                f"{side}_ankle_pitch_range_rad"
            ] = (
                max(
                    row[
                        f"{side}_ankle_pitch_rad"
                    ]
                    for row in values
                )
                - min(
                    row[
                        f"{side}_ankle_pitch_rad"
                    ]
                    for row in values
                )
            )

        return result

    summary = {
        "checkpoint": str(
            checkpoint
        ),
        "resolved_config": str(
            resolved_path
        ),
        "riser_height_m": (
            riser_height
        ),
        "first_riser_x_local_m": (
            first_riser_x
        ),
        "rows": len(rows),
        "phases": {
            name: summarize_phase(
                values
            )
            for name, values
            in phases.items()
        },
    }

    far = summary[
        "phases"
    ]["far_approach"]

    pre = summary[
        "phases"
    ]["pre_riser"]

    if (
        far is not None
        and pre is not None
    ):
        comparison = {}

        for side in (
            "left",
            "right",
        ):
            comparison[
                f"{side}_extra_ankle_height_vs_far_m"
            ] = (
                pre[
                    f"{side}_ankle_z_max_m"
                ]
                - far[
                    f"{side}_ankle_z_max_m"
                ]
            )

            comparison[
                f"{side}_extra_leg_shortening_vs_far_m"
            ] = (
                far[
                    f"{side}_leg_length_min_m"
                ]
                - pre[
                    f"{side}_leg_length_min_m"
                ]
            )

            comparison[
                f"{side}_extra_knee_range_vs_far_rad"
            ] = (
                pre[
                    f"{side}_knee_range_rad"
                ]
                - far[
                    f"{side}_knee_range_rad"
                ]
            )

        summary[
            "pre_riser_vs_far"
        ] = comparison

    summary_path = (
        output_dir
        / "mechanics_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "========================================"
    )
    print(
        " G1 FIRST-RISER MECHANICS SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "trace_csv=",
        csv_path,
        sep="",
    )

    print(
        "summary_json=",
        summary_path,
        sep="",
    )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
