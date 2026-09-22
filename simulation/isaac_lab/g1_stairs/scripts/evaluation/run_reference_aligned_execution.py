#!/usr/bin/env python3
"""032: execute the validated reference from its dynamically aligned state."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Execute the validated G1 stair reference after aligning "
        "root pose, root velocity, joint position, and joint velocity "
        "to reference frame 0."
    )
)

parser.add_argument(
    "--task",
    default="Unitree-IEA-G1-29dof-Stairs-v0",
)

parser.add_argument(
    "--reference",
    default=(
        "experiments/g1_stairs/"
        "reference_data/stair_ascent_reference.yaml"
    ),
)

parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--end-frame", type=int, default=246)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--zero-initial-velocity",
    action="store_true",
    help="Reset root and joints with zero velocity.",
)

parser.add_argument(
    "--hold-s",
    type=float,
    default=0.0,
    help="Hold the final commanded reference pose after playback.",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import simulation.isaac_lab.g1_stairs.register  # noqa: F401

from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS
from simulation.isaac_lab.g1_stairs.reference_motion import (
    load_reference_motion,
)


REPO_ROOT = Path(__file__).resolve().parents[5]


def as_ids(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for w,x,y,z quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[1:] *= -1.0
    return q


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def source_to_task_position(p: np.ndarray) -> np.ndarray:
    """Source frame -> thesis task frame: X=-Y, Y=X, Z=Z."""
    p = np.asarray(p, dtype=np.float64)

    return np.stack(
        (
            -p[..., 1],
            p[..., 0],
            p[..., 2],
        ),
        axis=-1,
    )


def source_to_task_quaternion(q: np.ndarray) -> np.ndarray:
    """Rotate source world orientation +90 deg about world Z."""
    q = quat_normalize(q)

    half = math.pi / 4.0
    q_world_rotation = np.asarray(
        [
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        ],
        dtype=np.float64,
    )

    return quat_normalize(
        quat_mul(q_world_rotation, q)
    )


def angular_velocity_from_quaternions(
    q0: np.ndarray,
    q1: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Estimate world angular velocity from consecutive wxyz poses."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    # Keep the shortest quaternion path.
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    q_delta = quat_normalize(
        quat_mul(q1, quat_conjugate(q0))
    )

    w = float(np.clip(q_delta[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)

    sin_half = math.sqrt(
        max(1.0 - w * w, 0.0)
    )

    if sin_half < 1.0e-8 or angle < 1.0e-8:
        return np.zeros(3, dtype=np.float64)

    axis = q_delta[1:] / sin_half

    # Map angle to shortest representation.
    if angle > math.pi:
        angle -= 2.0 * math.pi

    return axis * (angle / dt)


def main() -> int:
    reference_path = Path(args.reference)

    if not reference_path.is_absolute():
        reference_path = REPO_ROOT / reference_path

    motion = load_reference_motion(reference_path)

    if args.start_frame < 0:
        raise ValueError("start-frame must be >= 0")

    if args.end_frame >= motion.frame_count:
        raise ValueError(
            f"end-frame {args.end_frame} exceeds "
            f"{motion.frame_count - 1}"
        )

    if args.end_frame < args.start_frame:
        raise ValueError(
            "end-frame must be >= start-frame"
        )

    if args.start_frame + 1 >= motion.frame_count:
        raise ValueError(
            "At least two reference frames are required "
            "to estimate initial root velocity"
        )

    print("[032] loading environment", flush=True)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
    )

    env_cfg.seed = args.seed

    # One deterministic course only.
    terrain_generator = (
        env_cfg.scene.terrain.terrain_generator
    )
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None,
    )

    env.reset(seed=args.seed)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    sim = base_env.sim

    body_names = list(robot.body_names)

    try:
        pelvis_id = body_names.index("pelvis")
    except ValueError:
        pelvis_id = 0

    joint_result = robot.find_joints(
        list(BODY_JOINTS),
        preserve_order=True,
    )

    if isinstance(joint_result, tuple):
        joint_ids = as_ids(joint_result[0])
        resolved_joint_names = list(joint_result[1])
    else:
        joint_ids = as_ids(joint_result)
        resolved_joint_names = [
            robot.joint_names[index]
            for index in joint_ids
        ]

    if tuple(resolved_joint_names) != tuple(BODY_JOINTS):
        raise RuntimeError(
            "Resolved joint order does not match BODY_JOINTS.\n"
            f"expected={BODY_JOINTS}\n"
            f"actual={tuple(resolved_joint_names)}"
        )

    start = args.start_frame
    end = args.end_frame

    reference_q_np = np.asarray(
        motion.dof_pos[start : end + 1],
        dtype=np.float32,
    )

    reference_dq_np = np.asarray(
        motion.dof_vel[start : end + 1],
        dtype=np.float32,
    )

    body_pos_source = np.asarray(
        motion.body_pos_w[
            start : end + 1,
            pelvis_id,
        ],
        dtype=np.float64,
    )

    body_quat_source = np.asarray(
        motion.body_quat_w[
            start : end + 1,
            pelvis_id,
        ],
        dtype=np.float64,
    )

    reference_pos_task = source_to_task_position(
        body_pos_source
    )

    reference_quat_task = np.stack(
        [
            source_to_task_quaternion(q)
            for q in body_quat_source
        ],
        axis=0,
    )

    reference_dt = 1.0 / float(
        motion.frequency_hz
    )

    if args.zero_initial_velocity:
        initial_linear_velocity = np.zeros(
            3,
            dtype=np.float64,
        )
        initial_angular_velocity = np.zeros(
            3,
            dtype=np.float64,
        )
    else:
        # Initial velocity requires frame start+1 even when the
        # requested execution interval itself contains only one frame.
        initial_pos_pair_source = np.asarray(
            motion.body_pos_w[
                start : start + 2,
                pelvis_id,
            ],
            dtype=np.float64,
        )

        initial_pos_pair_task = (
            source_to_task_position(
                initial_pos_pair_source
            )
        )

        initial_quat_pair_source = np.asarray(
            motion.body_quat_w[
                start : start + 2,
                pelvis_id,
            ],
            dtype=np.float64,
        )

        initial_quat_pair_task = np.stack(
            [
                source_to_task_quaternion(q)
                for q in initial_quat_pair_source
            ],
            axis=0,
        )

        initial_linear_velocity = (
            initial_pos_pair_task[1]
            - initial_pos_pair_task[0]
        ) / reference_dt

        initial_angular_velocity = (
            angular_velocity_from_quaternions(
                initial_quat_pair_task[0],
                initial_quat_pair_task[1],
                reference_dt,
            )
        )

    physics_dt = float(sim.get_physics_dt())

    physics_steps_per_reference = max(
        1,
        int(round(reference_dt / physics_dt)),
    )

    device = robot.device

    reference_q = torch.as_tensor(
        reference_q_np,
        dtype=torch.float32,
        device=device,
    )

    reference_dq = torch.as_tensor(
        reference_dq_np,
        dtype=torch.float32,
        device=device,
    )

    #
    # Dynamic reference-aligned reset
    #
    # Preserve the environment's correct task start XY location.
    # Only use the reference's vertical root position and orientation.
    #
    current_pelvis = robot.data.body_pos_w[
        0,
        pelvis_id,
    ].clone()

    root_pose = torch.zeros(
        (1, 7),
        dtype=torch.float32,
        device=device,
    )

    root_pose[0, 0] = current_pelvis[0]
    root_pose[0, 1] = current_pelvis[1]
    root_pose[0, 2] = float(
        reference_pos_task[0, 2]
    )

    root_pose[0, 3:7] = torch.as_tensor(
        reference_quat_task[0],
        dtype=torch.float32,
        device=device,
    )

    root_velocity = torch.zeros(
        (1, 6),
        dtype=torch.float32,
        device=device,
    )

    root_velocity[0, :3] = torch.as_tensor(
        initial_linear_velocity,
        dtype=torch.float32,
        device=device,
    )

    root_velocity[0, 3:] = torch.as_tensor(
        initial_angular_velocity,
        dtype=torch.float32,
        device=device,
    )

    q0 = reference_q[0].unsqueeze(0)
    dq0 = reference_dq[0].unsqueeze(0)

    if args.zero_initial_velocity:
        root_velocity.zero_()
        dq0.zero_()

    print("=" * 80)
    print("032 DYNAMICALLY ALIGNED REFERENCE EXECUTION")
    print("=" * 80)
    print(f"reference:       {reference_path}")
    print(f"frames:          {start}..{end}")
    print(
        f"frequency:       "
        f"{motion.frequency_hz:.1f} Hz"
    )
    print(
        f"physics dt:      {physics_dt:.6f} s"
    )
    print(
        "physics/ref:     "
        f"{physics_steps_per_reference}"
    )
    print(
        "root position:   "
        f"{root_pose[0, :3].tolist()}"
    )
    print(
        "root quaternion: "
        f"{root_pose[0, 3:7].tolist()}"
    )
    print(
        "root lin vel:    "
        f"{root_velocity[0, :3].tolist()}"
    )
    print(
        "root ang vel:    "
        f"{root_velocity[0, 3:].tolist()}"
    )
    print(
        "joint vel RMS:   "
        f"{float(torch.sqrt(torch.mean(dq0 ** 2))):.4f}"
    )
    print()

    # Root link pose is the pelvis frame for this articulation.
    robot.write_root_pose_to_sim(root_pose)

    # Use root-link velocity because the reference pelvis trajectory
    # describes the root-link motion rather than an independently
    # reconstructed COM trajectory.
    robot.write_root_link_velocity_to_sim(
        root_velocity
    )

    robot.write_joint_state_to_sim(
        q0,
        dq0,
        joint_ids=joint_ids,
    )

    robot.reset()

    base_env.scene.write_data_to_sim()

    def simulate_target(
        target: torch.Tensor,
    ) -> None:
        target_batch = target.unsqueeze(0)

        for _ in range(
            physics_steps_per_reference
        ):
            robot.set_joint_position_target(
                target_batch,
                joint_ids=joint_ids,
            )

            base_env.scene.write_data_to_sim()
            sim.step()
            base_env.scene.update(physics_dt)

    #
    # No policy.
    # No residual.
    # No blending.
    #
    # q_target(t) = q_ref(t)
    #

    initial_world_xy = root_pose[
        0,
        :2,
    ].detach().clone()

    expected_reference_xy0 = (
        reference_pos_task[0, :2].copy()
    )

    fallen = False
    fall_frame = None

    lowest_pelvis_z = float("inf")
    max_joint_rmse = 0.0
    max_root_position_error = 0.0

    print(
        "Starting pure reference execution...",
        flush=True,
    )

    for local_index, target in enumerate(
        reference_q
    ):
        frame = start + local_index

        simulate_target(target)

        pelvis = robot.data.body_pos_w[
            0,
            pelvis_id,
        ]

        actual_q = robot.data.joint_pos[
            0,
            joint_ids,
        ]

        joint_rmse = float(
            torch.sqrt(
                torch.mean(
                    (actual_q - target) ** 2
                )
            ).item()
        )

        max_joint_rmse = max(
            max_joint_rmse,
            joint_rmse,
        )

        pelvis_z = float(pelvis[2].item())

        lowest_pelvis_z = min(
            lowest_pelvis_z,
            pelvis_z,
        )

        reference_delta_xy = (
            reference_pos_task[
                local_index,
                :2,
            ]
            - expected_reference_xy0
        )

        expected_world_pos = torch.tensor(
            [
                float(
                    initial_world_xy[0].item()
                    + reference_delta_xy[0]
                ),
                float(
                    initial_world_xy[1].item()
                    + reference_delta_xy[1]
                ),
                float(
                    reference_pos_task[
                        local_index,
                        2,
                    ]
                ),
            ],
            device=device,
        )

        root_position_error = float(
            torch.linalg.vector_norm(
                pelvis - expected_world_pos
            ).item()
        )

        max_root_position_error = max(
            max_root_position_error,
            root_position_error,
        )

        if (
            frame == start
            or frame % 25 == 0
            or frame == end
        ):
            print(
                f"frame={frame:3d} "
                f"time={local_index / motion.frequency_hz:5.2f}s "
                f"pelvis=({float(pelvis[0]):+.3f},"
                f"{float(pelvis[1]):+.3f},"
                f"{pelvis_z:+.3f}) "
                f"q_rmse={joint_rmse:.4f} "
                f"root_err={root_position_error:.4f}",
                flush=True,
            )

        # Conservative fall indicator.
        if pelvis_z < 0.45:
            fallen = True
            fall_frame = frame
            break

    if not fallen and args.hold_s > 0.0:
        hold_frames = max(
            1,
            int(round(args.hold_s * motion.frequency_hz)),
        )

        print(
            f"Holding final pose for {args.hold_s:.2f}s "
            f"({hold_frames} reference frames)...",
            flush=True,
        )

        final_target = reference_q[-1]

        for hold_index in range(hold_frames):
            simulate_target(final_target)

            pelvis = robot.data.body_pos_w[
                0,
                pelvis_id,
            ]

            pelvis_z = float(pelvis[2].item())

            if (
                hold_index == 0
                or (hold_index + 1) % 5 == 0
                or hold_index == hold_frames - 1
            ):
                print(
                    f"hold={hold_index + 1:3d}/{hold_frames} "
                    f"pelvis=({float(pelvis[0]):+.3f},"
                    f"{float(pelvis[1]):+.3f},"
                    f"{pelvis_z:+.3f})",
                    flush=True,
                )

            if pelvis_z < 0.45:
                fallen = True
                fall_frame = end
                break

    final_pelvis = robot.data.body_pos_w[
        0,
        pelvis_id,
    ]

    print()
    print("=" * 80)
    print("032 RESULT")
    print("=" * 80)
    print(f"fallen:                  {fallen}")
    print(f"fall frame:              {fall_frame}")
    print(
        "final pelvis:            "
        f"({float(final_pelvis[0]):+.3f}, "
        f"{float(final_pelvis[1]):+.3f}, "
        f"{float(final_pelvis[2]):+.3f})"
    )
    print(
        f"minimum pelvis Z:        "
        f"{lowest_pelvis_z:+.3f} m"
    )
    print(
        f"maximum joint RMSE:      "
        f"{max_joint_rmse:.4f} rad"
    )
    print(
        f"maximum root pos error:  "
        f"{max_root_position_error:.4f} m"
    )
    print("=" * 80)

    env.close()

    return 1 if fallen else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
