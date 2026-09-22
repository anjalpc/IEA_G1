#!/usr/bin/env python3
"""027: dynamically replay the validated stair reference on the IEA G1."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Replay the validated G1 stair reference in Isaac Lab."
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
parser.add_argument("--end-frame", type=int, default=260)
parser.add_argument("--blend-s", type=float, default=1.0)
parser.add_argument("--hold-s", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

# Register the project-owned task.
import simulation.isaac_lab.g1_stairs.register  # noqa: F401

from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS
from simulation.isaac_lab.g1_stairs.reference_motion import (
    load_reference_motion,
)


REPO_ROOT = Path(__file__).resolve().parents[5]


def as_ids(value):
    """Normalize Isaac Lab joint/body ID results to a Python list."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


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
        raise ValueError("end-frame must be >= start-frame")

    print("[027] 1/7 parse_env_cfg", flush=True)
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
    )

    print("[027] 2/7 gym.make", flush=True)
    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None,
    )

    print("[027] 3/7 env created", flush=True)
    print("[027] 4/7 env.reset", flush=True)
    env.reset(seed=args.seed)
    print("[027] 5/7 reset complete", flush=True)

    print("[027] 6/7 resolving robot", flush=True)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    sim = base_env.sim

    print("[027] 7/7 resolving 29 joints", flush=True)
    joint_result = robot.find_joints(
        list(BODY_JOINTS),
        preserve_order=True,
    )

    # Isaac Lab versions return either IDs directly or (IDs, names).
    if isinstance(joint_result, tuple):
        joint_ids = as_ids(joint_result[0])
        resolved_joint_names = list(joint_result[1])
    else:
        joint_ids = as_ids(joint_result)
        resolved_joint_names = [
            robot.joint_names[index] for index in joint_ids
        ]

    if tuple(resolved_joint_names) != tuple(BODY_JOINTS):
        raise RuntimeError(
            "Resolved robot joint order does not match BODY_JOINTS.\n"
            f"expected={BODY_JOINTS}\n"
            f"actual={tuple(resolved_joint_names)}"
        )

    print("=" * 80)
    print("027 REFERENCE PLAYBACK")
    print("=" * 80)
    print(f"task:        {args.task}")
    print(f"reference:   {reference_path}")
    print(
        f"frames:      {args.start_frame}..{args.end_frame}"
    )
    print(f"frequency:   {motion.frequency_hz:.1f} Hz")
    print(f"joints:      {len(joint_ids)}")
    print()

    physics_dt = float(sim.get_physics_dt())
    reference_dt = 1.0 / float(motion.frequency_hz)

    physics_steps_per_reference = max(
        1,
        int(round(reference_dt / physics_dt)),
    )

    print(f"physics dt:  {physics_dt:.6f} s")
    print(f"reference dt:{reference_dt:.6f} s")
    print(
        "physics steps per reference frame:",
        physics_steps_per_reference,
    )

    device = robot.device

    reference_q = torch.as_tensor(
        np.asarray(
            motion.dof_pos[
                args.start_frame : args.end_frame + 1
            ],
            dtype=np.float32,
        ),
        device=device,
    )

    initial_q = robot.data.joint_pos[
        0, joint_ids
    ].clone()

    first_q = reference_q[0]

    blend_frames = max(
        1,
        int(round(args.blend_s * motion.frequency_hz)),
    )

    print()
    print(
        f"Blending initial pose -> reference frame "
        f"{args.start_frame} over {blend_frames} frames"
    )

    def simulate_target(target: torch.Tensor) -> None:
        target_batch = target.unsqueeze(0)

        for _ in range(physics_steps_per_reference):
            robot.set_joint_position_target(
                target_batch,
                joint_ids=joint_ids,
            )

            base_env.scene.write_data_to_sim()
            sim.step()
            base_env.scene.update(physics_dt)

    # Smoothly move from the environment's reset posture into the
    # first reference posture. Root state is never teleported.
    for blend_index in range(blend_frames):
        alpha = float(blend_index + 1) / float(blend_frames)

        target = (
            (1.0 - alpha) * initial_q
            + alpha * first_q
        )

        simulate_target(target)

    print("Initial blend complete.")
    print()
    print("Starting dynamic reference playback...")

    fallen = False
    lowest_pelvis_z = float("inf")
    maximum_pelvis_x = -float("inf")

    body_names = list(robot.body_names)

    try:
        pelvis_id = body_names.index("pelvis")
    except ValueError:
        pelvis_id = 0

    ankle_candidates = {
        "left": (
            "left_ankle_roll_link",
            "left_ankle_pitch_link",
        ),
        "right": (
            "right_ankle_roll_link",
            "right_ankle_pitch_link",
        ),
    }

    ankle_ids = {}

    for side, candidates in ankle_candidates.items():
        for candidate in candidates:
            if candidate in body_names:
                ankle_ids[side] = body_names.index(candidate)
                break

    for local_index, target in enumerate(reference_q):
        frame = args.start_frame + local_index

        simulate_target(target)

        pelvis = robot.data.body_pos_w[
            0, pelvis_id
        ]

        pelvis_x = float(pelvis[0].item())
        pelvis_z = float(pelvis[2].item())

        maximum_pelvis_x = max(
            maximum_pelvis_x,
            pelvis_x,
        )

        lowest_pelvis_z = min(
            lowest_pelvis_z,
            pelvis_z,
        )

        # Conservative fall indication only.
        if pelvis_z < 0.45:
            fallen = True

        if (
            frame == args.start_frame
            or frame % 25 == 0
            or frame == args.end_frame
        ):
            text = (
                f"frame={frame:3d} "
                f"time={frame / motion.frequency_hz:5.2f}s "
                f"pelvis_x={pelvis_x:+.3f} "
                f"pelvis_z={pelvis_z:+.3f}"
            )

            for side in ("left", "right"):
                if side in ankle_ids:
                    p = robot.data.body_pos_w[
                        0, ankle_ids[side]
                    ]

                    text += (
                        f" {side[0].upper()}_ankle="
                        f"({float(p[0]):+.3f},"
                        f"{float(p[2]):+.3f})"
                    )

            print(text)

        if fallen:
            print()
            print(
                f"FALL detected around reference frame {frame}"
            )
            break

    hold_frames = max(
        0,
        int(round(args.hold_s * motion.frequency_hz)),
    )

    if not fallen and hold_frames:
        print()
        print(
            f"Holding final reference pose for "
            f"{args.hold_s:.2f}s"
        )

        final_target = reference_q[-1]

        for _ in range(hold_frames):
            simulate_target(final_target)

            pelvis_z = float(
                robot.data.body_pos_w[
                    0, pelvis_id, 2
                ].item()
            )

            lowest_pelvis_z = min(
                lowest_pelvis_z,
                pelvis_z,
            )

            if pelvis_z < 0.45:
                fallen = True
                break

    final_pelvis = robot.data.body_pos_w[
        0, pelvis_id
    ]

    final_x = float(final_pelvis[0].item())
    final_z = float(final_pelvis[2].item())

    print()
    print("=" * 80)
    print("027 RESULT")
    print("=" * 80)
    print(f"fallen:            {fallen}")
    print(f"final pelvis X:    {final_x:+.3f} m")
    print(f"final pelvis Z:    {final_z:+.3f} m")
    print(
        f"maximum pelvis X:  {maximum_pelvis_x:+.3f} m"
    )
    print(
        f"minimum pelvis Z:  {lowest_pelvis_z:+.3f} m"
    )
    print("=" * 80)

    env.close()

    return 1 if fallen else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
