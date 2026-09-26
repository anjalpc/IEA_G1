#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    default="Unitree-IEA-G1-29dof-Stairs-v0",
)

parser.add_argument(
    "--reference",
    default=(
        "experiments/g1_stairs/reference_data/"
        "stair_ascent_reference_50hz.npz"
    ),
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=247,
)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

print("[CLOSED] 01 launching Isaac", flush=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def quat_inv(q):
    out = q.clone()
    out[..., 1:] *= -1.0
    return out / torch.sum(
        q * q,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-12)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(q, v):
    xyz = q[..., 1:]
    w = q[..., :1]

    t = 2.0 * torch.cross(
        xyz,
        v,
        dim=-1,
    )

    return (
        v
        + w * t
        + torch.cross(
            xyz,
            t,
            dim=-1,
        )
    )


def quat_to_matrix(q):
    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-12)

    w, x, y, z = q.unbind(-1)

    return torch.stack(
        (
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w),

            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w),

            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rot6d(q):
    return quat_to_matrix(q)[..., :2].reshape(
        q.shape[:-1] + (6,)
    )


def append_history(history, value):
    return torch.cat(
        (
            history[1:],
            value.unsqueeze(0),
        ),
        dim=0,
    )


def source_to_task_position(p):
    out = p.clone()

    out[..., 0] = -p[..., 1]
    out[..., 1] = p[..., 0]
    out[..., 2] = p[..., 2]

    return out


def source_to_task_quaternion(q):
    q_turn = torch.tensor(
        [
            math.sqrt(0.5),
            0.0,
            0.0,
            math.sqrt(0.5),
        ],
        device=q.device,
        dtype=q.dtype,
    )

    return quat_mul(
        q_turn.expand_as(q),
        q,
    )


def angular_velocity_from_quaternions(
    q0,
    q1,
    dt,
):
    dq = quat_mul(
        q1,
        quat_inv(q0),
    )

    if dq[0] < 0:
        dq = -dq

    xyz = dq[1:]
    xyz_norm = torch.linalg.vector_norm(xyz)

    if xyz_norm < 1.0e-8:
        return torch.zeros(
            3,
            device=q0.device,
            dtype=q0.dtype,
        )

    angle = 2.0 * torch.atan2(
        xyz_norm,
        dq[0].clamp_min(1.0e-8),
    )

    axis = xyz / xyz_norm

    return axis * angle / dt


def main():
    import gymnasium as gym
    import numpy as np

    from isaaclab_tasks.utils import parse_env_cfg

    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from simulation.isaac_lab.g1_stairs.controller_port.model12_contract import (
        apply_model12_contract,
    )

    from simulation.isaac_lab.g1_stairs.controller_port.external_policy_adapter import (
        load_reference_policy,
    )

    from simulation.isaac_lab.g1_stairs.controller_port.reference_policy_contract import (
        build_static_terrain_contract,
    )

    reference_path = Path(args.reference)

    if not reference_path.is_absolute():
        reference_path = (
            Path("/workspace/IEA_G1")
            / reference_path
        )

    data = np.load(reference_path)

    device = args.device

    q_ref = torch.tensor(
        data["dof_pos"],
        dtype=torch.float32,
        device=device,
    )

    dq_ref = torch.tensor(
        data["dof_vel"],
        dtype=torch.float32,
        device=device,
    )

    body_pos_source = torch.tensor(
        data["body_pos_w"],
        dtype=torch.float32,
        device=device,
    )

    body_quat_source = torch.tensor(
        data["body_quat_w"],
        dtype=torch.float32,
        device=device,
    )

    body_pos_task = source_to_task_position(
        body_pos_source
    )

    body_quat_task = source_to_task_quaternion(
        body_quat_source
    )

    print(
        f"[CLOSED] reference frames={q_ref.shape[0]}",
        flush=True,
    )

    print(
        "[CLOSED] loading frozen pretrained actor",
        flush=True,
    )

    policy = load_reference_policy(
        device=device,
    )

    print(
        "[CLOSED] actor loaded and frozen",
        flush=True,
    )

    env_cfg = parse_env_cfg(
        args.task,
        device=device,
        num_envs=1,
    )

    env_cfg.seed = 42

    apply_model12_contract(
        env_cfg
    )

    # Qualification must observe the physical trajectory rather than
    # auto-reset immediately after a fall.
    if hasattr(env_cfg.terminations, "base_height"):
        env_cfg.terminations.base_height = None

    if hasattr(env_cfg.terminations, "bad_orientation"):
        env_cfg.terminations.bad_orientation = None

    env_cfg.episode_length_s = max(
        10.0,
        args.max_steps * 0.02 + 1.0,
    )

    terrain_generator = (
        env_cfg.scene.terrain.terrain_generator
    )

    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    print(
        "[CLOSED] creating IEA environment",
        flush=True,
    )

    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None,
    )

    try:
        env.reset(seed=42)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        if robot.num_joints != 29:
            raise RuntimeError(
                f"Expected 29 joints, got {robot.num_joints}"
            )

        # ---------------------------------------------------------------
        # Native controller reset contract:
        # initialize from reference motion, not static model default.
        # ---------------------------------------------------------------

        desired_start_x = (
            robot.data.root_pos_w[0, 0]
            .clone()
        )

        root_pos_0 = torch.tensor(
            [
                desired_start_x.item(),
                0.0,
                body_pos_task[0, 0, 2].item(),
            ],
            dtype=torch.float32,
            device=device,
        )

        root_quat_0 = body_quat_task[
            0,
            0,
        ].clone()

        dt = 0.02

        raw_lin_vel_0 = (
            body_pos_task[1, 0]
            - body_pos_task[0, 0]
        ) / dt

        root_lin_vel_0 = raw_lin_vel_0.clone()

        root_ang_vel_0 = (
            angular_velocity_from_quaternions(
                body_quat_task[0, 0],
                body_quat_task[1, 0],
                dt,
            )
        )

        root_state = torch.cat(
            (
                root_pos_0,
                root_quat_0,
                root_lin_vel_0,
                root_ang_vel_0,
            ),
            dim=0,
        ).unsqueeze(0)

        robot.write_joint_state_to_sim(
            q_ref[0].unsqueeze(0),
            dq_ref[0].unsqueeze(0),
        )

        robot.write_root_state_to_sim(
            root_state
        )

        base_env.sim.forward()

        stair_origin_w = torch.tensor(
            [
                root_pos_0[0] + 1.0,
                0.0,
                0.0,
            ],
            device=device,
            dtype=torch.float32,
        ).reshape(1, 3)

        print(
            "[CLOSED] aligned reference reset",
            flush=True,
        )

        print(
            f"[CLOSED] root_pos={robot.data.root_pos_w[0].tolist()}",
            flush=True,
        )

        print(
            f"[CLOSED] root_quat={robot.data.root_quat_w[0].tolist()}",
            flush=True,
        )

        print(
            f"[CLOSED] stair_origin={stair_origin_w[0].tolist()}",
            flush=True,
        )

        # ---------------------------------------------------------------
        # Isaac Lab history semantics:
        # oldest -> newest.
        # First push fills all history slots with first observation.
        # ---------------------------------------------------------------

        q_default = robot.data.default_joint_pos[0]
        dq_default = robot.data.default_joint_vel[0]

        root_quat = robot.data.root_quat_w[0]

        down = torch.tensor(
            [0.0, 0.0, -1.0],
            device=device,
        )

        gravity = quat_apply(
            quat_inv(root_quat.unsqueeze(0)),
            down.unsqueeze(0),
        )[0]

        ang_vel = robot.data.root_ang_vel_b[0]
        q_rel = robot.data.joint_pos[0] - q_default
        dq_rel = robot.data.joint_vel[0] - dq_default

        zero_action = torch.zeros(
            29,
            device=device,
        )

        gravity_hist = gravity.unsqueeze(0).repeat(10, 1)
        ang_vel_hist = ang_vel.unsqueeze(0).repeat(10, 1)
        q_hist = q_rel.unsqueeze(0).repeat(10, 1)
        dq_hist = dq_rel.unsqueeze(0).repeat(10, 1)
        action_hist = zero_action.unsqueeze(0).repeat(10, 1)

        encoder_index = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        )

        max_reference_frame = min(
            246,
            q_ref.shape[0] - 1,
        )

        start_x = robot.data.root_pos_w[
            0,
            0,
        ].item()

        max_x = start_x
        max_z = robot.data.root_pos_w[
            0,
            2,
        ].item()

        print()
        print("=" * 80, flush=True)
        print(
            "FROZEN PRETRAINED CONTROLLER CLOSED LOOP",
            flush=True,
        )
        print("=" * 80, flush=True)

        capture = []

        with torch.inference_mode():

            for step in range(args.max_steps):

                if step > 0:
                    root_quat = robot.data.root_quat_w[0]

                    gravity = quat_apply(
                        quat_inv(root_quat.unsqueeze(0)),
                        down.unsqueeze(0),
                    )[0]

                    ang_vel = robot.data.root_ang_vel_b[0]

                    q_rel = (
                        robot.data.joint_pos[0]
                        - q_default
                    )

                    dq_rel = (
                        robot.data.joint_vel[0]
                        - dq_default
                    )

                    gravity_hist = append_history(
                        gravity_hist,
                        gravity,
                    )

                    ang_vel_hist = append_history(
                        ang_vel_hist,
                        ang_vel,
                    )

                    q_hist = append_history(
                        q_hist,
                        q_rel,
                    )

                    dq_hist = append_history(
                        dq_hist,
                        dq_rel,
                    )

                terrain = build_static_terrain_contract(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    stair_origin_w,
                )

                # Exact instantiated native policy observation order:
                #
                # 0 base_ang_vel                     30
                # 1 joint_pos                       290
                # 2 joint_vel                       290
                # 3 actions                         290
                # 4 gravity_dir                      30
                # 5 object_pos_delta_multi_future    30
                # 6 object_ori_delta_multi_future_6d 60
                # 7 object_pos_b                      3
                # 8 object_ori_b_6d                   6
                #                                    ----
                #                                    1029
                actor_obs = torch.cat(
                    (
                        ang_vel_hist.reshape(-1),
                        q_hist.reshape(-1),
                        dq_hist.reshape(-1),
                        action_hist.reshape(-1),
                        gravity_hist.reshape(-1),

                        terrain[
                            "object_pos_delta_multi_future"
                        ][0],

                        terrain[
                            "object_ori_delta_multi_future_6d"
                        ][0],

                        terrain[
                            "object_pos_b"
                        ][0],

                        terrain[
                            "object_ori_b_6d"
                        ][0],
                    ),
                    dim=0,
                )

                if actor_obs.numel() != 1029:
                    raise RuntimeError(
                        f"actor_obs={actor_obs.numel()}"
                    )

                future_ids = (
                    torch.arange(
                        10,
                        device=device,
                        dtype=torch.long,
                    )
                    * 5
                    + step
                )

                future_ids = torch.clamp(
                    future_ids,
                    max=max_reference_frame,
                )

                # Exact released command_multi_future semantics:
                #
                # 1. flatten all future joint positions
                # 2. flatten all future joint velocities
                # 3. concatenate those two 290-D blocks
                # 4. observation term reshapes the resulting 580-D vector
                #    into (10, 58)
                #
                # This is intentionally NOT [q_t, dq_t] per future frame.
                command_future = torch.cat(
                    (
                        q_ref[future_ids].reshape(-1),
                        dq_ref[future_ids].reshape(-1),
                    ),
                    dim=0,
                ).reshape(10, 58)

                command_z = body_pos_task[
                    future_ids,
                    0,
                    2,
                ].reshape(10, 1)

                robot_quat_future = (
                    robot.data.root_quat_w[0]
                    .reshape(1, 4)
                    .expand(10, -1)
                )

                ref_root_quat = body_quat_task[
                    future_ids,
                    0,
                ]

                rel_ref_quat = quat_mul(
                    quat_inv(
                        robot_quat_future
                    ),
                    ref_root_quat,
                )

                anchor_ori_future = rot6d(
                    rel_ref_quat
                )

                tokenizer = torch.cat(
                    (
                        encoder_index,
                        command_future.reshape(-1),
                        command_z.reshape(-1),
                        anchor_ori_future.reshape(-1),
                        terrain[
                            "height_map_z_flat"
                        ][0],
                    ),
                    dim=0,
                )

                if tokenizer.numel() != 774:
                    raise RuntimeError(
                        f"tokenizer={tokenizer.numel()}"
                    )

                obs_dict = {
                    "actor_obs": actor_obs.unsqueeze(0),
                    "tokenizer": tokenizer.unsqueeze(0),
                }

                # Exact released evaluation behavior:
                # reset the Actor's internal rollout buffer before EVERY
                # control-step forward pass.
                #
                # The 10-frame proprio/action history is already explicitly
                # encoded inside actor_obs, so we must not accumulate another
                # temporal history inside Actor.rollout().
                policy.init_rollout()

                actor_state = policy.rollout(
                    obs_dict=obs_dict,
                )

                action = policy.action_mean.detach()

                if len(capture) < 30:
                    capture.append(
                        {
                            "step": step,
                            "actor_obs": actor_obs.detach().cpu(),
                            "tokenizer": tokenizer.detach().cpu(),
                            "action_mean": action[0].detach().cpu(),
                        }
                    )

                    torch.save(
                        capture,
                        "/workspace/IEA_G1/"
                        "experiments/g1_stairs/"
                        "iea_controller_capture.pt",
                    )

                if action.shape != (1, 29):
                    raise RuntimeError(
                        f"action shape={tuple(action.shape)}"
                    )

                if not torch.isfinite(action).all():
                    raise RuntimeError(
                        "non-finite policy action"
                    )

                # Exact released evaluation path uses deterministic action_mean.
                env.step(action)

                action_hist = append_history(
                    action_hist,
                    action[0],
                )

                root_pos = robot.data.root_pos_w[0]
                root_quat_now = robot.data.root_quat_w[0]

                gravity_now = quat_apply(
                    quat_inv(
                        root_quat_now.unsqueeze(0)
                    ),
                    down.unsqueeze(0),
                )[0]

                tilt = torch.acos(
                    torch.clamp(
                        -gravity_now[2],
                        -1.0,
                        1.0,
                    )
                ).item()

                x = root_pos[0].item()
                z = root_pos[2].item()

                max_x = max(max_x, x)
                max_z = max(max_z, z)

                if (
                    step == 0
                    or (step + 1) % 10 == 0
                    or step == args.max_steps - 1
                ):
                    hmap = terrain[
                        "height_map_z_flat"
                    ]

                    print(
                        "[CLOSED] "
                        f"step={step + 1:03d} "
                        f"t={(step + 1)*0.02:.2f} "
                        f"x={x:+.3f} "
                        f"z={z:+.3f} "
                        f"tilt={tilt:.3f} "
                        f"a_rms="
                        f"{torch.sqrt(torch.mean(action*action)).item():.3f} "
                        f"a_max="
                        f"{torch.max(torch.abs(action)).item():.3f} "
                        f"hmap=["
                        f"{hmap.min().item():+.3f},"
                        f"{hmap.max().item():+.3f}]",
                        flush=True,
                    )

                # Stop only after unmistakable physical collapse.
                if z < 0.35 or tilt > 1.35:
                    print(
                        "[CLOSED] physical collapse threshold reached",
                        flush=True,
                    )
                    break

        progress = max_x - start_x

        print()
        print("=" * 80, flush=True)
        print("CLOSED-LOOP RESULT", flush=True)
        print("=" * 80, flush=True)

        print(
            f"start_x={start_x:.6f}",
            flush=True,
        )

        print(
            f"max_x={max_x:.6f}",
            flush=True,
        )

        print(
            f"forward_progress={progress:.6f}",
            flush=True,
        )

        print(
            f"max_root_z={max_z:.6f}",
            flush=True,
        )

        print(
            "CHECKPOINT WEIGHTS REMAINED FROZEN",
            flush=True,
        )

        print(
            "NO PPO / NO BACKPROP / NO TRAINING",
            flush=True,
        )

    finally:
        env.close()


try:
    main()

except BaseException:
    print()
    print("[CLOSED] FAILURE", flush=True)
    traceback.print_exc()
    raise

finally:
    simulation_app.close()
