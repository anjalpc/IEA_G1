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

parser.add_argument(
    "--unitree-vx",
    type=float,
    default=0.40,
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
        render_mode="rgb_array",
    )

    video_dir = (
        "/workspace/IEA_G1/port/pretrained_controller/"
        "hybrid_task/outputs/videos"
    )

    Path(video_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,
        video_length=args.max_steps,
        name_prefix=(
            f"hybrid_vx_{args.unitree_vx:.2f}"
        ),
        disable_logger=True,
    )

    print(
        f"[VIDEO] recording to {video_dir}",
        flush=True,
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

        max_reference_frame = q_ref.shape[0] - 1

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

        # ===========================================================
        # HYBRID CONTROLLER
        #
        # Phase A:
        #   frozen pretrained stair controller
        #
        # Phase B:
        #   Unitree native 29-DoF velocity ONNX controller
        #
        # Everything in this experiment remains inference-only.
        # ===========================================================

        import onnxruntime as ort

        unitree_policy_path = (
            "/workspace/unitree_rl_lab/deploy/robots/"
            "g1_29dof/config/policy/velocity/v0/"
            "exported/policy.onnx"
        )

        print(
            f"[HYBRID] loading Unitree velocity policy: "
            f"{unitree_policy_path}",
            flush=True,
        )

        unitree_session = ort.InferenceSession(
            unitree_policy_path,
            providers=["CPUExecutionProvider"],
        )

        unitree_input_name = (
            unitree_session.get_inputs()[0].name
        )
        unitree_output_name = (
            unitree_session.get_outputs()[0].name
        )

        if unitree_session.get_inputs()[0].shape != [1, 480]:
            raise RuntimeError(
                "Unexpected Unitree ONNX input shape: "
                f"{unitree_session.get_inputs()[0].shape}"
            )

        if unitree_session.get_outputs()[0].shape != [1, 29]:
            raise RuntimeError(
                "Unexpected Unitree ONNX output shape: "
                f"{unitree_session.get_outputs()[0].shape}"
            )

        # Exact policy/articulation joint order.
        unitree_joint_order = (
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
        )

        actual_joint_order = tuple(robot.joint_names)

        if actual_joint_order != unitree_joint_order:
            print("[HYBRID] expected joint order:")
            for i, name in enumerate(unitree_joint_order):
                print(f"  {i:02d}: {name}")

            print("[HYBRID] actual joint order:")
            for i, name in enumerate(actual_joint_order):
                print(f"  {i:02d}: {name}")

            raise RuntimeError(
                "Unitree ONNX joint order does not match "
                "the active IEA articulation"
            )

        # Exact deploy.yaml default pose, in policy order.
        unitree_q_default = torch.tensor(
            [
                -0.1, -0.1, 0.0,
                 0.0,  0.0, 0.0,
                 0.0,  0.0, 0.0,
                 0.3,  0.3,
                 0.3,  0.3,
                -0.2, -0.2,
                 0.25, -0.25,
                 0.0,  0.0,
                 0.0,  0.0,
                 0.97, 0.97,
                 0.15, -0.15,
                 0.0,  0.0,
                 0.0,  0.0,
            ],
            dtype=torch.float32,
            device=device,
        )

        # Exact Unitree deployment PD gains.
        unitree_kp = torch.tensor(
            [
                100.0, 100.0, 200.0,
                100.0, 100.0, 40.0,
                100.0, 100.0, 40.0,
                150.0, 150.0,
                40.0, 40.0,
                40.0, 40.0,
                40.0, 40.0,
                40.0, 40.0,
                40.0, 40.0,
                40.0, 40.0,
                40.0, 40.0,
                40.0, 40.0,
                40.0, 40.0,
            ],
            dtype=torch.float32,
            device=device,
        )

        unitree_kd = torch.tensor(
            [
                2.0, 2.0, 5.0,
                2.0, 2.0, 10.0,
                2.0, 2.0, 10.0,
                4.0, 4.0,
                10.0, 10.0,
                2.0, 2.0,
                10.0, 10.0,
                2.0, 2.0,
                10.0, 10.0,
                10.0, 10.0,
                10.0, 10.0,
                10.0, 10.0,
                10.0, 10.0,
            ],
            dtype=torch.float32,
            device=device,
        )

        # The current environment was created with the model-12
        # action contract.  env.step() therefore still expects
        # model-12 normalized actions.
        #
        # We convert Unitree's desired joint target back through
        # that existing action transform so env.step() produces
        # exactly:
        #
        # q_target = unitree_q_default + 0.25 * unitree_raw_action
        #
        K5020 = 14.2506231
        K7520_14 = 40.1792385
        K7520_22 = 99.0984278
        K4010 = 16.7783275

        model12_scale_values = []

        for name in robot.joint_names:
            if (
                "wrist_pitch_joint" in name
                or "wrist_yaw_joint" in name
            ):
                value = 0.25 * 5.0 / K4010

            elif (
                "hip_yaw_joint" in name
                or name == "waist_yaw_joint"
            ):
                value = 0.25 * 88.0 / K7520_14

            elif (
                "hip_pitch_joint" in name
                or "hip_roll_joint" in name
                or "knee_joint" in name
            ):
                value = 0.25 * 139.0 / K7520_22

            else:
                # ankles, waist roll/pitch, shoulders,
                # elbows and wrist roll
                value = 0.25 * 25.0 / K5020

                if (
                    "ankle_" in name
                    or name == "waist_roll_joint"
                    or name == "waist_pitch_joint"
                ):
                    value = 0.25 * 50.0 / (2.0 * K5020)

            model12_scale_values.append(value)

        model12_action_scale = torch.tensor(
            model12_scale_values,
            dtype=torch.float32,
            device=device,
        )

        if model12_action_scale.shape != (29,):
            raise RuntimeError(
                "Model-12 action-scale reconstruction failed"
            )

        # Fixed flat-ground continuation command.
        unitree_command = torch.tensor(
            [args.unitree_vx, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        )

        # Handover gate:
        #
        # platform begins at x=-0.125.
        # Previous frozen-controller run remained healthy around
        # x=+0.05 before the synthetic reference transition failed.
        unitree_switch_x = 0.05
        unitree_switch_min_z = 1.15
        unitree_switch_max_tilt = 0.40

        unitree_active = False
        unitree_first_step = False
        unitree_switch_step = None

        unitree_ang_hist = None
        unitree_gravity_hist = None
        unitree_command_hist = None
        unitree_q_hist = None
        unitree_dq_hist = None
        unitree_action_hist = None

        unitree_raw_action = torch.zeros(
            (1, 29),
            dtype=torch.float32,
            device=device,
        )

        print(
            "[HYBRID] Unitree contract ready: "
            "obs=480 action=29 command=[0.40,0,0]",
            flush=True,
        )

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

                # ===================================================
                # HYBRID HANDOVER / UNITREE VELOCITY PHASE
                # ===================================================

                live_root_pos = robot.data.root_pos_w[0]
                live_root_quat = robot.data.root_quat_w[0]

                live_gravity = quat_apply(
                    quat_inv(live_root_quat.unsqueeze(0)),
                    down.unsqueeze(0),
                )[0]

                live_tilt = torch.acos(
                    torch.clamp(
                        -live_gravity[2],
                        -1.0,
                        1.0,
                    )
                ).item()

                if (
                    not unitree_active
                    and live_root_pos[0].item() >= unitree_switch_x
                    and live_root_pos[2].item() >= unitree_switch_min_z
                    and live_tilt <= unitree_switch_max_tilt
                ):
                    unitree_active = True
                    unitree_first_step = True
                    unitree_switch_step = step

                    # Switch PhysX joint gains to the exact Unitree
                    # velocity-policy deployment gains.
                    robot.write_joint_stiffness_to_sim(
                        unitree_kp.unsqueeze(0)
                    )
                    robot.write_joint_damping_to_sim(
                        unitree_kd.unsqueeze(0)
                    )

                    print()
                    print(
                        "[HYBRID] ========================================",
                        flush=True,
                    )
                    print(
                        "[HYBRID] SWITCH GRAIL -> UNITREE VELOCITY",
                        flush=True,
                    )
                    print(
                        f"[HYBRID] step={step + 1} "
                        f"x={live_root_pos[0].item():+.3f} "
                        f"z={live_root_pos[2].item():+.3f} "
                        f"tilt={live_tilt:.3f}",
                        flush=True,
                    )
                    print(
                        f"[HYBRID] command vx={args.unitree_vx:+.2f} m/s",
                        flush=True,
                    )
                    print(
                        "[HYBRID] ========================================",
                        flush=True,
                    )
                    print()

                if unitree_active:
                    unitree_ang = (
                        robot.data.root_ang_vel_b[0] * 0.2
                    )

                    unitree_gravity = live_gravity

                    unitree_q_rel = (
                        robot.data.joint_pos[0]
                        - unitree_q_default
                    )

                    unitree_dq = (
                        robot.data.joint_vel[0] * 0.05
                    )

                    if unitree_first_step:
                        # Native reset semantics:
                        # initialize all five history slots from the
                        # current live handover state.
                        unitree_ang_hist = (
                            unitree_ang.unsqueeze(0).repeat(5, 1)
                        )

                        unitree_gravity_hist = (
                            unitree_gravity.unsqueeze(0).repeat(5, 1)
                        )

                        unitree_command_hist = (
                            unitree_command.unsqueeze(0).repeat(5, 1)
                        )

                        unitree_q_hist = (
                            unitree_q_rel.unsqueeze(0).repeat(5, 1)
                        )

                        unitree_dq_hist = (
                            unitree_dq.unsqueeze(0).repeat(5, 1)
                        )

                        unitree_action_hist = torch.zeros(
                            (5, 29),
                            dtype=torch.float32,
                            device=device,
                        )

                        unitree_first_step = False

                    else:
                        unitree_ang_hist = append_history(
                            unitree_ang_hist,
                            unitree_ang,
                        )

                        unitree_gravity_hist = append_history(
                            unitree_gravity_hist,
                            unitree_gravity,
                        )

                        unitree_command_hist = append_history(
                            unitree_command_hist,
                            unitree_command,
                        )

                        unitree_q_hist = append_history(
                            unitree_q_hist,
                            unitree_q_rel,
                        )

                        unitree_dq_hist = append_history(
                            unitree_dq_hist,
                            unitree_dq,
                        )

                    # Exact Unitree observation term order.
                    # History is flattened within each term.
                    unitree_obs = torch.cat(
                        (
                            unitree_ang_hist.reshape(-1),
                            unitree_gravity_hist.reshape(-1),
                            unitree_command_hist.reshape(-1),
                            unitree_q_hist.reshape(-1),
                            unitree_dq_hist.reshape(-1),
                            unitree_action_hist.reshape(-1),
                        ),
                        dim=0,
                    )

                    if unitree_obs.numel() != 480:
                        raise RuntimeError(
                            f"Unitree obs dimension="
                            f"{unitree_obs.numel()}"
                        )

                    unitree_obs_np = (
                        unitree_obs
                        .unsqueeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype("float32")
                    )

                    unitree_action_np = unitree_session.run(
                        [unitree_output_name],
                        {
                            unitree_input_name:
                            unitree_obs_np
                        },
                    )[0]

                    unitree_raw_action = torch.from_numpy(
                        unitree_action_np
                    ).to(
                        device=device,
                        dtype=torch.float32,
                    )

                    if unitree_raw_action.shape != (1, 29):
                        raise RuntimeError(
                            "Unexpected Unitree action shape: "
                            f"{tuple(unitree_raw_action.shape)}"
                        )

                    if not torch.isfinite(
                        unitree_raw_action
                    ).all():
                        raise RuntimeError(
                            "Non-finite Unitree ONNX action"
                        )

                    # Exact Unitree deploy action decoder.
                    unitree_q_target = (
                        unitree_q_default
                        + 0.25 * unitree_raw_action[0]
                    )

                    # Convert desired q-target into the existing
                    # model-12 action-manager coordinate system.
                    action = (
                        (
                            unitree_q_target
                            - q_default
                        )
                        / model12_action_scale
                    ).unsqueeze(0)

                    if (
                        step == unitree_switch_step
                        or (step + 1) % 10 == 0
                    ):
                        print(
                            "[HYBRID] "
                            f"step={step + 1:03d} "
                            f"raw_rms="
                            f"{torch.sqrt(torch.mean(unitree_raw_action**2)).item():.3f} "
                            f"raw_max="
                            f"{torch.max(torch.abs(unitree_raw_action)).item():.3f}",
                            flush=True,
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

                if unitree_active:
                    unitree_action_hist = append_history(
                        unitree_action_hist,
                        unitree_raw_action[0],
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

                if unitree_active and x >= 1.875:
                    print()
                    print(
                        "[HYBRID] CONTINUATION_2M_REACHED",
                        flush=True,
                    )
                    print(
                        "[HYBRID] TASK_SUCCESS",
                        flush=True,
                    )
                    print(
                        f"[HYBRID] final_x={x:+.3f} "
                        f"switch_step={unitree_switch_step + 1}",
                        flush=True,
                    )
                    break

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
