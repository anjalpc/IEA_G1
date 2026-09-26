#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback
import faulthandler
import torch

faulthandler.enable()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    default="Unitree-IEA-G1-29dof-Stairs-v0",
)

parser.add_argument(
    "--reference",
    default=(
        "experiments/g1_stairs/"
        "reference_data/stair_ascent_reference_50hz.npz"
    ),
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[OBS] 01 launching Isaac", flush=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def quat_inv(q):
    out = q.clone()
    out[..., 1:] *= -1.0
    return out / torch.sum(q * q, dim=-1, keepdim=True)


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
    q_xyz = q[..., 1:]
    qw = q[..., :1]

    t = 2.0 * torch.cross(q_xyz, v, dim=-1)

    return v + qw * t + torch.cross(q_xyz, t, dim=-1)


def quat_to_matrix(q):
    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)

    w, x, y, z = q.unbind(-1)

    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),

            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),

            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rot6d(q):
    mat = quat_to_matrix(q)

    return mat[..., :2].reshape(
        q.shape[:-1] + (6,)
    )


def resolve_action_cfg(env_cfg):
    for name, value in vars(env_cfg.actions).items():
        if value is None:
            continue

        class_type = getattr(value, "class_type", None)

        if getattr(class_type, "__name__", "") == "JointPositionAction":
            return name, value

    raise RuntimeError(
        "JointPositionAction configuration not found"
    )


def apply_model12_contract(env_cfg):
    from isaaclab.actuators import ImplicitActuatorCfg

    K5020 = 14.2506231
    D5020 = 0.90722284
    A5020 = 0.003609725

    K7520_14 = 40.1792385
    D7520_14 = 2.55788977
    A7520_14 = 0.010177520

    K7520_22 = 99.0984278
    D7520_22 = 6.30880185
    A7520_22 = 0.025101925

    K4010 = 16.7783275
    D4010 = 1.0681415
    A4010 = 0.00425

    env_cfg.scene.robot.init_state.joint_pos = {
        ".*_hip_pitch_joint": -0.312,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        ".*_shoulder_yaw_joint": 0.0,
        ".*_elbow_joint": 0.6,
        ".*_wrist_roll_joint": 0.0,
        ".*_wrist_pitch_joint": 0.0,
        ".*_wrist_yaw_joint": 0.0,
    }

    env_cfg.scene.robot.init_state.joint_vel = {
        ".*": 0.0,
    }

    env_cfg.scene.robot.actuators = {
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=K7520_14,
            damping=D7520_14,
            armature=A7520_14,
        ),
        "hip_pitch_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=K7520_22,
            damping=D7520_22,
            armature=A7520_22,
        ),
        "feet_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * K5020,
            damping=2.0 * D5020,
            armature=2.0 * A5020,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=K5020,
            damping=D5020,
            armature=A5020,
        ),
        "wrist_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=5.0,
            velocity_limit_sim=22.0,
            stiffness=K4010,
            damping=D4010,
            armature=A4010,
        ),
    }

    _, action_cfg = resolve_action_cfg(env_cfg)

    action_cfg.use_default_offset = True

    action_cfg.scale = {
        ".*_hip_pitch_joint": 0.25 * 139.0 / K7520_22,
        ".*_hip_roll_joint": 0.25 * 139.0 / K7520_22,
        ".*_hip_yaw_joint": 0.25 * 88.0 / K7520_14,
        ".*_knee_joint": 0.25 * 139.0 / K7520_22,

        ".*_ankle_pitch_joint":
            0.25 * 50.0 / (2.0 * K5020),

        ".*_ankle_roll_joint":
            0.25 * 50.0 / (2.0 * K5020),

        "waist_yaw_joint":
            0.25 * 88.0 / K7520_14,

        "waist_roll_joint":
            0.25 * 50.0 / (2.0 * K5020),

        "waist_pitch_joint":
            0.25 * 50.0 / (2.0 * K5020),

        ".*_shoulder_pitch_joint":
            0.25 * 25.0 / K5020,

        ".*_shoulder_roll_joint":
            0.25 * 25.0 / K5020,

        ".*_shoulder_yaw_joint":
            0.25 * 25.0 / K5020,

        ".*_elbow_joint":
            0.25 * 25.0 / K5020,

        ".*_wrist_roll_joint":
            0.25 * 25.0 / K5020,

        ".*_wrist_pitch_joint":
            0.25 * 5.0 / K4010,

        ".*_wrist_yaw_joint":
            0.25 * 5.0 / K4010,
    }


def main():
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from simulation.isaac_lab.g1_stairs.controller_port.reference_policy_contract import (
        build_static_terrain_contract,
    )

    reference_path = Path(args.reference)

    if not reference_path.is_absolute():
        reference_path = (
            Path("/workspace/IEA_G1") / reference_path
        )

    print(
        f"[OBS] 02 reference={reference_path}",
        flush=True,
    )

    data = np.load(reference_path)

    q_ref = torch.tensor(
        data["dof_pos"],
        dtype=torch.float32,
        device=args.device,
    )

    dq_ref = torch.tensor(
        data["dof_vel"],
        dtype=torch.float32,
        device=args.device,
    )

    body_pos = torch.tensor(
        data["body_pos_w"],
        dtype=torch.float32,
        device=args.device,
    )

    body_quat = torch.tensor(
        data["body_quat_w"],
        dtype=torch.float32,
        device=args.device,
    )

    print(
        f"[OBS] 03 reference frames={q_ref.shape[0]}",
        flush=True,
    )

    print("[OBS] 04 before parse_env_cfg", flush=True)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
    )

    print("[OBS] 05 after parse_env_cfg", flush=True)

    env_cfg.seed = 42

    print("[OBS] 06 before model-12 contract", flush=True)

    apply_model12_contract(env_cfg)

    print("[OBS] 07 after model-12 contract", flush=True)

    terrain_generator = (
        env_cfg.scene.terrain.terrain_generator
    )
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    print("[OBS] 08 before gym.make", flush=True)

    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None,
    )

    print("[OBS] 09 after gym.make", flush=True)

    try:
        print("[OBS] 10 before env.reset", flush=True)

        env.reset(seed=42)

        print("[OBS] 11 after env.reset", flush=True)

        print("[OBS] 12 before env.unwrapped", flush=True)

        base_env = env.unwrapped

        print("[OBS] 13 after env.unwrapped", flush=True)

        robot = base_env.scene["robot"]

        print("[OBS] 14 robot resolved", flush=True)

        device = robot.device

        print(f"[OBS] 15 robot.device={device}", flush=True)

        joint_count = robot.num_joints

        print(f"[OBS] 16 robot.num_joints={joint_count}", flush=True)

        if joint_count != 29:
            raise RuntimeError(
                f"Expected 29 joints, got {joint_count}"
            )

        print("[OBS] 17 before live joint state read", flush=True)

        #
        # Exact actor history order from released terrain policy:
        #
        # gravity_dir
        # base_ang_vel
        # joint_pos
        # joint_vel
        # actions
        #
        # 10 frames each.
        #

        q = robot.data.joint_pos[0]
        print("[OBS] 18 joint_pos read", flush=True)

        dq = robot.data.joint_vel[0]
        print("[OBS] 19 joint_vel read", flush=True)

        q_default = robot.data.default_joint_pos[0]
        print("[OBS] 20 default_joint_pos read", flush=True)

        dq_default = robot.data.default_joint_vel[0]
        print("[OBS] 21 default_joint_vel read", flush=True)

        q_rel = q - q_default
        dq_rel = dq - dq_default

        print("[OBS] 22 joint relative state built", flush=True)

        root_quat = robot.data.root_quat_w[0]

        print(
            f"[OBS] 23 root_quat={root_quat.tolist()}",
            flush=True,
        )

        down = torch.tensor(
            [0.0, 0.0, -1.0],
            device=device,
        )

        print("[OBS] 24 before gravity transform", flush=True)

        gravity = quat_apply(
            quat_inv(root_quat.unsqueeze(0)),
            down.unsqueeze(0),
        )[0]

        print(
            f"[OBS] 25 gravity={gravity.tolist()}",
            flush=True,
        )

        #
        # Isaac root_ang_vel_b is already body-frame angular velocity.
        #
        base_ang_vel = robot.data.root_ang_vel_b[0]

        print(
            f"[OBS] 26 base_ang_vel={base_ang_vel.tolist()}",
            flush=True,
        )

        previous_action = torch.zeros(
            29,
            device=device,
        )

        gravity_hist = gravity.repeat(10)
        ang_vel_hist = base_ang_vel.repeat(10)
        q_hist = q_rel.repeat(10)
        dq_hist = dq_rel.repeat(10)
        action_hist = previous_action.repeat(10)

        initial_root_pos_w = robot.data.root_pos_w.clone()

        stair_origin_w = initial_root_pos_w.clone()
        stair_origin_w[:, 0] += 1.0
        stair_origin_w[:, 2] = 0.0

        print(
            f"[OBS] initial_root_pos_w={initial_root_pos_w[0].tolist()}",
            flush=True,
        )

        print(
            f"[OBS] stair_origin_w={stair_origin_w[0].tolist()}",
            flush=True,
        )

        terrain_contract = build_static_terrain_contract(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            stair_origin_w,
        )

        object_pos_b = terrain_contract[
            "object_pos_b"
        ][0]

        object_ori_b_6d = terrain_contract[
            "object_ori_b_6d"
        ][0]

        object_pos_delta = terrain_contract[
            "object_pos_delta_multi_future"
        ][0]

        object_ori_delta = terrain_contract[
            "object_ori_delta_multi_future_6d"
        ][0]

        print("[OBS] 27 before actor_obs concat", flush=True)

        actor_obs = torch.cat(
            (
                gravity_hist,
                ang_vel_hist,
                q_hist,
                dq_hist,
                action_hist,
                object_pos_b,
                object_ori_b_6d,
                object_pos_delta,
                object_ori_delta,
            ),
            dim=0,
        )

        print(
            f"[OBS] 28 actor_obs_numel={actor_obs.numel()}",
            flush=True,
        )

        if actor_obs.numel() != 1029:
            raise RuntimeError(
                f"actor_obs expected 1029, "
                f"got {actor_obs.numel()}"
            )

        #
        # Tokenizer reference future.
        #
        # External controller uses 10 future frames spaced by 0.1 s.
        # Reference data is 50 Hz, so frame spacing = 5.
        #
        print("[OBS] 29 entering tokenizer/reference block", flush=True)

        frame = 0

        print("[OBS] 30 before future_ids arange", flush=True)

        future_ids = torch.arange(
            10,
            device=device,
            dtype=torch.long,
        ) * 5 + frame

        print(
            f"[OBS] 31 future_ids_raw={future_ids.tolist()}",
            flush=True,
        )

        future_ids = torch.clamp(
            future_ids,
            max=q_ref.shape[0] - 1,
        )

        print(
            f"[OBS] 32 future_ids_clamped={future_ids.tolist()}",
            flush=True,
        )

        print("[OBS] 33 before reference indexing", flush=True)

        future_q = q_ref[future_ids]

        print(
            f"[OBS] 34 future_q_shape={tuple(future_q.shape)}",
            flush=True,
        )

        future_dq = dq_ref[future_ids]

        print(
            f"[OBS] 35 future_dq_shape={tuple(future_dq.shape)}",
            flush=True,
        )

        print("[OBS] 36 before command_future concat", flush=True)

        command_future = torch.cat(
            (
                future_q,
                future_dq,
            ),
            dim=-1,
        )

        print(
            f"[OBS] 37 command_future_shape={tuple(command_future.shape)}",
            flush=True,
        )

        if command_future.shape != (10, 58):
            raise RuntimeError(
                f"command_future shape={command_future.shape}"
            )

        #
        # Reference pelvis orientation.
        # Reference body index 0 is used by the extracted motion
        # contract as the root/pelvis anchor.
        #
        print("[OBS] 38 before reference root quaternion indexing", flush=True)

        ref_root_quat = body_quat[
            future_ids,
            0,
        ]

        print(
            f"[OBS] 39 ref_root_quat_shape={tuple(ref_root_quat.shape)}",
            flush=True,
        )

        print(
            f"[OBS] 40 ref_root_quat_0={ref_root_quat[0].tolist()}",
            flush=True,
        )

        robot_quat_expand = root_quat.unsqueeze(0).repeat(
            10,
            1,
        )

        print(
            f"[OBS] 41 robot_quat_expand_shape={tuple(robot_quat_expand.shape)}",
            flush=True,
        )

        print("[OBS] 42 before relative quaternion transform", flush=True)

        rel_ref_quat = quat_mul(
            quat_inv(robot_quat_expand),
            ref_root_quat,
        )

        print(
            f"[OBS] 43 rel_ref_quat_shape={tuple(rel_ref_quat.shape)}",
            flush=True,
        )

        print("[OBS] 44 before rot6d", flush=True)

        anchor_ori_future = rot6d(rel_ref_quat)

        print(
            f"[OBS] 45 anchor_ori_future_shape={tuple(anchor_ori_future.shape)}",
            flush=True,
        )

        if anchor_ori_future.shape != (10, 6):
            raise RuntimeError(
                "anchor orientation shape mismatch"
            )

        print("[OBS] 46 before command_z extraction", flush=True)

        command_z = body_pos[
            future_ids,
            0,
            2,
        ].reshape(10, 1)

        print(
            f"[OBS] 47 command_z_shape={tuple(command_z.shape)}",
            flush=True,
        )

        #
        # Exact tokenizer order:
        # encoder_index
        # command_multi_future_nonflat
        # command_z_multi_future_nonflat
        # motion_anchor_ori_b_mf_nonflat
        # height_map_z_flat
        #
        print("[OBS] 48 before encoder index", flush=True)

        encoder_index = torch.tensor(
            [1.0, 0.0, 0.0],
            device=device,
        )

        print(
            f"[OBS] 49 encoder_index={encoder_index.tolist()}",
            flush=True,
        )

        #
        # Height map is isolated for this contract gate.
        # Exact released configuration is 1.5 m × 1.5 m,
        # 0.15 m spacing = 11 × 11 = 121.
        #
        print("[OBS] 50 before height-map placeholder", flush=True)

        height_map_z = terrain_contract[
            "height_map_z_flat"
        ][0]

        print(
            f"[OBS] 51 height_map_numel={height_map_z.numel()}",
            flush=True,
        )

        print("[OBS] 52 before tokenizer concat", flush=True)

        tokenizer = torch.cat(
            (
                encoder_index,
                command_future.reshape(-1),
                command_z.reshape(-1),
                anchor_ori_future.reshape(-1),
                height_map_z,
            ),
            dim=0,
        )

        print(
            f"[OBS] 53 tokenizer_numel={tokenizer.numel()}",
            flush=True,
        )

        print(
            f"[OBS] object_pos_b={object_pos_b.tolist()}",
            flush=True,
        )

        print(
            f"[OBS] object_ori_b_6d={object_ori_b_6d.tolist()}",
            flush=True,
        )

        print(
            "[OBS] object_pos_delta_linf="
            f"{torch.max(torch.abs(object_pos_delta)).item():.9f}",
            flush=True,
        )

        print(
            "[OBS] height_map min/max="
            f"{height_map_z.min().item():.6f}/"
            f"{height_map_z.max().item():.6f}",
            flush=True,
        )

        print(
            "[OBS] height_map="
            f"{height_map_z.reshape(11, 11).tolist()}",
            flush=True,
        )

        print()
        print("[OBS] 54 semantic contract audit", flush=True)

        # -------------------------------------------------------------
        # A. Reference-array contract
        # -------------------------------------------------------------
        print(
            f"[OBS] 55 reference q0={q_ref[0].tolist()}",
            flush=True,
        )

        print(
            f"[OBS] 56 reference root quat0={body_quat[0, 0].tolist()}",
            flush=True,
        )

        ref_quat_norm = torch.linalg.vector_norm(
            body_quat[0, 0]
        ).item()

        print(
            f"[OBS] 57 reference root quat norm={ref_quat_norm:.9f}",
            flush=True,
        )

        # -------------------------------------------------------------
        # B. Exact local checkpoint configuration.
        # This is more authoritative than assumptions from public source.
        # -------------------------------------------------------------
        config_path = Path(
            "/workspace/GRAIL/imports/SONIC/"
            "models/terrain_release/config.yaml"
        )

        print(
            f"[OBS] 58 checkpoint config={config_path}",
            flush=True,
        )

        if not config_path.is_file():
            raise RuntimeError(
                f"Checkpoint config missing: {config_path}"
            )

        import yaml

        checkpoint_cfg = yaml.safe_load(
            config_path.read_text()
        )

        def find_key(obj, wanted, path="root"):
            found = []

            if isinstance(obj, dict):
                for key, value in obj.items():
                    child = f"{path}.{key}"

                    if key == wanted:
                        found.append((child, value))

                    found.extend(
                        find_key(value, wanted, child)
                    )

            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    found.extend(
                        find_key(
                            value,
                            wanted,
                            f"{path}[{i}]",
                        )
                    )

            return found

        for wanted in (
            "encoder_sample_probs",
            "use_encoder",
            "num_future_frames",
            "dt_future_ref_frames",
            "height_map_size",
            "height_map_resolution",
            "use_height_map",
        ):
            matches = find_key(
                checkpoint_cfg,
                wanted,
            )

            print(
                f"[OBS] CFG {wanted} matches={matches}",
                flush=True,
            )

        # -------------------------------------------------------------
        # C. Actual IEA scene representation.
        # Released terrain controller expects a scene object for its
        # object observations and height-map raycasts.
        # -------------------------------------------------------------
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        prim_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
        ]

        object_paths = [
            p for p in prim_paths
            if (
                p.endswith("/Object")
                or "/Object/" in p
                or "/object" in p.lower()
            )
        ]

        terrain_paths = [
            p for p in prim_paths
            if (
                "/ground" in p.lower()
                or "terrain" in p.lower()
                or "stair" in p.lower()
            )
        ]

        print(
            f"[OBS] 59 object-like prim count={len(object_paths)}",
            flush=True,
        )

        for p in object_paths[:30]:
            print(
                f"[OBS]     OBJECT_PRIM {p}",
                flush=True,
            )

        print(
            f"[OBS] 60 terrain-like prim count={len(terrain_paths)}",
            flush=True,
        )

        for p in terrain_paths[:50]:
            print(
                f"[OBS]     TERRAIN_PRIM {p}",
                flush=True,
            )

        # -------------------------------------------------------------
        # D. Scene entities exposed through Isaac Lab.
        # -------------------------------------------------------------
        print(
            "[OBS] 61 rigid object keys="
            f"{list(base_env.scene.rigid_objects.keys())}",
            flush=True,
        )

        print(
            "[OBS] 62 articulation keys="
            f"{list(base_env.scene.articulations.keys())}",
            flush=True,
        )

        print(
            "[OBS] 63 sensor keys="
            f"{list(base_env.scene.sensors.keys())}",
            flush=True,
        )

        if tokenizer.numel() != 774:
            raise RuntimeError(
                f"tokenizer expected 774, "
                f"got {tokenizer.numel()}"
            )

        print()
        print("=" * 80)
        print("REFERENCE OBSERVATION CONTRACT")
        print("=" * 80)

        print(
            f"actor_obs_shape=(1,1,{actor_obs.numel()})"
        )

        print(
            f"tokenizer_shape=(1,1,{tokenizer.numel()})"
        )

        print(
            f"gravity={gravity.tolist()}"
        )

        print(
            f"base_ang_vel={base_ang_vel.tolist()}"
        )

        print(
            f"q_rel_l2={torch.linalg.vector_norm(q_rel).item():.8f}"
        )

        print(
            f"future_ids={future_ids.tolist()}"
        )

        print(
            "command_future_shape="
            f"{tuple(command_future.shape)}"
        )

        print(
            "anchor_ori_future_shape="
            f"{tuple(anchor_ori_future.shape)}"
        )

        print()
        print("PASS exact actor dimension = 1029", flush=True)
        print("PASS exact tokenizer dimension = 774", flush=True)
        print("PASS exact actor term order locked", flush=True)
        print("PASS 10-frame future reference = 0.1 s spacing", flush=True)
        print("PASS live proprioception acquired", flush=True)
        print()
        print(
            "HOLD: object-state and height-map terms are "
            "shape-valid placeholders only"
        )
        print(
            "NO POLICY FORWARD APPLIED",
            flush=True,
        )

        # =============================================================
        # ZERO-NORMALIZED-ACTION HOLD
        #
        # Native action semantics:
        # q_target = q_default_model12 + scale * normalized_action
        #
        # Therefore zero normalized action must hold the model-12
        # default pose.
        # =============================================================

        print()
        print("=" * 80, flush=True)
        print("ZERO-ACTION HOLD: 2.0 s", flush=True)
        print("=" * 80, flush=True)

        hold_steps = 100
        zero_action = torch.zeros(
            (1, 29),
            dtype=torch.float32,
            device=device,
        )

        hold_root_start = robot.data.root_pos_w[0].clone()

        terminated_any = False
        truncated_any = False

        for hold_step in range(hold_steps):

            step_result = env.step(zero_action)

            if len(step_result) == 5:
                (
                    _obs,
                    _reward,
                    terminated,
                    truncated,
                    _info,
                ) = step_result

                terminated_any = (
                    terminated_any
                    or bool(terminated.any().item())
                )

                truncated_any = (
                    truncated_any
                    or bool(truncated.any().item())
                )

            q_now = robot.data.joint_pos[0]
            dq_now = robot.data.joint_vel[0]
            root_pos = robot.data.root_pos_w[0]
            root_quat_now = robot.data.root_quat_w[0]

            gravity_now = quat_apply(
                quat_inv(root_quat_now.unsqueeze(0)),
                down.unsqueeze(0),
            )[0]

            upright_cos = torch.clamp(
                -gravity_now[2],
                -1.0,
                1.0,
            )

            tilt_rad = torch.acos(
                upright_cos
            ).item()

            q_error = q_now - robot.data.default_joint_pos[0]

            if (
                hold_step == 0
                or (hold_step + 1) % 25 == 0
                or hold_step == hold_steps - 1
            ):
                print(
                    "[HOLD] "
                    f"step={hold_step + 1:03d} "
                    f"t={(hold_step + 1) * 0.02:.2f}s "
                    f"x={root_pos[0].item():+.4f} "
                    f"y={root_pos[1].item():+.4f} "
                    f"z={root_pos[2].item():+.4f} "
                    f"tilt={tilt_rad:.4f} "
                    f"q_rms="
                    f"{torch.sqrt(torch.mean(q_error * q_error)).item():.5f} "
                    f"dq_rms="
                    f"{torch.sqrt(torch.mean(dq_now * dq_now)).item():.5f}",
                    flush=True,
                )

            if terminated_any or truncated_any:
                print(
                    "[HOLD] environment terminated before 2.0 s",
                    flush=True,
                )
                break

        hold_root_end = robot.data.root_pos_w[0].clone()

        hold_xy_drift = torch.linalg.vector_norm(
            hold_root_end[:2] - hold_root_start[:2]
        ).item()

        final_root_quat = robot.data.root_quat_w[0]
        final_gravity = quat_apply(
            quat_inv(final_root_quat.unsqueeze(0)),
            down.unsqueeze(0),
        )[0]

        final_tilt = torch.acos(
            torch.clamp(
                -final_gravity[2],
                -1.0,
                1.0,
            )
        ).item()

        final_q_error = (
            robot.data.joint_pos[0]
            - robot.data.default_joint_pos[0]
        )

        final_q_rms = torch.sqrt(
            torch.mean(final_q_error * final_q_error)
        ).item()

        final_z = robot.data.root_pos_w[0, 2].item()

        print()
        print(
            f"[HOLD] final_z={final_z:.6f}",
            flush=True,
        )
        print(
            f"[HOLD] xy_drift={hold_xy_drift:.6f}",
            flush=True,
        )
        print(
            f"[HOLD] final_tilt_rad={final_tilt:.6f}",
            flush=True,
        )
        print(
            f"[HOLD] final_q_rms={final_q_rms:.6f}",
            flush=True,
        )
        print(
            f"[HOLD] terminated={terminated_any}",
            flush=True,
        )
        print(
            f"[HOLD] truncated={truncated_any}",
            flush=True,
        )

        hold_pass = (
            not terminated_any
            and not truncated_any
            and final_z > 0.55
            and final_tilt < 0.50
            and hold_xy_drift < 0.30
        )

        if hold_pass:
            print(
                "PASS ZERO-ACTION HOLD",
                flush=True,
            )
        else:
            print(
                "FAIL ZERO-ACTION HOLD",
                flush=True,
            )

        print(
            "NO PRETRAINED POLICY ACTIONS APPLIED",
            flush=True,
        )

    finally:
        env.close()


try:
    main()

except BaseException:
    print()
    print("[OBS] FAILURE")
    traceback.print_exc()
    raise

finally:
    simulation_app.close()
