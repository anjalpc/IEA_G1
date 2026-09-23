"""Closed-loop qualification entry point for a pretrained G1 controller."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

print("[QUAL] 01 script started", flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default=None,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[QUAL] 02 launching Isaac application", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[QUAL] 03 Isaac application launched", flush=True)


def main() -> None:
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    # Explicitly register the local stair task.
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    task_id = args.task or "Unitree-IEA-G1-29dof-Stairs-v0"

    print(f"[QUAL] 04 building environment config: {task_id}", flush=True)

    env_cfg = parse_env_cfg(
        task_id,
        device=args.device,
        num_envs=1,
    )

    env_cfg.seed = 42


    #
    # Pretrained controller model-12 contract.
    #
    from isaaclab.actuators import ImplicitActuatorCfg

    K_5020 = 14.2506231
    D_5020 = 0.90722284
    A_5020 = 0.003609725

    K_7520_14 = 40.1792385
    D_7520_14 = 2.55788977
    A_7520_14 = 0.010177520

    K_7520_22 = 99.0984278
    D_7520_22 = 6.30880185
    A_7520_22 = 0.025101925

    K_4010 = 16.7783275
    D_4010 = 1.0681415
    A_4010 = 0.00425

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
            stiffness=K_7520_14,
            damping=D_7520_14,
            armature=A_7520_14,
        ),

        "hip_pitch_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=K_7520_22,
            damping=D_7520_22,
            armature=A_7520_22,
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
            stiffness=2.0 * K_5020,
            damping=2.0 * D_5020,
            armature=2.0 * A_5020,
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
            stiffness=K_5020,
            damping=D_5020,
            armature=A_5020,
        ),

        "wrist_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=5.0,
            velocity_limit_sim=22.0,
            stiffness=K_4010,
            damping=D_4010,
            armature=A_4010,
        ),
    }

    #
    # Exact normalized-action transform used by the pretrained policy.
    #
    scale_map = {
        ".*_hip_pitch_joint": 0.25 * 139.0 / K_7520_22,
        ".*_hip_roll_joint": 0.25 * 139.0 / K_7520_22,
        ".*_hip_yaw_joint": 0.25 * 88.0 / K_7520_14,
        ".*_knee_joint": 0.25 * 139.0 / K_7520_22,

        ".*_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * K_5020),
        ".*_ankle_roll_joint": 0.25 * 50.0 / (2.0 * K_5020),

        "waist_yaw_joint": 0.25 * 88.0 / K_7520_14,
        "waist_roll_joint": 0.25 * 50.0 / (2.0 * K_5020),
        "waist_pitch_joint": 0.25 * 50.0 / (2.0 * K_5020),

        ".*_shoulder_pitch_joint": 0.25 * 25.0 / K_5020,
        ".*_shoulder_roll_joint": 0.25 * 25.0 / K_5020,
        ".*_shoulder_yaw_joint": 0.25 * 25.0 / K_5020,
        ".*_elbow_joint": 0.25 * 25.0 / K_5020,
        ".*_wrist_roll_joint": 0.25 * 25.0 / K_5020,

        ".*_wrist_pitch_joint": 0.25 * 5.0 / K_4010,
        ".*_wrist_yaw_joint": 0.25 * 5.0 / K_4010,
    }

    # Resolve the existing JointPositionAction configuration without
    # assuming the StairActionsCfg field name.
    action_cfg = None
    action_cfg_name = None

    for name, value in vars(env_cfg.actions).items():
        if value is None:
            continue

        class_type = getattr(value, "class_type", None)
        class_name = getattr(class_type, "__name__", "")

        if class_name == "JointPositionAction":
            action_cfg = value
            action_cfg_name = name
            break

    if action_cfg is None:
        available = {
            name: type(value).__name__
            for name, value in vars(env_cfg.actions).items()
            if not name.startswith("_")
        }
        raise RuntimeError(
            "Could not locate JointPositionActionCfg in StairActionsCfg. "
            f"Available fields: {available}"
        )

    print(
        f"[QUAL] resolved action config field={action_cfg_name}",
        flush=True,
    )

    action_cfg.use_default_offset = True
    action_cfg.scale = scale_map

    print("[QUAL] model-12 contract injected before gym.make()", flush=True)
    print(
        "[QUAL] config hip_pitch_default="
        f"{env_cfg.scene.robot.init_state.joint_pos['.*_hip_pitch_joint']}"
    )
    print(
        "[QUAL] config knee_default="
        f"{env_cfg.scene.robot.init_state.joint_pos['.*_knee_joint']}"
    )
    print(
        "[QUAL] config hip_pitch_scale="
        f"{scale_map['.*_hip_pitch_joint']:.6f}"
    )
    print(
        "[QUAL] config hip_yaw_scale="
        f"{scale_map['.*_hip_yaw_joint']:.6f}"
    )
    print(
        "[QUAL] config ankle_scale="
        f"{scale_map['.*_ankle_pitch_joint']:.6f}"
    )

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    print(f"[QUAL] 05 creating task: {task_id}", flush=True)

    env = gym.make(
        task_id,
        cfg=env_cfg,
        render_mode=None,
    )

    try:
        unwrapped = env.unwrapped

        print("[QUAL] 06 task created", flush=True)

        if not hasattr(unwrapped, "scene"):
            raise RuntimeError("Task does not expose an Isaac Lab scene")

        try:
            robot = unwrapped.scene["robot"]
        except KeyError as exc:
            raise RuntimeError(
                "Scene does not contain asset 'robot'"
            ) from exc

        joint_names = list(robot.joint_names)

        print("[QUAL] 07 robot articulation located", flush=True)
        print(f"[QUAL]     joint_count={len(joint_names)}")

        if len(joint_names) != 29:
            raise RuntimeError(
                f"Expected 29 body joints, found {len(joint_names)}"
            )

        print("[QUAL] 08 joint order")
        for index, name in enumerate(joint_names):
            print(f"[QUAL]     {index:02d}: {name}")

        default_pos = robot.data.default_joint_pos[0].detach().cpu()
        default_vel = robot.data.default_joint_vel[0].detach().cpu()

        if default_pos.numel() != 29:
            raise RuntimeError(
                f"Expected 29 default positions, got {default_pos.numel()}"
            )

        print("[QUAL] 09 default-state contract")
        print(
            "[QUAL]     q_default="
            + " ".join(f"{x:.6f}" for x in default_pos.tolist())
        )
        print(
            "[QUAL]     dq_default="
            + " ".join(f"{x:.6f}" for x in default_vel.tolist())
        )

        #
        # Reset only. No pretrained action is applied yet.
        #
        print("[QUAL] 10 resetting live environment", flush=True)

        reset_result = env.reset()

        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result

        print("[QUAL] 11 environment reset completed", flush=True)

        q = robot.data.joint_pos[0].detach().cpu()
        dq = robot.data.joint_vel[0].detach().cpu()

        print(
            "[QUAL]     q_live="
            + " ".join(f"{x:.6f}" for x in q.tolist())
        )

        print(
            "[QUAL]     dq_live="
            + " ".join(f"{x:.6f}" for x in dq.tolist())
        )

        print()
        print("[QUAL] 12 action boundary audit")

        action_manager = unwrapped.action_manager

        print(f"[QUAL]     action_dim={action_manager.total_action_dim}")

        for term_name, term in action_manager._terms.items():
            print(
                f"[QUAL]     action_term={term_name} "
                f"type={type(term).__name__}"
            )

            for attr_name in (
                "joint_ids",
                "_joint_ids",
                "scale",
                "_scale",
                "offset",
                "_offset",
                "raw_actions",
                "processed_actions",
            ):
                if not hasattr(term, attr_name):
                    continue

                value = getattr(term, attr_name)

                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()

                    print(
                        f"[QUAL]       {attr_name}_shape="
                        f"{tuple(value.shape)}"
                    )

                    if value.numel() <= 64:
                        print(
                            f"[QUAL]       {attr_name}="
                            + " ".join(
                                f"{x:.6f}"
                                for x in value.reshape(-1).tolist()
                            )
                        )
                else:
                    print(
                        f"[QUAL]       {attr_name}={value}"
                    )

        print()
        print("[QUAL] 13 articulation actuator audit")

        for actuator_name, actuator in robot.actuators.items():
            print(
                f"[QUAL]     actuator={actuator_name} "
                f"type={type(actuator).__name__}"
            )

            for attr_name in (
                "joint_names",
                "stiffness",
                "damping",
                "armature",
                "effort_limit",
                "effort_limit_sim",
                "velocity_limit",
                "velocity_limit_sim",
            ):
                if not hasattr(actuator, attr_name):
                    continue

                value = getattr(actuator, attr_name)

                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                    print(
                        f"[QUAL]       {attr_name}="
                        + " ".join(
                            f"{x:.6f}"
                            for x in value.reshape(-1).tolist()
                        )
                    )
                else:
                    print(
                        f"[QUAL]       {attr_name}={value}"
                    )

        print()
        print("=" * 80)
        print("LIVE CONTROLLER BINDING GATE")
        print("=" * 80)
        print("task:", task_id)
        print("joint_count:", len(joint_names))
        print("device:", robot.device)
        print()
        print("PASS: existing IEA stair task launched")
        print("PASS: live G1 articulation located")
        print("PASS: live G1 exposes exactly 29 body joints")
        print("PASS: default and live joint state are readable")
        print()
        print("NO POLICY ACTIONS WERE APPLIED")

    finally:
        env.close()


try:
    main()

except BaseException:
    print()
    print("[QUAL] FAILURE", flush=True)
    traceback.print_exc()
    raise

finally:
    print("[QUAL] closing Isaac application", flush=True)
    simulation_app.close()
