from __future__ import annotations

import math


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
