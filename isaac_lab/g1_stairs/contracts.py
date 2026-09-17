"""Pure-Python contracts shared by training configuration and local tests."""

BODY_JOINTS = (
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

# Unitree RL Lab's launcher only offers registered task IDs containing
# "Unitree" as argparse choices.
TASK_ID = "Unitree-IEA-G1-29dof-Stairs-v0"
UPSTREAM_TASK_ID = "Unitree-G1-29dof-Velocity"
FIXED_FORWARD_COMMAND_MPS = 0.35
NOMINAL_RISER_HEIGHT_M = 0.20
MODE_MACHINE = 5
