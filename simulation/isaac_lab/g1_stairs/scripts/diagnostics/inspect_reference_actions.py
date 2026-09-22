from pathlib import Path

import numpy as np

from simulation.isaac_lab.g1_stairs.reference_motion import (
    load_reference_motion,
)


ROOT = Path(__file__).resolve().parents[5]

motion = load_reference_motion(
    ROOT
    / "experiments"
    / "g1_stairs"
    / "reference_data"
    / "stair_ascent_reference.yaml"
)

joint_names = (
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

default = np.asarray([
    -0.1, -0.1, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.3, 0.3,
    0.3, 0.3,
    -0.2, -0.2,
    0.25, -0.25,
    0.0, 0.0,
    0.0, 0.0,
    0.97, 0.97,
    0.15, -0.15,
    0.0, 0.0,
    0.0, 0.0,
], dtype=float)

scale = 0.25

# Inspect only the source portion relevant to the three-step task.
end_frame = 260

q = motion.dof_pos[: end_frame + 1]
actions = (q - default[None, :]) / scale

print("=" * 80)
print("REFERENCE -> IEA_G1 ACTION ANALYSIS")
print("=" * 80)

print("frames:", len(actions))
print("max abs action:", float(np.abs(actions).max()))
print("mean abs action:", float(np.abs(actions).mean()))
print("RMS action:", float(np.sqrt(np.mean(actions ** 2))))

print()
print("PER-JOINT MAX ABS ACTION")
print("=" * 80)

max_per_joint = np.abs(actions).max(axis=0)

for name, value in sorted(
    zip(joint_names, max_per_joint),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{name:30s} {value:7.3f}")

print()
print("FRAMES WITH LARGEST ACTION")
print("=" * 80)

frame_max = np.abs(actions).max(axis=1)

for frame in np.argsort(frame_max)[-15:][::-1]:
    joint = int(np.argmax(np.abs(actions[frame])))

    print(
        f"frame={frame:3d} "
        f"time={frame/motion.frequency_hz:5.2f}s "
        f"max={frame_max[frame]:6.3f} "
        f"joint={joint_names[joint]}"
    )
