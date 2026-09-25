#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/ubuntu/workspace/IEA_G1/motion_imitation")
MAHR = ROOT / "MAHR"

sys.path.insert(0, str(MAHR))

from config.MocapConfig.CMUAsfAmcConfig import (
    CMU_AsfAmc_WholeBody_Config,
)


ASF = MAHR / "mahr/rsc/all_asfamc/subjects/14/14.asf"
AMC = MAHR / "mahr/rsc/all_asfamc/subjects/14/14_21_ascent_170_540.amc"

TRAJ = (
    ROOT
    / "outputs/cmu_to_g1_trial21_smoke/"
      "optimized_trajectories_vector_graph/"
      "14_21_ascent_170_540/"
      "14_21_ascent_170_540_gd_v2.json"
)

URDF = (
    MAHR
    / "mahr/rsc/assets/urdf/unitree_g1/"
      "g1_29dof_stl.urdf"
)


# ============================================================
# Helpers
# ============================================================

def localize(points, origin, quat_xyzw):
    """
    Convert world points into root-local coordinates.
    """
    R = Rotation.from_quat(quat_xyzw)
    return {
        name: R.inv().apply(np.asarray(pos) - np.asarray(origin))
        for name, pos in points.items()
    }


def print_points(title, points):
    print()
    print(title)
    print("-" * len(title))
    print(f"{'name':24s} {'x':>10s} {'y':>10s} {'z':>10s}")

    for name, p in points.items():
        print(
            f"{name:24s} "
            f"{p[0]:10.4f} "
            f"{p[1]:10.4f} "
            f"{p[2]:10.4f}"
        )


def print_lr(label, left, right):
    d = np.asarray(left) - np.asarray(right)

    print(
        f"{label:16s} "
        f"L_y={left[1]: .4f} "
        f"R_y={right[1]: .4f} "
        f"L-R_y={d[1]: .4f} "
        f"distance={np.linalg.norm(d): .4f}"
    )


# ============================================================
# 1. Load CMU frame 0
# ============================================================

print("========================================")
print(" LEG MAPPING DIAGNOSTIC - FRAME 0")
print("========================================")

mocap = CMU_AsfAmc_WholeBody_Config(
    norm_pose_type="stand"
)

mocap.load_motion(
    str(ASF),
    str(AMC),
    120.0,
)

human = mocap.uni_repr.to_dict(is_local=False)

names = human["names"]
name_to_idx = {
    name: i for i, name in enumerate(names)
}

required_human = [
    "root",

    "lfemur",
    "ltibia",
    "lfoot",
    "ltoes",

    "rfemur",
    "rtibia",
    "rfoot",
    "rtoes",
]

missing = [
    x for x in required_human
    if x not in name_to_idx
]

if missing:
    raise RuntimeError(
        f"Missing CMU joints: {missing}"
    )

frame = 0

human_world = {
    name:
    np.asarray(
        human["global_translations"][frame][name_to_idx[name]]
    ).reshape(3)
    for name in required_human
}

root_idx = name_to_idx["root"]

human_root_pos = np.asarray(
    human["global_translations"][frame][root_idx]
).reshape(3)

human_root_quat = np.asarray(
    human["global_rotations"][frame][root_idx]
).reshape(4)

human_local = localize(
    human_world,
    human_root_pos,
    human_root_quat,
)

print_points(
    "CMU WORLD POSITIONS",
    human_world,
)

print_points(
    "CMU ROOT-LOCAL POSITIONS",
    human_local,
)


# ============================================================
# 2. Load retargeted G1 frame 0
# ============================================================

payload = json.loads(
    TRAJ.read_text()
)

g1_frame = payload["trajectory"][0]

spec = mujoco.MjSpec.from_file(str(URDF))

urdf_dir = URDF.parent

def resolve(path):
    if not path:
        return path

    p = Path(path)

    if p.is_absolute():
        return str(p)

    return str(
        (urdf_dir / p).resolve()
    )


spec.meshdir = resolve(spec.meshdir)
spec.texturedir = resolve(spec.texturedir)
spec.modelfiledir = resolve(spec.modelfiledir)

if spec.compiler is not None:
    spec.compiler.meshdir = resolve(
        spec.compiler.meshdir
    )


# Add free joint to the robot root
root_body = spec.worldbody.bodies[0]

has_free = any(
    joint.type == mujoco.mjtJoint.mjJNT_FREE
    for joint in root_body.joints
)

if not has_free:
    root_body.add_joint(
        type=mujoco.mjtJoint.mjJNT_FREE,
        name="diagnostic_root_free_joint",
    )


model = spec.compile()
data = mujoco.MjData(model)


# ============================================================
# Apply root pose
# ============================================================

free_joint_id = None

for jid in range(model.njnt):
    if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        free_joint_id = jid
        break

if free_joint_id is None:
    raise RuntimeError("No free root joint found")

addr = int(
    model.jnt_qposadr[free_joint_id]
)

root_pos = np.asarray(
    g1_frame["RootPos"],
    dtype=float,
)

root_quat_xyzw = np.asarray(
    g1_frame["RootOrient"],
    dtype=float,
)

# MuJoCo expects wxyz
root_quat_wxyz = np.array([
    root_quat_xyzw[3],
    root_quat_xyzw[0],
    root_quat_xyzw[1],
    root_quat_xyzw[2],
])

data.qpos[addr:addr + 3] = root_pos
data.qpos[addr + 3:addr + 7] = root_quat_wxyz


# ============================================================
# Apply G1 joint positions
# ============================================================

joint_addr = {}

for jid in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        jid,
    )

    if name:
        joint_addr[name] = int(
            model.jnt_qposadr[jid]
        )


for name, value in g1_frame["JointPos"].items():
    if name not in joint_addr:
        raise RuntimeError(
            f"G1 trajectory joint absent from model: {name}"
        )

    data.qpos[
        joint_addr[name]
    ] = float(value)


mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# Read marker BODY positions
# ============================================================

marker_names = [
    "pelvis_marker",

    "left_knee_marker",
    "left_ankle_marker",
    "left_toe_marker",

    "right_knee_marker",
    "right_ankle_marker",
    "right_toe_marker",
]

g1_world = {}

for name in marker_names:

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise RuntimeError(
            f"Marker body not found: {name}"
        )

    g1_world[name] = (
        data.xpos[bid].copy()
    )


g1_local = localize(
    g1_world,
    root_pos,
    root_quat_xyzw,
)

print_points(
    "G1 WORLD MARKER POSITIONS",
    g1_world,
)

print_points(
    "G1 ROOT-LOCAL MARKER POSITIONS",
    g1_local,
)


# ============================================================
# 3. Left / right ordering
# ============================================================

print()
print("===== CMU LEFT/RIGHT SEPARATION =====")

print_lr(
    "hip/femur",
    human_local["lfemur"],
    human_local["rfemur"],
)

print_lr(
    "knee/tibia",
    human_local["ltibia"],
    human_local["rtibia"],
)

print_lr(
    "ankle/foot",
    human_local["lfoot"],
    human_local["rfoot"],
)

print_lr(
    "toe",
    human_local["ltoes"],
    human_local["rtoes"],
)


print()
print("===== G1 LEFT/RIGHT SEPARATION =====")

print_lr(
    "knee",
    g1_local["left_knee_marker"],
    g1_local["right_knee_marker"],
)

print_lr(
    "ankle",
    g1_local["left_ankle_marker"],
    g1_local["right_ankle_marker"],
)

print_lr(
    "toe",
    g1_local["left_toe_marker"],
    g1_local["right_toe_marker"],
)


# ============================================================
# 4. Current graph correspondence
# ============================================================

print()
print("===== CURRENT LEG GRAPH CORRESPONDENCE =====")

pairs = [
    ("lfemur", "left_knee_marker"),
    ("ltibia", "left_ankle_marker"),
    ("lfoot", "left_toe_marker"),

    ("rfemur", "right_knee_marker"),
    ("rtibia", "right_ankle_marker"),
    ("rfoot", "right_toe_marker"),
]

for human_name, robot_name in pairs:

    hp = human_local[human_name]
    rp = g1_local[robot_name]

    print()
    print(
        f"{human_name:10s} -> {robot_name}"
    )

    print(
        f"  CMU local : "
        f"[{hp[0]: .4f}, "
        f"{hp[1]: .4f}, "
        f"{hp[2]: .4f}]"
    )

    print(
        f"  G1  local : "
        f"[{rp[0]: .4f}, "
        f"{rp[1]: .4f}, "
        f"{rp[2]: .4f}]"
    )


print()
print("========================================")
print(" DIAGNOSTIC COMPLETE")
print("========================================")
