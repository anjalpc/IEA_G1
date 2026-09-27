"""Project-owned Unitree G1 asset configurations for stair training.

Historical experiments keep their existing URDF-based robot configuration.
The specialist stair profile uses Unitree's official 29-DoF USD together with
the nominal PD gains from the shipped hardware deployment contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import copy

from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_CFG,
    UnitreeUrdfFileCfg,
    UnitreeUsdFileCfg,
)

UNITREE_ROS_ROOT = Path(
    os.environ.get("UNITREE_ROS_ROOT", "/workspace/unitree_ros")
)

G1_29DOF_URDF = (
    UNITREE_ROS_ROOT
    / "robots"
    / "g1_description"
    / "g1_29dof_rev_1_0.urdf"
)

G1_STAIRS_ROBOT_CFG = UNITREE_G1_29DOF_CFG.replace(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(G1_29DOF_URDF),
    )
)


# Specialist profile -------------------------------------------------------
#
# Keep this separate from G1_STAIRS_ROBOT_CFG so historical experiments
# remain reproducible. The specialist profile is selected dynamically by
# specialist_runtime.py.

UNITREE_MODEL_ROOT = Path(
    os.environ.get("UNITREE_MODEL_ROOT", "/workspace/unitree_model")
)

G1_29DOF_USD = (
    UNITREE_MODEL_ROOT
    / "G1"
    / "29dof"
    / "usd"
    / "g1_29dof_rev_1_0"
    / "g1_29dof_rev_1_0.usd"
)

G1_SPECIALIST_ROBOT_CFG = copy.deepcopy(UNITREE_G1_29DOF_CFG)
G1_SPECIALIST_ROBOT_CFG.spawn = UnitreeUsdFileCfg(
    usd_path=str(G1_29DOF_USD),
)

# Unitree's shipped G1 29-DoF deployment YAML commands these nominal gains
# on the physical robot. Current upstream simulation source lowered the arm
# derivative gains to 1.0 without changing the deployed policy/controller
# contract. The specialist plant therefore pins the complete nominal PD gain
# map explicitly instead of inheriting that drift.
G1_SPECIALIST_ROBOT_CFG.actuators["N7520-14.3"].stiffness = {
    ".*_hip_.*": 100.0,
    "waist_yaw_joint": 200.0,
}
G1_SPECIALIST_ROBOT_CFG.actuators["N7520-14.3"].damping = {
    ".*_hip_.*": 2.0,
    "waist_yaw_joint": 5.0,
}

G1_SPECIALIST_ROBOT_CFG.actuators["N7520-22.5"].stiffness = {
    ".*_hip_roll_.*": 100.0,
    ".*_knee_.*": 150.0,
}
G1_SPECIALIST_ROBOT_CFG.actuators["N7520-22.5"].damping = {
    ".*_hip_roll_.*": 2.0,
    ".*_knee_.*": 4.0,
}

G1_SPECIALIST_ROBOT_CFG.actuators["N5020-16"].stiffness = {
    ".*_shoulder_.*": 40.0,
    ".*_elbow_.*": 40.0,
    ".*_wrist_roll.*": 40.0,
    ".*_ankle_.*": 40.0,
    "waist_.*_joint": 200.0,
}
G1_SPECIALIST_ROBOT_CFG.actuators["N5020-16"].damping = {
    ".*_shoulder_.*": 10.0,
    ".*_elbow_.*": 10.0,
    ".*_wrist_roll.*": 10.0,
    ".*_ankle_.*": 2.0,
    "waist_.*_joint": 5.0,
}

G1_SPECIALIST_ROBOT_CFG.actuators["W4010-25"].stiffness = 40.0
G1_SPECIALIST_ROBOT_CFG.actuators["W4010-25"].damping = 10.0

# Geometry-conditioned locomotion profile ---------------------------------
#
# The official Unitree velocity-policy deployment contract uses:
#   hips       100 / 2
#   knees      150 / 4
#   ankles      40 / 2
#   waist      200 / 5
#   upper body  40 / 10
#
# UNITREE_G1_29DOF_CFG currently differs from that deployed controller
# contract in the waist roll/pitch stiffness and upper-body damping.
#
# The specialist profile above already pins the complete deployment map.
# New geometry-conditioned locomotion therefore uses that qualified plant
# rather than the historical URDF/upstream-gain configuration.
G1_GEOMETRY_LOCOMOTION_ROBOT_CFG = copy.deepcopy(
    G1_SPECIALIST_ROBOT_CFG
)

# On this branch the normal stair environment is the geometry-locomotion
# environment. Historical behavior remains preserved on the archived
# g1-stair-rl / g1-stair-specialist branches and tags.
G1_STAIRS_ROBOT_CFG = copy.deepcopy(
    G1_GEOMETRY_LOCOMOTION_ROBOT_CFG
)
