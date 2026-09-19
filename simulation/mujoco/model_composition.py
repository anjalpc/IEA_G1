#!/usr/bin/env python3
"""Compose the vendored G1 MJCF with both vendored BrainCo Revo2 hands."""

from __future__ import annotations

from pathlib import Path

import mujoco


SIMULATION_ROOT = Path(__file__).resolve().parent
G1_SCENE = SIMULATION_ROOT / "vendor" / "unitree_g1" / "scene_29dof.xml"
BRAINCO_ROOT = SIMULATION_ROOT / "vendor" / "brainco_revo2"

BODY_ACTUATOR_COUNT = 29
HAND_CHANNELS_PER_SIDE = 6
HAND_ACTUATOR_COUNT = 2 * HAND_CHANNELS_PER_SIDE

# The manufacturer interface exposes six channels. Five distal joints follow
# their proximal joints mechanically and therefore are constraints, not extra
# command channels.
HAND_DRIVEN_JOINTS = (
    ("thumb_proximal", (0.0, 1.04720)),
    ("thumb_metacarpal", (0.0, 1.55334)),
    ("index_proximal", (0.0, 1.41)),
    ("middle_proximal", (0.0, 1.41)),
    ("ring_proximal", (0.0, 1.41)),
    ("pinky_proximal", (0.0, 1.41)),
)
MIMIC_MULTIPLIERS = {
    "thumb": 1.0945,
    "index": 1.131,
    "middle": 1.267,
    "ring": 1.252,
    "pinky": 1.271,
}

# The Unitree placeholder hand extends along wrist +X; the BrainCo URDF hand
# extends along base +Z. The +90 degree Y component aligns those axes. A
# mirrored +/-90 degree twist around the BrainCo longitudinal Z axis then puts
# each palm inward toward its corresponding hip in the relaxed G1 posture.
# Positions reuse the original placeholder attachment origins in the Unitree
# MJCF. Quaternions are MuJoCo wxyz values.
HAND_MOUNTS = {
    "right": ((0.0415, -0.003, 0.0), (0.5, 0.5, 0.5, 0.5)),
    "left": ((0.0415, 0.003, 0.0), (0.5, -0.5, 0.5, -0.5)),
}


def _load_hand(side: str) -> mujoco.MjSpec:
    urdf = BRAINCO_ROOT / "urdf" / f"revo2_{side}_hand.urdf"
    mesh_root = BRAINCO_ROOT / "meshes" / f"revo2_{side}_hand"
    assets = {path.name: path.read_bytes() for path in mesh_root.iterdir()
              if path.is_file()}
    hand = mujoco.MjSpec.from_string(urdf.read_text(encoding="utf-8"), assets=assets)

    # The vendor URDF contains no passive damping or reflected rotor inertia.
    # Without simulation-side values the coupled finger links can oscillate
    # indefinitely after a command or whole-arm movement.
    for joint in hand.joints:
        joint.damping = 0.20
        joint.armature = 1.0e-5
        joint.frictionloss = 0.01

    # URDF import strips package:// paths to basenames. After attachment the
    # parent MJCF mesh directory would otherwise be used, so retain explicit
    # paths to the vendored BrainCo assets.
    for mesh in hand.meshes:
        mesh.file = str(mesh_root / mesh.file)
    return hand


def _remove_placeholder(spec: mujoco.MjSpec, side: str) -> None:
    wrist = spec.body(f"{side}_wrist_yaw_link")
    if wrist is None:
        raise ValueError(f"G1 model is missing {side}_wrist_yaw_link")
    placeholders = [geom for geom in wrist.geoms
                    if geom.meshname == f"{side}_rubber_hand"]
    if len(placeholders) != 1:
        raise ValueError(
            f"expected one {side} rubber-hand placeholder, found {len(placeholders)}")
    spec.delete(placeholders[0])


def _add_hand_controls(spec: mujoco.MjSpec, side: str) -> None:
    for short_name, control_range in HAND_DRIVEN_JOINTS:
        joint_name = f"{side}_{short_name}_joint"
        actuator = spec.add_actuator(
            name=f"{side}_brainco_{short_name}", target=joint_name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        # These gains drive the detailed imported hand inertia and mimic
        # constraints in MuJoCo. They are simulation-only and are not hardware
        # motor gains.
        actuator.set_to_position(kp=160.0, kv=8.0)
        actuator.ctrllimited = True
        actuator.ctrlrange = control_range

    for finger, multiplier in MIMIC_MULTIPLIERS.items():
        equality = spec.add_equality(name=f"{side}_{finger}_mimic")
        equality.type = mujoco.mjtEq.mjEQ_JOINT
        equality.name1 = f"{side}_{finger}_distal_joint"
        equality.name2 = f"{side}_{finger}_proximal_joint"
        equality.data[0] = 0.0
        equality.data[1] = multiplier
        # Keep the mechanical following error small under finger contact and
        # position-control loads while retaining a numerically stable soft
        # constraint.
        equality.solref = (0.002, 1.0)


def build_combined_spec() -> mujoco.MjSpec:
    """Return an editable G1 specification with both BrainCo hands attached."""
    # Compose from the full vendor scene, not the bare robot MJCF. The scene
    # supplies the visible/contacting floor, lighting, and camera defaults.
    spec = mujoco.MjSpec.from_file(str(G1_SCENE))
    for side in ("right", "left"):
        _remove_placeholder(spec, side)
        wrist = spec.body(f"{side}_wrist_yaw_link")
        position, quaternion = HAND_MOUNTS[side]
        frame = wrist.add_frame(
            name=f"{side}_brainco_mount", pos=position, quat=quaternion)
        spec.attach(_load_hand(side), frame=frame)
        _add_hand_controls(spec, side)
    return spec


def build_combined_model() -> mujoco.MjModel:
    """Compile and return the integrated G1 and dual-BrainCo model."""
    return build_combined_spec().compile()


def hand_actuator_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_brainco_{name}" for name, _ in HAND_DRIVEN_JOINTS)


def validate_combined_model(model: mujoco.MjModel) -> None:
    """Raise if the compiled model does not contain the promised integration."""
    expected_actuators = BODY_ACTUATOR_COUNT + HAND_ACTUATOR_COUNT
    if model.nu != expected_actuators:
        raise ValueError(f"expected {expected_actuators} actuators, found {model.nu}")
    if model.neq != 2 * len(MIMIC_MULTIPLIERS):
        raise ValueError(
            f"expected 10 hand mimic constraints, found {model.neq}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") < 0:
        raise ValueError("combined model is missing the Unitree scene floor")
    if tuple(model.opt.gravity) != (0.0, 0.0, -9.81):
        raise ValueError(f"unexpected simulation gravity: {model.opt.gravity}")

    for side in ("right", "left"):
        base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_base_link")
        if base_id < 0:
            raise ValueError(f"combined model is missing {side} BrainCo base")
        parent_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[base_id])
        if parent_name != f"{side}_wrist_yaw_link":
            raise ValueError(
                f"{side} BrainCo base is attached to {parent_name}, not its wrist")
        for name in hand_actuator_names(side):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0:
                raise ValueError(f"combined model is missing actuator {name}")
