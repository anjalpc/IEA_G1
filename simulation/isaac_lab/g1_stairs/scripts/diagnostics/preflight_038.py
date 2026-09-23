#!/usr/bin/env python3

from pathlib import Path
import ast
import yaml


ROOT = Path(__file__).resolve().parents[5]

CFG = (
    ROOT
    / "experiments/g1_stairs/configs/"
      "038b_geometry_careful_ascent_learning.yaml"
)

ENV = ROOT / "simulation/isaac_lab/g1_stairs/env_cfg.py"
OBS = ROOT / "simulation/isaac_lab/g1_stairs/terrain_observations.py"
REW = ROOT / "simulation/isaac_lab/g1_stairs/unified_rewards.py"
TRAIN = (
    ROOT
    / "simulation/isaac_lab/g1_stairs/"
      "scripts/training/train_isaac_stairs.py"
)
REPLAY = (
    ROOT
    / "simulation/isaac_lab/g1_stairs/"
      "evaluation/step_up_checkpoint_validation.py"
)


def merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for key, value in override.items():
            out[key] = (
                merge(out[key], value)
                if key in out
                else value
            )
        return out
    return override


def resolve(path):
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    base = (
        data.get("inheritance", {})
        .get("base_config")
    )

    if not base:
        return data

    return merge(
        resolve(ROOT / base),
        data,
    )


def fail(message):
    raise SystemExit(f"[FAIL] {message}")


resolved = resolve(CFG)

print("========================================")
print(" G1 038 FULL HOST PREFLIGHT")
print("========================================")


# Frozen architecture.
obs_cfg = resolved["observations"]
init = resolved["policy_initialization"]

if obs_cfg["actor_height_scan"] is not False:
    fail("actor height scan changed")

if obs_cfg["critic_height_scan"] is not True:
    fail("critic height scan changed")

checks = {
    "locomotion_observations": 480,
    "actor_height_scan_observations": 0,
    "explicit_geometry_observations": 5,
    "stair_observations": 485,
    "actions": 29,
}

for key, expected in checks.items():
    if init[key] != expected:
        fail(f"{key} changed: {init[key]} != {expected}")

print("[ OK ] frozen actor 485 / critic contract preserved")


# Terrain contract.
terrain = resolved["terrain"]

if terrain["riser_height_range_m"] != [0.06, 0.12]:
    fail("6-12 cm range changed")

if terrain["tread_depth_m"] != 0.25:
    fail("25 cm tread changed")

if terrain["step_count"] != 3:
    fail("three-step contract changed")

print("[ OK ] terrain contract preserved")


# Dynamic command schedule.
command = resolved["command"]

if float(command["forward_velocity_mps"]) != 0.50:
    fail("approach command must remain 0.50")

schedule = command["stair_velocity_schedule"]

expected_schedule = {
    "enabled": True,
    "approach_velocity_mps": 0.50,
    "stair_velocity_mps": 0.18,
    "slowdown_start_distance_m": 0.60,
    "slowdown_end_distance_m": 0.35,
    "platform_velocity_mps": 0.50,
    "platform_accel_distance_m": 0.40,
}

if schedule != expected_schedule:
    fail(
        f"velocity schedule mismatch: {schedule}"
    )

print("[ OK ] 0.50 -> 0.18 -> 0.50 speed schedule exact")


# Reward stage.
stage_name = (
    resolved["reward_experiment"]["active_stage"]
)

stage = (
    resolved["reward_experiment"]
    ["rewards"]
    ["stage_plan"]
    [stage_name]
)

expected_terms = {
    "track_stair_velocity_world",
    "track_ang_vel_z",
    "upright",
    "action_rate",
    "joint_acc",
    "dof_pos_limits",
    "feet_slide",
    "undesired_contacts",
    "upright_forward_velocity_shortfall",
    "stair_heading_alignment",
    "terrain_relative_swing_clearance_dynamic",
    "terrain_relative_clearance_shortfall",
    "terrain_riser_face_blocked",
    "terrain_tread_touchdown",
}

if set(stage) != expected_terms:
    fail(
        "038 reward set mismatch:\n"
        f"actual={sorted(stage)}"
    )

print("[ OK ] active 038 reward stage exact")


# StairRewardsCfg catalog.
env_tree = ast.parse(
    ENV.read_text(encoding="utf-8")
)

reward_class = next(
    node
    for node in env_tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "StairRewardsCfg"
)

catalog = {
    target.id
    for node in reward_class.body
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name)
}

for name in expected_terms:
    if name not in catalog:
        fail(
            f"{name} absent from StairRewardsCfg"
        )

print("[ OK ] all active reward catalog fields exist")


# Callable symbols.
reward_tree = ast.parse(
    REW.read_text(encoding="utf-8")
)

reward_symbols = {
    node.name
    for node in reward_tree.body
    if isinstance(node, ast.FunctionDef)
}

for symbol in {
    "track_stair_velocity_world_exp",
    "stair_heading_alignment",
    "terrain_relative_swing_clearance_dynamic",
    "terrain_relative_clearance_shortfall",
    "terrain_tread_touchdown",
    "upright_forward_velocity_shortfall",
}:
    if symbol not in reward_symbols:
        fail(
            f"{symbol} callable missing"
        )

print("[ OK ] all new reward callables exist")


# Command observation function.
obs_tree = ast.parse(
    OBS.read_text(encoding="utf-8")
)

obs_symbols = {
    node.name
    for node in obs_tree.body
    if isinstance(node, ast.FunctionDef)
}

if "stair_velocity_command" not in obs_symbols:
    fail("stair_velocity_command missing")

if "stair_geometry" not in obs_symbols:
    fail("5-D stair_geometry missing")

print("[ OK ] dynamic command + 5-D geometry exist")


# Runtime and replay hooks.
train_source = TRAIN.read_text(encoding="utf-8")
replay_source = REPLAY.read_text(encoding="utf-8")

for required in (
    '"stair_velocity_schedule"',
    "terrain_observations.stair_velocity_command",
    "track_stair_velocity_world",
):
    if required not in train_source:
        fail(
            f"training hook missing: {required}"
        )

for required in (
    '"stair_velocity_schedule"',
    "terrain_observations.stair_velocity_command",
):
    if required not in replay_source:
        fail(
            f"replay hook missing: {required}"
        )

print("[ OK ] training and replay consume dynamic speed schedule")


# Ensure old historical reward remains.
reward_source = REW.read_text(encoding="utf-8")

if "def terrain_relative_swing_clearance(" not in reward_source:
    fail("historical 035 reward was deleted")

if (
    "terrain_riser_face_strike = "
    "terrain_riser_face_strike_proxy"
    not in reward_source
):
    fail("historical riser alias missing")

print("[ OK ] backward compatibility preserved")


print()
print("========================================")
print(" PREFLIGHT PASSED")
print(" 038 IS STATICALLY READY FOR ISAAC")
print("========================================")
