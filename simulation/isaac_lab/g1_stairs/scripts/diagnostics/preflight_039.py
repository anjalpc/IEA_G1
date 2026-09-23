#!/usr/bin/env python3

from pathlib import Path
import ast
import yaml


ROOT = Path(__file__).resolve().parents[5]

CFG = (
    ROOT
    / "experiments/g1_stairs/configs/"
      "039b_upward_lift_discovery_learning.yaml"
)

ENV = ROOT / "simulation/isaac_lab/g1_stairs/env_cfg.py"
REW = ROOT / "simulation/isaac_lab/g1_stairs/unified_rewards.py"


def fail(message):
    raise SystemExit(f"[FAIL] {message}")


def merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)

        for key, value in override.items():
            result[key] = (
                merge(result[key], value)
                if key in result
                else value
            )

        return result

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


print("========================================")
print(" G1 039 FULL HOST PREFLIGHT")
print("========================================")

resolved = resolve(CFG)


# Frozen architecture.
init = resolved["policy_initialization"]

expected = {
    "locomotion_observations": 480,
    "actor_height_scan_observations": 0,
    "explicit_geometry_observations": 5,
    "stair_observations": 485,
    "actions": 29,
}

for key, expected_value in expected.items():
    if init.get(key) != expected_value:
        fail(
            f"{key}: expected {expected_value}, "
            f"got {init.get(key)}"
        )

if resolved["observations"]["actor_height_scan"] is not False:
    fail("actor scan changed")

if resolved["observations"]["critic_height_scan"] is not True:
    fail("critic scan changed")

print("[ OK ] frozen 485-D actor architecture")


# Terrain.
terrain = resolved["terrain"]

if terrain["riser_height_range_m"] != [0.06, 0.12]:
    fail("terrain riser range changed")

if int(terrain["step_count"]) != 3:
    fail("step count changed")

if float(terrain["tread_depth_m"]) != 0.25:
    fail("tread depth changed")

print("[ OK ] frozen 8-bin 6-12 cm terrain contract")


# Speed schedule.
schedule = resolved["command"]["stair_velocity_schedule"]

expected_schedule = {
    "enabled": True,
    "approach_velocity_mps": 0.50,
    "stair_velocity_mps": 0.10,
    "slowdown_start_distance_m": 0.60,
    "slowdown_end_distance_m": 0.30,
    "platform_velocity_mps": 0.50,
    "platform_accel_distance_m": 0.40,
}

if schedule != expected_schedule:
    fail(
        f"speed schedule mismatch: {schedule}"
    )

print("[ OK ] careful speed = 0.50 -> 0.10 -> 0.50 m/s")


# Active reward stage.
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
    "upright_forward_velocity_shortfall_careful",
    "stair_heading_alignment",
    "terrain_relative_swing_clearance_upward",
    "terrain_relative_clearance_shortfall",
    "terrain_riser_face_blocked",
    "terrain_tread_touchdown",
}

if set(stage) != expected_terms:
    fail(
        "039 reward set mismatch:\n"
        f"{sorted(stage)}"
    )

if float(
    stage["terrain_relative_swing_clearance_upward"]
) != 3.0:
    fail("upward clearance weight must be +3.0")

if float(stage["track_stair_velocity_world"]) != 1.0:
    fail("world velocity weight must be +1.0")

print("[ OK ] exact 039 reward stage")


# Catalog.
tree = ast.parse(
    ENV.read_text(encoding="utf-8")
)

cls = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "StairRewardsCfg"
)

catalog = {
    target.id
    for node in cls.body
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name)
}

for name in expected_terms:
    if name not in catalog:
        fail(
            f"{name} absent from StairRewardsCfg"
        )

print("[ OK ] all active 039 catalog fields exist")


# Callable.
reward_tree = ast.parse(
    REW.read_text(encoding="utf-8")
)

functions = {
    node.name
    for node in reward_tree.body
    if isinstance(node, ast.FunctionDef)
}

if "terrain_relative_swing_clearance_upward" not in functions:
    fail("new upward-clearance callable missing")

if "terrain_relative_swing_clearance_dynamic" not in functions:
    fail("038 dynamic reward was deleted")

if "upright_forward_velocity_shortfall" not in functions:
    fail("historical forward shortfall callable missing")

print("[ OK ] new callable present and historical callables preserved")


# Verify upward reward semantics.
source = REW.read_text(encoding="utf-8")

start = source.find(
    "def terrain_relative_swing_clearance_upward("
)

end = source.find(
    "\ndef ",
    start + 10,
)

if end < 0:
    end = len(source)

body = source[start:end]

for marker in (
    "riser_height_per_env(env)",
    "upward_gate",
    "motion_gate",
    "foot_vel[:, :, 2]",
):
    if marker not in body:
        fail(
            f"upward reward missing marker: {marker}"
        )

print("[ OK ] lift reward uses upward foot motion + exact riser height")


# Verify careful forward alias.
mapping_ok = False
minimum_ok = False

for node in cls.body:
    if not isinstance(node, ast.Assign):
        continue

    names = [
        target.id
        for target in node.targets
        if isinstance(target, ast.Name)
    ]

    if names != [
        "upright_forward_velocity_shortfall_careful"
    ]:
        continue

    if not isinstance(node.value, ast.Call):
        continue

    for kw in node.value.keywords:
        if kw.arg == "func":
            value = kw.value

            if (
                isinstance(value, ast.Attribute)
                and value.attr
                == "upright_forward_velocity_shortfall"
            ):
                mapping_ok = True

        if kw.arg == "params":
            if not isinstance(kw.value, ast.Dict):
                continue

            for key_node, value_node in zip(
                kw.value.keys,
                kw.value.values,
            ):
                if not (
                    isinstance(key_node, ast.Constant)
                    and key_node.value
                    == "minimum_forward_velocity_mps"
                ):
                    continue

                if not isinstance(
                    value_node,
                    ast.Constant,
                ):
                    continue

                if float(value_node.value) == 0.05:
                    minimum_ok = True

if not mapping_ok:
    fail("careful shortfall callable mapping incorrect")

if not minimum_ok:
    fail("careful shortfall minimum must be 0.05")

print("[ OK ] careful shortfall allows near-riser preparation")


print()
print("========================================")
print(" PREFLIGHT PASSED")
print(" 039 IS STATICALLY READY FOR ISAAC")
print("========================================")
