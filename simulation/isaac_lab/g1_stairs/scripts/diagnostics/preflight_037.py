#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[5]

CFG = (
    ROOT
    / "experiments/g1_stairs/configs/"
      "037b_forward_engagement_learning.yaml"
)

ENV = (
    ROOT
    / "simulation/isaac_lab/g1_stairs/env_cfg.py"
)

REWARDS = (
    ROOT
    / "simulation/isaac_lab/g1_stairs/unified_rewards.py"
)


def fail(message):
    raise SystemExit(
        f"[FAIL] {message}"
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
        path.read_text(
            encoding="utf-8"
        )
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


resolved = resolve(CFG)

print("========================================")
print(" G1 037 PREFLIGHT")
print("========================================")


# --------------------------------------------------
# Frozen architecture
# --------------------------------------------------

obs = resolved["observations"]
init = resolved["policy_initialization"]

assert obs["actor_height_scan"] is False
assert obs["critic_height_scan"] is True

assert init["locomotion_observations"] == 480
assert init["actor_height_scan_observations"] == 0
assert init["explicit_geometry_observations"] == 5
assert init["stair_observations"] == 485
assert init["actions"] == 29

print("[ OK ] frozen 485-D actor contract")


terrain = resolved["terrain"]

assert terrain["riser_height_range_m"] == [
    0.06,
    0.12,
]
assert terrain["tread_depth_m"] == 0.25
assert terrain["step_count"] == 3

print("[ OK ] frozen 8-bin 6-12 cm terrain contract")


# --------------------------------------------------
# 037 command
# --------------------------------------------------

command = resolved["command"]

if float(
    command["forward_velocity_mps"]
) != 0.50:
    fail("forward command is not 0.50 m/s")

print("[ OK ] command = 0.50 m/s")


# --------------------------------------------------
# Reward stage
# --------------------------------------------------

stage_name = (
    resolved["reward_experiment"]
    ["active_stage"]
)

stage = (
    resolved["reward_experiment"]
    ["rewards"]
    ["stage_plan"]
    [stage_name]
)

expected = {
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "upright",
    "action_rate",
    "joint_acc",
    "dof_pos_limits",
    "feet_slide",
    "undesired_contacts",
    "upright_forward_velocity_shortfall",
    "terrain_relative_clearance_shortfall",
    "terrain_riser_face_blocked",
    "terrain_tread_touchdown",
}

if set(stage) != expected:
    fail(
        "037 active reward set mismatch:\n"
        f"actual={sorted(stage)}\n"
        f"expected={sorted(expected)}"
    )

if float(
    stage["track_lin_vel_xy"]
) != 1.5:
    fail(
        "track_lin_vel_xy must be +1.5"
    )

if float(
    stage[
        "upright_forward_velocity_shortfall"
    ]
) != -1.0:
    fail(
        "forward shortfall must be -1.0"
    )

if float(
    stage["terrain_riser_face_blocked"]
) != -2.0:
    fail(
        "persistent riser term must be -2.0"
    )

print("[ OK ] 037 reward stage exact")


# --------------------------------------------------
# Catalog + reward symbols
# --------------------------------------------------

env_source = ENV.read_text(
    encoding="utf-8"
)

reward_source = REWARDS.read_text(
    encoding="utf-8"
)

env_tree = ast.parse(env_source)
reward_tree = ast.parse(reward_source)

catalog = set()

for node in env_tree.body:
    if (
        isinstance(node, ast.ClassDef)
        and node.name == "StairRewardsCfg"
    ):
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        catalog.add(
                            target.id
                        )

symbols = set()

for node in reward_tree.body:
    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        symbols.add(node.name)

    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                symbols.add(target.id)

for name in expected:
    if name.startswith("terrain_") or name.startswith("upright_forward"):
        if name not in catalog:
            fail(
                f"{name} absent from StairRewardsCfg"
            )

required_symbols = {
    "upright_forward_velocity_shortfall",
    "terrain_riser_face_strike",
    "terrain_relative_clearance_shortfall",
    "terrain_tread_touchdown",
}

for name in required_symbols:
    if name not in symbols:
        fail(
            f"{name} absent from unified_rewards.py"
        )

print("[ OK ] reward catalog and callable symbols")


print()
print("========================================")
print(" PREFLIGHT PASSED")
print(" 037 IS READY FOR ISAAC CONTRACT")
print("========================================")
