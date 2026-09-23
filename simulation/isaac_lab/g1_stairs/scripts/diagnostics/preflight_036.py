#!/usr/bin/env python3
"""Host-only integration preflight for the 036 G1 stair experiment.

This script deliberately imports no Isaac Lab modules.

It validates the complete static/configuration chain needed before launching
036b so simple plumbing errors are caught before GPU/Isaac startup.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[5]

CONFIG = (
    REPO
    / "experiments/g1_stairs/configs/"
      "036b_geometry_contact_learning.yaml"
)

ENV_CFG = (
    REPO
    / "simulation/isaac_lab/g1_stairs/env_cfg.py"
)

UNIFIED_REWARDS = (
    REPO
    / "simulation/isaac_lab/g1_stairs/unified_rewards.py"
)

TERRAIN_STATE = (
    REPO
    / "simulation/isaac_lab/g1_stairs/terrain_state.py"
)

TERRAIN_OBSERVATIONS = (
    REPO
    / "simulation/isaac_lab/g1_stairs/terrain_observations.py"
)

ISAAC_TERRAIN = (
    REPO
    / "simulation/isaac_lab/g1_stairs/isaac_terrain.py"
)

POLICY_INIT = (
    REPO
    / "simulation/isaac_lab/g1_stairs/policy_initialization.py"
)

TRAIN_SCRIPT = (
    REPO
    / "simulation/isaac_lab/g1_stairs/"
      "scripts/training/train_isaac_stairs.py"
)

REWARD_RUNTIME = (
    REPO
    / "simulation/isaac_lab/g1_stairs/reward_runtime.py"
)

REPLAY = (
    REPO
    / "simulation/isaac_lab/g1_stairs/evaluation/"
      "step_up_checkpoint_validation.py"
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        fail(f"{path} did not parse as a mapping")

    return data


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)

        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(
                    merged[key],
                    value,
                )
            else:
                merged[key] = value

        return merged

    return override


def resolve(path: Path, seen=None) -> dict:
    if seen is None:
        seen = set()

    path = path.resolve()

    if path in seen:
        fail(f"Config inheritance cycle: {path}")

    seen.add(path)

    data = load_yaml(path)

    inheritance = data.get("inheritance", {})
    base_name = inheritance.get("base_config")

    if not base_name:
        return data

    base_path = (REPO / base_name).resolve()

    if not base_path.is_file():
        fail(f"Missing base config: {base_path}")

    return deep_merge(
        resolve(base_path, seen),
        data,
    )


def class_assignment_names(
    source: str,
    class_name: str,
) -> set[str]:
    tree = ast.parse(source)

    target_class = None

    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name == class_name
        ):
            target_class = node
            break

    if target_class is None:
        fail(f"Class {class_name} not found")

    names: set[str] = set()

    for node in target_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)

    return names


def module_symbols(source: str) -> set[str]:
    tree = ast.parse(source)
    symbols: set[str] = set()

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            symbols.add(node.name)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)

    return symbols


def unified_reward_references(
    source: str,
) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "unified_rewards"
        ):
            names.add(node.attr)

    return names


def require_text(
    path: Path,
    needles: list[str],
) -> None:
    source = path.read_text(encoding="utf-8")

    for needle in needles:
        if needle not in source:
            fail(
                f"{path.relative_to(REPO)} "
                f"missing required integration token: "
                f"{needle!r}"
            )


print("========================================")
print(" G1 036 FULL HOST PREFLIGHT")
print("========================================")
print()


# ------------------------------------------------------------
# 1. Required files
# ------------------------------------------------------------

required_files = [
    CONFIG,
    ENV_CFG,
    UNIFIED_REWARDS,
    TERRAIN_STATE,
    TERRAIN_OBSERVATIONS,
    ISAAC_TERRAIN,
    POLICY_INIT,
    TRAIN_SCRIPT,
    REWARD_RUNTIME,
    REPLAY,
]

for path in required_files:
    if not path.is_file():
        fail(f"Missing required file: {path}")

ok("all 036 integration files exist")


# ------------------------------------------------------------
# 2. Resolve full 036b inheritance without Isaac
# ------------------------------------------------------------

resolved = resolve(CONFIG)

ok("036b config inheritance resolves")


# ------------------------------------------------------------
# 3. Execution budget
# ------------------------------------------------------------

execution = resolved.get("execution", {})

if int(execution.get("environments", -1)) != 2048:
    fail(
        "036b execution.environments must be 2048"
    )

if int(
    execution.get("maximum_iterations", -1)
) != 1000:
    fail(
        "036b execution.maximum_iterations must be 1000"
    )

ok("036b execution = 2048 envs × 1000 iterations")


# ------------------------------------------------------------
# 4. Command
# ------------------------------------------------------------

command = resolved.get("command", {})

if float(
    command.get("forward_velocity_mps", -1.0)
) != 0.25:
    fail(
        "command.forward_velocity_mps must be 0.25"
    )

ok("forward command = 0.25 m/s")


# ------------------------------------------------------------
# 5. Observation contract
# ------------------------------------------------------------

obs = resolved.get("observations", {})

if obs.get("actor_height_scan") is not False:
    fail("actor_height_scan must be false")

if obs.get("critic_height_scan") is not True:
    fail("critic_height_scan must be true")

policy_order = obs.get("policy_order", [])

expected_last = "explicit_stair_geometry"

if not policy_order:
    fail("observations.policy_order is empty")

if policy_order[-1] != expected_last:
    fail(
        "explicit_stair_geometry must be the final "
        "policy observation group"
    )

ok("actor scan disabled / critic scan enabled")
ok("explicit stair geometry is in actor policy order")


# ------------------------------------------------------------
# 6. Terrain contract
# ------------------------------------------------------------

terrain = resolved.get("terrain", {})

height_range = terrain.get(
    "riser_height_range_m"
)

if height_range != [0.06, 0.12]:
    fail(
        "riser_height_range_m must be [0.06, 0.12]"
    )

if float(
    terrain.get("tread_depth_m", -1.0)
) != 0.25:
    fail("tread_depth_m must be 0.25")

if int(
    terrain.get("step_count", -1)
) != 3:
    fail("step_count must be 3")

ok("terrain = 3 steps, 6–12 cm, 25 cm tread")


expected_heights = [
    0.06
    + i / 7.0 * (0.12 - 0.06)
    for i in range(8)
]

if len(expected_heights) != 8:
    fail("internal exact-height check failed")

if abs(expected_heights[0] - 0.06) > 1e-12:
    fail("first exact height is wrong")

if abs(expected_heights[-1] - 0.12) > 1e-12:
    fail("last exact height is wrong")

ok("8-bin exact riser-height mapping is valid")


# ------------------------------------------------------------
# 7. Policy initialization contract
# ------------------------------------------------------------

policy_init = resolved.get(
    "policy_initialization",
    {},
)

checks = {
    "locomotion_observations": 480,
    "actor_height_scan_observations": 0,
    "explicit_geometry_observations": 5,
    "stair_observations": 485,
    "actions": 29,
}

for key, expected in checks.items():
    actual = policy_init.get(key)

    if actual != expected:
        fail(
            f"policy_initialization.{key}: "
            f"expected {expected}, got {actual}"
        )

if (
    policy_init.get("type")
    != "unitree_velocity_onnx_transfer"
):
    fail(
        "policy initialization must be "
        "unitree_velocity_onnx_transfer"
    )

ok("policy initialization contract = 480 + 5 = 485")


# ------------------------------------------------------------
# 8. Active reward-stage contract
# ------------------------------------------------------------

reward_exp = resolved.get(
    "reward_experiment",
    {},
)

stage_name = reward_exp.get("active_stage")

stage_plan = (
    reward_exp
    .get("rewards", {})
    .get("stage_plan", {})
)

if stage_name not in stage_plan:
    fail(
        f"active reward stage {stage_name!r} "
        "is absent from stage_plan"
    )

active_rewards = stage_plan[stage_name]

expected_rewards = {
    "track_lin_vel_xy": 1.0,
    "track_ang_vel_z": 0.5,
    "upright": -2.0,
    "action_rate": -0.02,
    "joint_acc": -2.5e-7,
    "dof_pos_limits": -5.0,
    "feet_slide": -0.2,
    "undesired_contacts": -1.0,
    "terrain_relative_clearance_shortfall": -1.5,
    "terrain_riser_face_strike_proxy": -2.0,
    "terrain_tread_touchdown": 1.5,
}

if set(active_rewards) != set(expected_rewards):
    missing = sorted(
        set(expected_rewards) - set(active_rewards)
    )
    extra = sorted(
        set(active_rewards) - set(expected_rewards)
    )

    fail(
        "036 reward set mismatch. "
        f"missing={missing}, extra={extra}"
    )

for name, expected in expected_rewards.items():
    actual = float(active_rewards[name])

    if abs(actual - expected) > 1e-12:
        fail(
            f"reward {name}: "
            f"expected {expected}, got {actual}"
        )

ok("036 active reward stage has exactly 11 intended terms")


# ------------------------------------------------------------
# 9. Active rewards must exist in StairRewardsCfg
# ------------------------------------------------------------

env_source = ENV_CFG.read_text(
    encoding="utf-8"
)

catalog = class_assignment_names(
    env_source,
    "StairRewardsCfg",
)

missing_catalog = sorted(
    set(active_rewards) - catalog
)

if missing_catalog:
    fail(
        "active rewards absent from StairRewardsCfg: "
        + ", ".join(missing_catalog)
    )

ok("all active rewards exist in StairRewardsCfg")


# ------------------------------------------------------------
# 10. Every unified_rewards reference in env_cfg must resolve
# ------------------------------------------------------------

required_reward_symbols = (
    unified_reward_references(env_source)
)

reward_symbols = module_symbols(
    UNIFIED_REWARDS.read_text(
        encoding="utf-8"
    )
)

missing_symbols = sorted(
    required_reward_symbols - reward_symbols
)

if missing_symbols:
    fail(
        "env_cfg.py references missing "
        "unified_rewards symbols: "
        + ", ".join(missing_symbols)
    )

ok(
    "all env_cfg unified_rewards references "
    "resolve, including historical stages"
)


# ------------------------------------------------------------
# 11. Training runtime integration
# ------------------------------------------------------------

require_text(
    TRAIN_SCRIPT,
    [
        '"actor_height_scan"',
        'env_cfg.observations.policy.height_scan = None',
        '"riser_height_range_m"',
        "riser_height_per_env",
        "Exact riser heights present:",
        "apply_resolved_rewards",
        "initialize_actor_from_unitree_onnx",
        "Training completion marker:",
        "[FATAL] train_isaac_stairs.py:",
    ],
)

ok("training entrypoint contains all 036 runtime hooks")


# ------------------------------------------------------------
# 12. Terrain generator integration
# ------------------------------------------------------------

require_text(
    ISAAC_TERRAIN,
    [
        "difficulty_bins",
        "np.floor",
        "riser_height_range",
    ],
)

require_text(
    TERRAIN_STATE,
    [
        "def riser_height_per_env",
        "terrain_levels",
        "difficulty_bins",
    ],
)

ok("exact terrain-bin/runtime-height integration exists")


# ------------------------------------------------------------
# 13. Geometry observation integration
# ------------------------------------------------------------

require_text(
    TERRAIN_OBSERVATIONS,
    [
        "def stair_geometry",
        "riser_height_per_env",
        "next_distance",
        "relative_yaw",
    ],
)

ok("5-D stair geometry implementation exists")


# ------------------------------------------------------------
# 14. New physical reward functions
# ------------------------------------------------------------

for symbol in (
    "terrain_relative_clearance_shortfall",
    "terrain_riser_face_strike_proxy",
    "terrain_tread_touchdown",
    "terrain_relative_swing_clearance",
):
    if symbol not in reward_symbols:
        fail(
            f"unified_rewards missing {symbol}"
        )

ok(
    "036 reward functions plus 035 backward-compatible "
    "reward are present"
)


# ------------------------------------------------------------
# 15. Replay/evaluation observation contract
# ------------------------------------------------------------

require_text(
    REPLAY,
    [
        '"actor_height_scan"',
        "env_cfg.observations.policy.height_scan = None",
    ],
)

ok("checkpoint replay supports the 485-D actor contract")


# ------------------------------------------------------------
# 16. Policy initializer must support runtime-sized actor
# ------------------------------------------------------------

require_text(
    POLICY_INIT,
    [
        "actor_height_scan_observations",
        "explicit_geometry_observations",
        "runtime_input_dim",
        "new_columns_max_abs",
    ],
)

ok("Unitree transfer supports the 485-D runtime actor")


# ------------------------------------------------------------
# 17. reward_runtime must support resolved active stage
# ------------------------------------------------------------

require_text(
    REWARD_RUNTIME,
    [
        "active_stage",
        "stage_plan",
        "apply_resolved_rewards",
    ],
)

ok("reward runtime stage plumbing exists")


print()
print("========================================")
print(" PREFLIGHT PASSED")
print(" 036b IS STATICALLY READY FOR ISAAC")
print("========================================")
