"""Prepare C7 actor expansion and matched strict-gait smoke/train configs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.reward_registry import materialize_legacy_reward_config
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
)


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / (
    "experiments/g1_stairs/logs/rsl_rl/"
    "unitree_iea_g1_29dof_stairs_v0/"
    "c6_4_centered_shaping_fullresume_continue1500_20261014_150920/"
    "model_2049.pt"
)
BASE = Path("experiments/g1_stairs/configs/c7_strict_gait_base.yaml")
OUTPUT = ROOT / (
    "experiments/g1_stairs/outputs/initialization/"
    "c7_model2049_actor492_actions30.pt"
)
SMOKE = ROOT / "experiments/g1_stairs/configs/c7_strict_gait_smoke.yaml"
TRAIN = ROOT / "experiments/g1_stairs/configs/c7_strict_gait_train1500.yaml"

SOURCE_OBS = 489
TARGET_OBS = 492
ACTIONS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_forward(state, keys, observations):
    x = observations
    for index, key in enumerate(keys):
        x = F.linear(x, state[key], state[key.replace(".weight", ".bias")])
        if index < len(keys) - 1:
            x = F.elu(x)
    return x


def expand_actor_observations(state: dict[str, torch.Tensor]):
    """Append three zero-weight foothold XYZ columns without changing outputs."""
    keys = sorted(
        key for key in state
        if key.startswith("actor.") and key.endswith(".weight")
    )
    expected = [(512, SOURCE_OBS), (256, 512), (128, 256), (ACTIONS, 128)]
    if [tuple(state[key].shape) for key in keys] != expected:
        raise ValueError("unexpected C6.4 actor structure")
    if tuple(state["std"].shape) != (ACTIONS,):
        raise ValueError("expected 30 action standard deviations")
    if not torch.isfinite(state["std"]).all() or not (state["std"] > 0).all():
        raise ValueError("invalid source action standard deviations")

    expanded = {
        key: value.clone()
        for key, value in state.items()
        if key.startswith("actor.") or key == "std"
    }
    first = keys[0]
    expanded[first] = torch.cat(
        (state[first], state[first].new_zeros((512, TARGET_OBS - SOURCE_OBS))),
        dim=1,
    )

    if tuple(expanded[first].shape) != (512, TARGET_OBS):
        raise AssertionError("C7 first actor layer expansion failed")
    if torch.count_nonzero(expanded[first][:, SOURCE_OBS:]) != 0:
        raise AssertionError("new C7 observation columns must start at zero")
    for key in expanded:
        if key != first and not torch.equal(expanded[key], state[key]):
            raise AssertionError(f"unexpected actor mutation: {key}")
    return expanded, keys


def matched_output_check(source, expanded, keys):
    generator = torch.Generator().manual_seed(2049)
    dtype = source[keys[0]].dtype
    old_obs = torch.randn(256, SOURCE_OBS, generator=generator, dtype=dtype)
    new_obs = torch.randn(
        256, TARGET_OBS - SOURCE_OBS, generator=generator, dtype=dtype
    ) * 10.0
    with torch.inference_mode():
        reference = actor_forward(source, keys, old_obs)
        actual = actor_forward(expanded, keys, torch.cat((old_obs, new_obs), dim=1))
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    return float((actual - reference).abs().max().item())


def leaf_config(*, name: str, output_hash: str, smoke: bool):
    terrain = (
        {
            "riser_heights_m": [0.04],
            "curriculum": {"enabled": False, "initial_max_level": 0},
        }
        if smoke
        else {
            "riser_heights_m": [0.04, 0.08, 0.12, 0.16, 0.20],
            "curriculum": {
                "enabled": True,
                "initial_max_level": 0,
                "platform_margin_m": 0.10,
                "minimum_upright_gravity_z": 0.75,
                "minimum_base_clearance_m": 0.55,
                "minimum_nonflat_env_fraction": 1.0,
                "require_strict_success_for_promotion": True,
            },
        }
    )
    execution = (
        {"environments": 32, "maximum_iterations": 2}
        if smoke else {"maximum_iterations": 1500}
    )
    return {
        "schema_version": 1,
        "inheritance": {
            "base_config": BASE.as_posix(),
            "resolved_config_required": True,
        },
        "experiment": {
            "id": name,
            "purpose": (
                "C7 strict-gait runtime smoke at fixed 4 cm; no learning claim."
                if smoke else
                "C7 strict alternating stair curriculum 4/8/12/16/20 cm from C6.4 model2049 actor."
            ),
        },
        "execution": execution,
        "terrain": terrain,
        "policy_initialization": {
            "type": "rsl_rl_checkpoint_actor_transfer",
            "source": {
                "task": "C6.4_model2049_expanded_for_C7",
                "path": "/workspace/IEA_G1/" + str(OUTPUT.relative_to(ROOT)),
                "sha256": output_hash,
            },
            "locomotion_observations": 480,
            "explicit_geometry_observations": 5,
            "transition_state_observations": 2,
            "coordinated_foothold_observations": 3,
            "gait_phase_observations": 2,
            "stair_observations": TARGET_OBS,
            "actions": ACTIONS,
            "new_observation_initialization": "zero",
            "critic_initialization": "fresh",
            "optimizer_initialization": "fresh",
            "pretrained_actor_trainable": True,
        },
    }


def validate_leaf(path: Path, *, smoke: bool):
    resolved, _ = resolve_config(path)
    resolved = materialize_legacy_reward_config(resolved)
    if resolved["policy_initialization"]["stair_observations"] != TARGET_OBS:
        raise AssertionError("C7 actor observation contract must be 492")
    if resolved["policy_initialization"]["actions"] != ACTIONS:
        raise AssertionError("C7 action contract must remain 30")
    if not resolved["observations"]["coordinated_foothold_xyz"]:
        raise AssertionError("C7 coordinated foothold observation is disabled")
    if not resolved["episode_contract"]["strict_stair_gait"]:
        raise AssertionError("C7 strict gait contract is disabled")
    if not resolved["acquisition_credit"]["require_first_touchdown_placement"]:
        raise AssertionError("C7 first touchdown placement gate is disabled")
    params = resolved["reward_parameters"]
    for name in ("terrain_relative_clearance", "forward_insertion"):
        if params[name]["tread_margin_m"] != 0.025:
            raise AssertionError(f"{name} tread margin drifted")
        if params[name]["insertion_target_m"] != 0.010:
            raise AssertionError(f"{name} centered insertion target drifted")
    if params["track_stair_axis_velocity"]["ascent_scale"] != 0.25:
        raise AssertionError("C7 ascent velocity gate drifted")
    weights = active_reward_weights(resolved)

    # Mirror reward_runtime.py preconditions before any Isaac launch.
    c4 = bool(resolved.get("gait", {}).get("coordinated_motion", False))
    expected_contact_weight = 0.0 if c4 else -0.10

    if weights.get("phase_swing_contact") != expected_contact_weight:
        raise AssertionError(
            "C7 phase_swing_contact disagrees with coordinated-motion mode"
        )

    if c4 and weights.get("coordinated_motion") != -1.0:
        raise AssertionError(
            "C7 coordinated_motion must be -1.0"
        )

    if not weights.get("terrain_relative_clearance"):
        raise AssertionError(
            "C7 requires terrain_relative_clearance"
        )

    overrides = resolved.get("reward_parameters", {}) or {}
    inactive_overrides = sorted(
        name
        for name, params in overrides.items()
        if params is not None and name not in weights
    )
    if inactive_overrides:
        raise AssertionError(
            "C7 has reward_parameters for inactive rewards: "
            + ", ".join(inactive_overrides)
        )

    required = {
        "geometry_failure_termination_event": -1.0,
        "first_touchdown_placement": 0.5,
        "preferred_transition_milestones": 1.0,
        "coordinated_motion": -1.0,
    }
    for name, value in required.items():
        if weights.get(name) != value:
            raise AssertionError(f"C7 reward mismatch: {name}={weights.get(name)}")
    heights = resolved["terrain"]["riser_heights_m"]
    expected = [0.04] if smoke else [0.04, 0.08, 0.12, 0.16, 0.20]
    if heights != expected:
        raise AssertionError(f"C7 terrain mismatch: {heights}")

    initial = int(resolved["terrain"]["curriculum"]["initial_max_level"])
    if not 0 <= initial < len(heights):
        raise AssertionError(
            f"C7 initial_max_level {initial} invalid for {len(heights)} heights"
        )
    if smoke and initial != 0:
        raise AssertionError("C7 smoke initial_max_level must be 0")

    return resolved


def main():
    for path in (OUTPUT, SMOKE, TRAIN):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if checkpoint.get("iter") != 2049:
        raise ValueError(f"expected source iteration 2049, got {checkpoint.get('iter')}")
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("source checkpoint is missing model_state_dict")

    expanded, keys = expand_actor_observations(source_state)
    error = matched_output_check(source_state, expanded, keys)
    source_hash = sha256(SOURCE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as stream:
        torch.save({
            "model_state_dict": expanded,
            "iter": 2049,
            "c7_initialization": {
                "source": str(SOURCE),
                "source_sha256": source_hash,
                "source_observations": SOURCE_OBS,
                "target_observations": TARGET_OBS,
                "critic_initialization": "fresh",
                "optimizer_initialization": "fresh",
            },
        }, stream)
    output_hash = sha256(OUTPUT)

    for path, smoke in ((SMOKE, True), (TRAIN, False)):
        config = leaf_config(name=path.stem, output_hash=output_hash, smoke=smoke)
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        validate_leaf(path, smoke=smoke)

    print(f"PASS: source actor preserved; max numerical error={error:.9g}")
    print("PASS: three new foothold XYZ actor columns are zero initialized")
    print("PASS: critic/optimizer intentionally fresh because observation size changed")
    print("PASS: smoke contract=4cm strict gait")
    print("PASS: train curriculum=4/8/12/16/20cm, strict-success promotion")
    print("Source SHA256:", source_hash)
    print("Expanded SHA256:", output_hash)
    print("Smoke:", SMOKE.relative_to(ROOT))
    print("Train:", TRAIN.relative_to(ROOT))


if __name__ == "__main__":
    main()
