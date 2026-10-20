"""Prepare a behavior-preserving 485/29 -> 492/30 actor expansion from V07."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from simulation.isaac_lab.g1_stairs.reward_registry import materialize_legacy_reward_config
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import resolve_config


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / (
    "experiments/g1_stairs/logs/rsl_rl/"
    "unitree_iea_g1_29dof_stairs_v0/"
    "geometry_locomotion_v07_resume_high2000_20260929_145620/"
    "model_2130.pt"
)
SOURCE_SHA256 = "70f319de7e95ee3eaa68cd01e0dd57235efa6822aee4d1c51362d11137da5cee"
BASE = Path("experiments/g1_stairs/configs/geometry_locomotion_v07_resume_high2000.yaml")
OUTPUT = ROOT / (
    "experiments/g1_stairs/outputs/initialization/"
    "v07_model2130_actor492_actions30.pt"
)
CONTROL = ROOT / "experiments/g1_stairs/configs/v07_model2130_expanded492_control.yaml"

SOURCE_OBS = 485
TARGET_OBS = 492
SOURCE_ACTIONS = 29
TARGET_ACTIONS = 30
FREQUENCY_STD = 0.10


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


def expand_v07_actor(state: dict[str, torch.Tensor]):
    """Append seven zero-weight observations and one zero-mean cadence action."""
    keys = sorted(
        key for key in state
        if key.startswith("actor.") and key.endswith(".weight")
    )
    expected = [
        (512, SOURCE_OBS),
        (256, 512),
        (128, 256),
        (SOURCE_ACTIONS, 128),
    ]
    if [tuple(state[key].shape) for key in keys] != expected:
        raise ValueError("unexpected V07 actor structure")
    if tuple(state["std"].shape) != (SOURCE_ACTIONS,):
        raise ValueError("expected 29 V07 action standard deviations")
    if not torch.isfinite(state["std"]).all() or not (state["std"] > 0).all():
        raise ValueError("invalid V07 action standard deviations")

    expanded = {
        key: value.clone()
        for key, value in state.items()
        if key.startswith("actor.") or key == "std"
    }
    first, last = keys[0], keys[-1]
    last_bias = last.replace(".weight", ".bias")

    expanded[first] = torch.cat(
        (state[first], state[first].new_zeros((512, TARGET_OBS - SOURCE_OBS))),
        dim=1,
    )
    expanded[last] = torch.cat(
        (state[last], state[last].new_zeros((TARGET_ACTIONS - SOURCE_ACTIONS, 128))),
        dim=0,
    )
    expanded[last_bias] = torch.cat(
        (state[last_bias], state[last_bias].new_zeros(TARGET_ACTIONS - SOURCE_ACTIONS))
    )
    expanded["std"] = torch.cat(
        (state["std"], state["std"].new_tensor([FREQUENCY_STD]))
    )

    if not torch.equal(expanded[first][:, :SOURCE_OBS], state[first]):
        raise AssertionError("V07 actor input prefix changed")
    if torch.count_nonzero(expanded[first][:, SOURCE_OBS:]) != 0:
        raise AssertionError("new observation columns must start at zero")
    if not torch.equal(expanded[last][:SOURCE_ACTIONS], state[last]):
        raise AssertionError("V07 physical action rows changed")
    if torch.count_nonzero(expanded[last][SOURCE_ACTIONS:]) != 0:
        raise AssertionError("cadence mean weights must start at zero")
    if not torch.equal(expanded[last_bias][:SOURCE_ACTIONS], state[last_bias]):
        raise AssertionError("V07 physical action biases changed")
    if torch.count_nonzero(expanded[last_bias][SOURCE_ACTIONS:]) != 0:
        raise AssertionError("cadence mean bias must start at zero")
    if not torch.equal(expanded["std"][:SOURCE_ACTIONS], state["std"]):
        raise AssertionError("V07 physical action standard deviations changed")

    changed = {first, last, last_bias, "std"}
    for key in expanded:
        if key not in changed and not torch.equal(expanded[key], state[key]):
            raise AssertionError(f"unexpected actor mutation: {key}")
    return expanded, keys


def matched_output_check(source, expanded, keys):
    generator = torch.Generator().manual_seed(2130)
    dtype = source[keys[0]].dtype
    old_obs = torch.randn(256, SOURCE_OBS, generator=generator, dtype=dtype)
    new_obs = torch.randn(
        256, TARGET_OBS - SOURCE_OBS, generator=generator, dtype=dtype
    ) * 10.0
    with torch.inference_mode():
        reference = actor_forward(source, keys, old_obs)
        actual = actor_forward(expanded, keys, torch.cat((old_obs, new_obs), dim=1))
    torch.testing.assert_close(
        actual[:, :SOURCE_ACTIONS], reference, rtol=1e-5, atol=1e-5
    )
    if torch.count_nonzero(actual[:, SOURCE_ACTIONS:]) != 0:
        raise AssertionError("expanded cadence mean must be exactly zero")
    return float((actual[:, :SOURCE_ACTIONS] - reference).abs().max().item())


def control_config(*, output_hash: str):
    """Current-runtime control that changes architecture, not V07 task semantics."""
    return {
        "schema_version": 1,
        "inheritance": {
            "base_config": BASE.as_posix(),
            "resolved_config_required": True,
        },
        "experiment": {
            "id": CONTROL.stem,
            "purpose": (
                "Behavior-preserving V07 model2130 actor expansion from 485/29 "
                "to 492/30. Fixed 1 Hz cadence; no C7 strict gait or coordinated-motion objective."
            ),
        },
        "execution": {"maximum_iterations": 1},
        "observations": {
            "preferred_transition_foot": True,
            "preferred_foothold": False,
            "preferred_foothold_ablation": False,
            "coordinated_foothold_xyz": True,
            "gait_phase": True,
            "actor_height_scan": False,
            "critic_height_scan": True,
        },
        "actions": {
            "dimension": TARGET_ACTIONS,
            "physical_joint_dimension": SOURCE_ACTIONS,
        },
        "gait": {
            "enabled": True,
            "mode": "fixed",
            "coordinated_motion": False,
            "adaptive_timing": False,
            "frequency_std": FREQUENCY_STD,
        },
        "policy_initialization": {
            "type": "rsl_rl_checkpoint_actor_transfer",
            "source": {
                "task": "V07_model2130_expanded_for_492_30_control",
                "path": "/workspace/IEA_G1/" + str(OUTPUT.relative_to(ROOT)),
                "sha256": output_hash,
            },
            "locomotion_observations": 480,
            "explicit_geometry_observations": 5,
            "transition_state_observations": 2,
            "gait_phase_observations": 2,
            "coordinated_foothold_observations": 3,
            "stair_observations": TARGET_OBS,
            "actions": TARGET_ACTIONS,
            "new_observation_initialization": "zero",
            "critic_initialization": "fresh",
            "optimizer_initialization": "fresh",
            "pretrained_actor_trainable": True,
        },
        # gait_runtime requires this bookkeeping term when gait is enabled but
        # coordinated motion is disabled. It does not affect deterministic policy actions.
        "reward_experiment": {
            "rewards": {
                "stage_plan": {
                    "geometry_locomotion_v05b_finish_stabilization": {
                        "phase_swing_contact": -0.10,
                    }
                }
            }
        },
    }


def validate_control(path: Path):
    resolved, _ = resolve_config(path)
    resolved = materialize_legacy_reward_config(resolved)
    if resolved["policy_initialization"]["stair_observations"] != TARGET_OBS:
        raise AssertionError("expanded V07 actor must declare 492 observations")
    if resolved["policy_initialization"]["actions"] != TARGET_ACTIONS:
        raise AssertionError("expanded V07 actor must declare 30 outputs")
    if resolved["gait"]["mode"] != "fixed":
        raise AssertionError("initial V07 expansion control must hold cadence fixed")
    if resolved["gait"].get("coordinated_motion", False):
        raise AssertionError("initial V07 expansion control must not enable coordinated motion")
    if resolved.get("episode_contract", {}).get("strict_stair_gait", False):
        raise AssertionError("initial V07 expansion control must not add strict gait termination")

    base, _ = resolve_config(ROOT / BASE)
    base = materialize_legacy_reward_config(base)
    expected_weights = dict(active_reward_weights(base), phase_swing_contact=-0.10)
    if active_reward_weights(resolved) != expected_weights:
        raise AssertionError("V07 reward contract changed beyond gait bookkeeping")

    for section in (
        "terrain",
        "reset_distribution",
        "reset_randomization",
        "episode_contract",
        "acquisition_credit",
        "reward_parameters",
        "command",
    ):
        if resolved.get(section) != base.get(section):
            raise AssertionError(f"V07 control section drifted: {section}")
    return resolved


def main():
    for path in (OUTPUT, CONTROL):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("V07 model2130 SHA256 does not match the verified control")

    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if checkpoint.get("iter") != 2130:
        raise ValueError(f"expected source iteration 2130, got {checkpoint.get('iter')}")
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("V07 checkpoint is missing model_state_dict")

    expanded, keys = expand_v07_actor(source_state)
    error = matched_output_check(source_state, expanded, keys)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as stream:
        torch.save({
            "model_state_dict": expanded,
            "iter": 2130,
            "v07_492_30_initialization": {
                "source": str(SOURCE),
                "source_sha256": SOURCE_SHA256,
                "source_observations": SOURCE_OBS,
                "target_observations": TARGET_OBS,
                "source_actions": SOURCE_ACTIONS,
                "target_actions": TARGET_ACTIONS,
                "frequency_std": FREQUENCY_STD,
                "critic_initialization": "fresh",
                "optimizer_initialization": "fresh",
            },
        }, stream)
    output_hash = sha256(OUTPUT)

    config = control_config(output_hash=output_hash)
    with CONTROL.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    validate_control(CONTROL)

    print(f"PASS: V07 physical actor preserved; max numerical error={error:.9g}")
    print("PASS: seven new observation columns are zero initialized")
    print("PASS: cadence actor mean is zero; fixed gait runtime maps cadence to exactly 1 Hz")
    print("PASS: original 29 action std values preserved; cadence std=0.10")
    print("PASS: critic and optimizer are intentionally fresh")
    print("PASS: V07 task/reward/termination contract retained for expansion control")
    print("Source SHA256:", SOURCE_SHA256)
    print("Expanded SHA256:", output_hash)
    print("Checkpoint:", OUTPUT.relative_to(ROOT))
    print("Control config:", CONTROL.relative_to(ROOT))
    print("No training started.")


if __name__ == "__main__":
    main()
