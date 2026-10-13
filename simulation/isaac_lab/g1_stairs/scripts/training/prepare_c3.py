"""Prepare actor-only C3 initialization and matched smoke configs."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
    resolve_config,
    deep_merge,
)
from simulation.isaac_lab.g1_stairs.reward_registry import (
    materialize_legacy_reward_config,
)
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / (
    "experiments/g1_stairs/logs/rsl_rl/"
    "unitree_iea_g1_29dof_stairs_v0/"
    "g1_fresh_stageC1_transitionobs04_continue300_20261013_111708/"
    "model_309.pt"
)
BASE = Path(
    "experiments/g1_stairs/configs/"
    "g1_fresh_stageC1_transitionobs04_continue300.yaml"
)
OUTPUT = ROOT / (
    "experiments/g1_stairs/outputs/initialization/"
    "c3_model309_actor489_actions30.pt"
)
FREQUENCY_STD = 0.10


def sha256(path):
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


def main():
    config_paths = {
        mode: ROOT / f"experiments/g1_stairs/configs/c3_{mode}_smoke.yaml"
        for mode in ("fixed", "learned")
    }
    for path in (OUTPUT, *config_paths.values()):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if checkpoint.get("iter") != 309:
        raise ValueError(f"Expected iteration 309, got {checkpoint.get('iter')}")

    state = checkpoint["model_state_dict"]
    keys = sorted(
        key for key in state
        if key.startswith("actor.") and key.endswith(".weight")
    )
    expected = [(512, 487), (256, 512), (128, 256), (29, 128)]
    if [tuple(state[key].shape) for key in keys] != expected:
        raise ValueError("Unexpected source actor structure")
    if state["std"].shape != (29,):
        raise ValueError("Expected 29 source action standard deviations")
    if not torch.isfinite(state["std"]).all() or not (state["std"] > 0).all():
        raise ValueError("Invalid source standard deviations")

    expanded = {
        key: value.clone()
        for key, value in state.items()
        if key.startswith("actor.") or key == "std"
    }
    first, last = keys[0], keys[-1]
    last_bias = last.replace(".weight", ".bias")

    expanded[first] = torch.cat(
        (state[first], state[first].new_zeros((512, 2))), dim=1
    )
    expanded[last] = torch.cat(
        (state[last], state[last].new_zeros((1, 128))), dim=0
    )
    expanded[last_bias] = torch.cat(
        (state[last_bias], state[last_bias].new_zeros(1))
    )
    expanded["std"] = torch.cat(
        (state["std"], state["std"].new_tensor([FREQUENCY_STD]))
    )

    assert torch.equal(expanded[first][:, :487], state[first])
    assert torch.count_nonzero(expanded[first][:, 487:]) == 0
    assert torch.equal(expanded[last][:29], state[last])
    assert torch.count_nonzero(expanded[last][29:]) == 0
    assert expanded[last_bias][29].item() == 0
    assert torch.equal(expanded[last_bias][:29], state[last_bias])
    assert torch.equal(expanded["std"][:29], state["std"])

    changed = {first, last, last_bias, "std"}
    for key in expanded:
        if key not in changed:
            assert torch.equal(expanded[key], state[key]), key

    # Independent numerical check: arbitrary phase inputs must have no effect.
    generator = torch.Generator().manual_seed(123)
    dtype = state[first].dtype
    observations = torch.randn(256, 487, generator=generator, dtype=dtype)
    clocks = torch.randn(256, 2, generator=generator, dtype=dtype) * 10
    with torch.inference_mode():
        reference = actor_forward(state, keys, observations)
        actual = actor_forward(
            expanded, keys, torch.cat((observations, clocks), dim=1)
        )
    torch.testing.assert_close(
        actual[:, :29], reference, rtol=1e-5, atol=1e-5
    )
    assert torch.count_nonzero(actual[:, 29]) == 0
    error = (actual[:, :29] - reference).abs().max().item()
    print(f"PASS: Physical actor preserved; maximum numerical error={error:.9g}")
    print("PASS: New mean frequency action=0 -> deterministic frequency=1 Hz")

    base, _ = resolve_config(ROOT / BASE)
    stage = base["reward_experiment"]["active_stage"]
    source_hash = sha256(SOURCE)

    # Actor-only artifact: the normal actor-transfer loader creates a fresh
    # critic and optimizer. This is intentionally NOT a resumable checkpoint.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as stream:
        torch.save({
            "model_state_dict": expanded,
            "iter": 309,
            "c3_initialization": {
                "source": str(SOURCE),
                "source_sha256": source_hash,
                "frequency_std": FREQUENCY_STD,
                "critic_initialization": "fresh",
                "optimizer_initialization": "fresh",
            },
        }, stream)
    output_hash = sha256(OUTPUT)

    configs = {}
    for mode, path in config_paths.items():
        configs[mode] = {
            "schema_version": 1,
            "inheritance": {
                "base_config": BASE.as_posix(),
                "resolved_config_required": True,
            },
            "experiment": {
                "id": path.stem,
                "purpose": (
                    f"C3 {mode} cadence smoke from C1 model309. "
                    "Preserve B7 rewards and add phase swing-contact shaping."
                ),
            },
            "execution": {"maximum_iterations": 1},
            "observations": {
                "preferred_transition_foot": True,
                "preferred_foothold": False,
                "preferred_foothold_ablation": False,
                "gait_phase": True,
                "actor_height_scan": False,
                "critic_height_scan": True,
            },
            "actions": {
                "dimension": 30,
                "physical_joint_dimension": 29,
            },
            "acceptance": {"action_dimension": 30},
            "gait": {
                "enabled": True,
                "mode": mode,
                "frequency_std": FREQUENCY_STD,
            },
            "policy_initialization": {
                "type": "rsl_rl_checkpoint_actor_transfer",
                "source": {
                    "task": "C1_model309_expanded_for_C3",
                    "path": str(OUTPUT),
                    "sha256": output_hash,
                },
                "locomotion_observations": 480,
                "terrain_observations": 0,
                "actor_height_scan_observations": 0,
                "explicit_geometry_observations": 5,
                "transition_state_observations": 2,
                "gait_phase_observations": 2,
                "stair_observations": 489,
                "actions": 30,
                "new_observation_initialization": "zero",
                "critic_initialization": "fresh",
                "optimizer_initialization": "fresh",
                "pretrained_actor_trainable": True,
            },
            "reward_experiment": {
                "rewards": {
                    "stage_plan": {
                        stage: {"phase_swing_contact": -0.10}
                    }
                }
            },
        }

    baseline = materialize_legacy_reward_config(deep_merge({}, base))
    baseline_weights = active_reward_weights(baseline)
    expected_weights = dict(baseline_weights, phase_swing_contact=-0.10)

    for mode, config in configs.items():
        effective = deep_merge(base, config)
        effective = materialize_legacy_reward_config(effective)
        assert active_reward_weights(effective) == expected_weights
        for section in (
            "terrain", "ppo", "reset_distribution", "reset_randomization",
            "episode_contract", "reward_parameters", "command",
        ):
            assert effective.get(section) == base.get(section), section
        assert effective["execution"]["environments"] == base["execution"]["environments"]
        assert effective["observations"]["preferred_foothold"] is False

    from copy import deepcopy
    fixed_compare = deepcopy(configs["fixed"])
    learned_compare = deepcopy(configs["learned"])
    for config in (fixed_compare, learned_compare):
        config.pop("experiment")
        config["gait"].pop("mode")
    assert fixed_compare == learned_compare

    assert configs["fixed"]["gait"]["mode"] == "fixed"
    assert configs["learned"]["gait"]["mode"] == "learned"

    for mode, path in config_paths.items():
        with path.open("x") as stream:
            yaml.safe_dump(configs[mode], stream, sort_keys=False)
        effective, _ = resolve_config(path)
        assert effective["policy_initialization"]["stair_observations"] == 489
        assert effective["policy_initialization"]["actions"] == 30
        print("Config:", path.relative_to(ROOT))

    print("PASS: Matched configs; existing rewards/settings preserved.")
    print("Actor: 489; expected critic: 691; policy actions: 30; robot actions: 29")
    print("Source SHA256:", source_hash)
    print("Expanded SHA256:", output_hash)
    print("Checkpoint:", OUTPUT)
    print("No training started.")


if __name__ == "__main__":
    main()
