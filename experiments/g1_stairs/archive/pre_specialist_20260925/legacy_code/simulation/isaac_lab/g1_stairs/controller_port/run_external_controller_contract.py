"""Qualification of an external pretrained terrain controller."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

print("[PORT] 01 script started", flush=True)

from isaaclab.app import AppLauncher

print("[PORT] 02 AppLauncher imported", flush=True)

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[PORT] 03 launching Isaac application", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[PORT] 04 Isaac application launched", flush=True)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def main() -> None:
    import torch
    from omegaconf import OmegaConf

    external_root = require_env("EXTERNAL_POLICY_ROOT")
    external_package = require_env("EXTERNAL_POLICY_PACKAGE")
    checkpoint_path = require_env("EXTERNAL_POLICY_CHECKPOINT")
    config_path = require_env("EXTERNAL_POLICY_CONFIG")

    if external_root not in sys.path:
        sys.path.insert(0, external_root)

    print("[PORT] 05 external runtime configured", flush=True)

    for required_path in (checkpoint_path, config_path):
        if not os.path.isfile(required_path):
            raise FileNotFoundError(required_path)

    print("[PORT] 06 checkpoint and config found", flush=True)

    print("[PORT] 07 importing policy implementation", flush=True)

    actor_module = importlib.import_module(
        f"{external_package}.trl.modules.actor_critic_modules"
    )
    importlib.import_module(
        f"{external_package}.trl.modules.universal_token_modules"
    )

    common = importlib.import_module(
        f"{external_package}.trl.utils.common"
    )

    print("[PORT] 08 policy implementation imported", flush=True)

    print("[PORT] 09 preparing checkpoint compatibility", flush=True)

    import trl
    import trl.trainer.utils
    from trl.experimental.ppo import ppo_trainer

    trl.trainer.utils.OnlineTrainerState = ppo_trainer.OnlineTrainerState
    trl.trainer.utils.exact_div = ppo_trainer.exact_div

    print("[PORT] 10 loading checkpoint", flush=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "actor_model_state_dict" in checkpoint:
        state_key = "actor_model_state_dict"
    elif "policy_state_dict" in checkpoint:
        state_key = "policy_state_dict"
    else:
        raise RuntimeError("Policy state dictionary not found in checkpoint")

    state_dict = checkpoint[state_key]

    print("[PORT] 11 checkpoint loaded", flush=True)

    print("[PORT] 12 loading released model configuration", flush=True)

    release_cfg = OmegaConf.load(config_path)

    actor_cfg = release_cfg.algo.config.actor
    algo_cfg = release_cfg.algo.config

    # Inference does not require the training-only auxiliary reconstruction
    # loss. Its implementation pulls in the offline motion-processing stack.
    # Disable only auxiliary-loss construction while preserving the released
    # encoder, quantizer, decoders, action head, and checkpoint parameters.
    actor_cfg.has_aux_loss = False
    actor_cfg.backbone.aux_loss_func = {}
    actor_cfg.backbone.aux_loss_coef = {}

    print("[PORT] 13 released model configuration loaded", flush=True)
    print("[PORT]     training-only auxiliary losses disabled", flush=True)

    #
    # Runtime-generated observation dimensions are not persisted in the
    # released YAML. Reconstruct them from the documented observation
    # contract and verify them against the checkpoint architecture.
    #
    actor_obs_dim = 1029

    tokenizer_names = [
        "encoder_index",
        "command_multi_future_nonflat",
        "command_z_multi_future_nonflat",
        "motion_anchor_ori_b_mf_nonflat",
        "height_map_z_flat",
    ]

    tokenizer_dims = {
        "encoder_index": [3],
        "command_multi_future_nonflat": [10, 58],
        "command_z_multi_future_nonflat": [10, 1],
        "motion_anchor_ori_b_mf_nonflat": [10, 6],
        "height_map_z_flat": [121],
    }

    tokenizer_total_dim = sum(
        int(torch.tensor(shape).prod().item())
        for shape in tokenizer_dims.values()
    )

    if tokenizer_total_dim != 774:
        raise RuntimeError(
            f"Unexpected tokenizer dimension: {tokenizer_total_dim}"
        )

    env_cfg = OmegaConf.create(
        {
            "robot": {
                "actions_dim": 29,
                "algo_obs_dim_dict": {
                    "actor_obs": actor_obs_dim,
                    "tokenizer": tokenizer_total_dim,
                },
            },
            "obs": {
                "group_obs_dims": {
                    "tokenizer": tokenizer_dims,
                },
                "group_obs_names": {
                    "tokenizer": tokenizer_names,
                },
                "obs_dims": {
                    "actor_obs": actor_obs_dim,
                    "tokenizer": tokenizer_total_dim,
                },
            },
        }
    )

    print("[PORT] 14 reconstructed runtime dimensions", flush=True)
    print(f"[PORT]     actor_obs={actor_obs_dim}")
    print(f"[PORT]     tokenizer={tokenizer_total_dim}")
    print(f"[PORT]     actions=29")

    #
    # Hard architecture checks against checkpoint tensor shapes.
    #
    encoder_key = "actor_module.encoders.g1.module.0.weight"
    decoder_key = "actor_module.decoders.g1_dyn.module.0.weight"

    if encoder_key not in state_dict:
        raise RuntimeError(f"Checkpoint missing {encoder_key}")

    if decoder_key not in state_dict:
        raise RuntimeError(f"Checkpoint missing {decoder_key}")

    encoder_input = int(state_dict[encoder_key].shape[1])
    decoder_input = int(state_dict[decoder_key].shape[1])

    if encoder_input != 1920:
        raise RuntimeError(
            f"Expected encoder input 1920, got {encoder_input}"
        )

    if decoder_input != 1093:
        raise RuntimeError(
            f"Expected decoder input 1093, got {decoder_input}"
        )

    if decoder_input - 64 != actor_obs_dim:
        raise RuntimeError(
            "Actor observation reconstruction does not match checkpoint"
        )

    print("[PORT] 15 checkpoint architecture dimensions verified", flush=True)

    print("[PORT] 16 instantiating released Actor", flush=True)

    policy = common.custom_instantiate(
        actor_cfg,
        env_config=env_cfg,
        algo_config=algo_cfg,
        module_dim_dict={},
        backbone_kwargs={},
        _resolve=False,
    )

    print("[PORT] 17 Actor instantiated", flush=True)

    model_state = policy.state_dict()

    print("[PORT] 18 comparing model and checkpoint state dictionaries", flush=True)

    model_keys = set(model_state.keys())
    checkpoint_keys = set(state_dict.keys())

    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)

    shape_mismatches = []

    for key in sorted(model_keys & checkpoint_keys):
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape):
            shape_mismatches.append(
                (
                    key,
                    tuple(model_state[key].shape),
                    tuple(state_dict[key].shape),
                )
            )

    print(f"[PORT]     model tensors={len(model_state)}")
    print(f"[PORT]     checkpoint tensors={len(state_dict)}")
    print(f"[PORT]     missing={len(missing)}")
    print(f"[PORT]     unexpected={len(unexpected)}")
    print(f"[PORT]     shape_mismatches={len(shape_mismatches)}")

    if missing:
        print("[PORT] missing keys:")
        for key in missing:
            print(f"  {key}")

    if unexpected:
        print("[PORT] unexpected keys:")
        for key in unexpected:
            print(f"  {key}")

    if shape_mismatches:
        print("[PORT] shape mismatches:")
        for key, model_shape, checkpoint_shape in shape_mismatches:
            print(
                f"  {key}: model={model_shape}, "
                f"checkpoint={checkpoint_shape}"
            )

    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Released Actor architecture does not exactly match checkpoint"
        )

    print("[PORT] 19 performing strict policy-state load", flush=True)

    policy.load_state_dict(
        state_dict,
        strict=True,
    )

    policy.eval()

    print("[PORT] 20 strict policy-state load passed", flush=True)

    total_parameters = sum(p.numel() for p in policy.parameters())

    #
    # Gate 2C: deterministic neural-controller forward pass.
    #
    # actor_obs layout:
    #   base_ang_vel:                    3 x 10 = 30
    #   joint_pos:                      29 x 10 = 290
    #   joint_vel:                      29 x 10 = 290
    #   previous actions:               29 x 10 = 290
    #   gravity_dir:                     3 x 10 = 30
    #   object_pos_b:                            = 3
    #   object_ori_b_6d:                         = 6
    #   object_pos_delta_multi_future:   3 x 10 = 30
    #   object_ori_delta_multi_future:   6 x 10 = 60
    #                                              ----
    #                                              1029
    #
    print("[PORT] 21 building contract-smoke observation", flush=True)

    actor_obs = torch.zeros((1, 1, actor_obs_dim), dtype=torch.float32)

    # Neutral projected gravity for each of the ten history frames.
    gravity_start = 30 + 290 + 290 + 290
    gravity = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=torch.float32,
    ).repeat(10)
    actor_obs[0, 0, gravity_start : gravity_start + 30] = gravity

    # Tokenizer terms in the exact registered order.
    encoder_index = torch.tensor(
        [1.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    command_future = torch.zeros(
        (10, 58),
        dtype=torch.float32,
    )

    command_z_future = torch.zeros(
        (10, 1),
        dtype=torch.float32,
    )

    # Identity rotation in 6D representation:
    # first two columns of a 3x3 identity rotation matrix.
    identity_rot_6d = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    anchor_orientation_future = identity_rot_6d.repeat(10, 1)

    height_map = torch.zeros(
        (121,),
        dtype=torch.float32,
    )

    tokenizer = torch.cat(
        [
            encoder_index.reshape(-1),
            command_future.reshape(-1),
            command_z_future.reshape(-1),
            anchor_orientation_future.reshape(-1),
            height_map.reshape(-1),
        ],
        dim=0,
    )

    if tokenizer.numel() != tokenizer_total_dim:
        raise RuntimeError(
            f"Tokenizer construction mismatch: "
            f"{tokenizer.numel()} != {tokenizer_total_dim}"
        )

    obs_dict = {
        "actor_obs": actor_obs,
        "tokenizer": tokenizer.reshape(1, 1, -1),
    }

    print("[PORT] 22 running deterministic policy forward", flush=True)

    with torch.no_grad():
        action_1 = policy(obs_dict)
        action_2 = policy(obs_dict)

    print("[PORT] 23 policy forward completed", flush=True)

    if tuple(action_1.shape) != (1, 1, 29):
        raise RuntimeError(
            f"Unexpected action shape: {tuple(action_1.shape)}"
        )

    if not torch.isfinite(action_1).all():
        raise RuntimeError("Policy produced NaN or Inf actions")

    if not torch.equal(action_1, action_2):
        max_delta = torch.max(torch.abs(action_1 - action_2)).item()
        raise RuntimeError(
            f"Policy forward is not deterministic; max_delta={max_delta}"
        )

    action_flat = action_1.reshape(-1)

    print("[PORT] 24 deterministic output verified", flush=True)
    print(
        "[PORT]     action_min="
        f"{action_flat.min().item():.6f}"
    )
    print(
        "[PORT]     action_max="
        f"{action_flat.max().item():.6f}"
    )
    print(
        "[PORT]     action_mean="
        f"{action_flat.mean().item():.6f}"
    )
    print(
        "[PORT]     action_l2="
        f"{torch.linalg.vector_norm(action_flat).item():.6f}"
    )

    print("[PORT]     action_vector:")
    print(
        "           "
        + " ".join(f"{x:.6f}" for x in action_flat.tolist())
    )

    print()
    print("=" * 80)
    print("GATE 2B RESULT")
    print("=" * 80)
    print("actor_obs_dim:", actor_obs_dim)
    print("tokenizer_dim:", tokenizer_total_dim)
    print("action_dim:", 29)
    print("encoder_input_dim:", encoder_input)
    print("decoder_input_dim:", decoder_input)
    print("policy_parameters:", total_parameters)
    print("checkpoint_tensors:", len(state_dict))
    print()
    print("PASS: exact released Actor architecture instantiated")
    print("PASS: checkpoint matched with zero missing tensors")
    print("PASS: checkpoint matched with zero unexpected tensors")
    print("PASS: checkpoint matched with zero shape mismatches")
    print("PASS: strict policy-state load completed")

    print()
    print("=" * 80)
    print("GATE 2C RESULT")
    print("=" * 80)
    print("input_actor_obs:", tuple(actor_obs.shape))
    print("input_tokenizer:", tuple(obs_dict["tokenizer"].shape))
    print("output_actions:", tuple(action_1.shape))
    print()
    print("PASS: exact released policy executed a forward pass")
    print("PASS: policy output contains 29 finite actions")
    print("PASS: repeated deterministic forward is identical")


try:
    main()

except BaseException:
    print()
    print("[PORT] FAILURE", flush=True)
    traceback.print_exc()
    raise

finally:
    print("[PORT] closing Isaac application", flush=True)
    simulation_app.close()
