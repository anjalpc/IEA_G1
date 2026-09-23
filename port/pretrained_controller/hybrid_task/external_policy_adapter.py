from __future__ import annotations

import importlib
import os
import sys

import torch
from omegaconf import OmegaConf


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable is not set: {name}"
        )
    return value


def load_reference_policy(device: str):
    external_root = _require_env("EXTERNAL_POLICY_ROOT")
    external_package = _require_env("EXTERNAL_POLICY_PACKAGE")
    checkpoint_path = _require_env("EXTERNAL_POLICY_CHECKPOINT")
    config_path = _require_env("EXTERNAL_POLICY_CONFIG")

    if external_root not in sys.path:
        sys.path.insert(0, external_root)

    importlib.import_module(
        f"{external_package}.trl.modules.universal_token_modules"
    )

    common = importlib.import_module(
        f"{external_package}.trl.utils.common"
    )

    import trl
    import trl.trainer.utils
    from trl.experimental.ppo import ppo_trainer

    trl.trainer.utils.OnlineTrainerState = (
        ppo_trainer.OnlineTrainerState
    )
    trl.trainer.utils.exact_div = (
        ppo_trainer.exact_div
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        raise RuntimeError(
            "Policy state dictionary not found"
        )

    release_cfg = OmegaConf.load(config_path)

    actor_cfg = release_cfg.algo.config.actor
    algo_cfg = release_cfg.algo.config

    # Training-only auxiliary reconstruction is not needed for inference.
    actor_cfg.has_aux_loss = False
    actor_cfg.backbone.aux_loss_func = {}
    actor_cfg.backbone.aux_loss_coef = {}

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

    env_cfg = OmegaConf.create(
        {
            "robot": {
                "actions_dim": 29,
                "algo_obs_dim_dict": {
                    "actor_obs": actor_obs_dim,
                    "tokenizer": 774,
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
                    "tokenizer": 774,
                },
            },
        }
    )

    policy = common.custom_instantiate(
        actor_cfg,
        env_config=env_cfg,
        algo_config=algo_cfg,
        module_dim_dict={},
        backbone_kwargs={},
        _resolve=False,
    )

    policy.load_state_dict(
        state_dict,
        strict=True,
    )

    policy = policy.to(device)
    policy.eval()
    policy.init_rollout()

    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    return policy
