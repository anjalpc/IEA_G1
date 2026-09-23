"""Shared runtime helpers for exact per-environment stair geometry."""

from __future__ import annotations

import torch


def stair_cfg(env):
    return (
        env.scene.terrain.cfg.terrain_generator
        .sub_terrains["three_step_ascent"]
    )


def riser_height_per_env(env) -> torch.Tensor:
    """Return the exact riser height for every environment.

    Variable-height training uses quantized terrain difficulty bins.
    Terrain level therefore maps exactly to the generated riser height.
    Fixed-height evaluation falls back to the scalar configured height.
    """

    cfg = stair_cfg(env)

    low = float(cfg.riser_height_range[0])
    high = float(cfg.riser_height_range[1])

    dtype = env.scene.env_origins.dtype
    device = env.scene.env_origins.device

    if abs(high - low) <= 1.0e-12:
        return torch.full(
            (env.num_envs,),
            low,
            dtype=dtype,
            device=device,
        )

    levels = getattr(
        env.scene.terrain,
        "terrain_levels",
        None,
    )

    if levels is None:
        raise RuntimeError(
            "Variable stair heights require terrain_levels."
        )

    bins = int(getattr(cfg, "difficulty_bins", 1))

    if bins < 2:
        raise RuntimeError(
            "Variable stair heights require difficulty_bins >= 2."
        )

    return (
        low
        + levels.to(dtype=dtype)
        / float(bins - 1)
        * (high - low)
    )
