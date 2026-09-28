"""Per-environment stair-height curriculum for geometry locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .terrain_state import riser_height_per_env, stair_cfg


def _counter_buffers(env, levels: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create persistent per-level attempt/success counters on demand."""

    attempts = getattr(env, "_g1_stair_curriculum_attempts", None)
    successes = getattr(env, "_g1_stair_curriculum_successes", None)

    if attempts is None or attempts.numel() != levels:
        attempts = torch.zeros(levels, dtype=torch.long, device=device)
        successes = torch.zeros(levels, dtype=torch.long, device=device)
        env._g1_stair_curriculum_attempts = attempts
        env._g1_stair_curriculum_successes = successes

    return attempts, successes


def stair_platform_levels(
    env,
    env_ids: Sequence[int],
    asset_name: str = "robot",
    platform_margin_m: float = 0.10,
    minimum_upright_gravity_z: float = 0.75,
    minimum_base_clearance_m: float = 0.55,
) -> dict[str, float]:
    """Promote only after a physically credible platform traversal.

    This is deliberately a curriculum criterion, not a reward.  It therefore
    does not prescribe a stepping sequence or add a hidden stair state machine.
    An environment moves up one exact riser-height row when the base reaches
    the upper platform, has risen with the terrain, and remains sufficiently
    upright.  Otherwise non-flat rows move down one level on reset.
    """

    if platform_margin_m < 0.0:
        raise ValueError("platform_margin_m must be >= 0")
    if not 0.0 < minimum_upright_gravity_z <= 1.0:
        raise ValueError(
            "minimum_upright_gravity_z must be in (0, 1]"
        )
    if minimum_base_clearance_m <= 0.0:
        raise ValueError("minimum_base_clearance_m must be > 0")

    if len(env_ids) == 0:
        terrain = env.scene.terrain
        return {
            "level_mean": float(terrain.terrain_levels.float().mean().item()),
            "reset_success_rate": 0.0,
        }

    terrain = env.scene.terrain
    robot = env.scene[asset_name]
    cfg = stair_cfg(env)

    env_ids = torch.as_tensor(
        env_ids,
        dtype=torch.long,
        device=terrain.terrain_levels.device,
    )

    old_levels = terrain.terrain_levels[env_ids].clone()
    riser_height = riser_height_per_env(env)[env_ids]

    local_x = (
        robot.data.root_pos_w[env_ids, 0]
        - env.scene.env_origins[env_ids, 0]
    )
    local_z = (
        robot.data.root_pos_w[env_ids, 2]
        - env.scene.env_origins[env_ids, 2]
    )

    platform_start = (
        float(cfg.start_clearance)
        + int(cfg.step_count) * float(cfg.tread_depth)
    )
    total_rise = int(cfg.step_count) * riser_height

    upright = (
        -robot.data.projected_gravity_b[env_ids, 2]
        >= minimum_upright_gravity_z
    )
    reached_platform = local_x >= (
        platform_start + platform_margin_m
    )
    rose_with_terrain = local_z >= (
        total_rise + minimum_base_clearance_m
    )

    completed_episode = env.episode_length_buf[env_ids] > 0
    success = (
        completed_episode
        & reached_platform
        & rose_with_terrain
        & upright
    )

    level_count = int(terrain.max_terrain_level)
    attempts, successes = _counter_buffers(
        env,
        level_count,
        old_levels.device,
    )

    attempts.scatter_add_(
        0,
        old_levels,
        completed_episode.to(torch.long),
    )
    successes.scatter_add_(0, old_levels, success.to(torch.long))

    # Keep environments that solve the hardest row at that row instead of
    # triggering TerrainImporter's default random wrap-around behavior.
    move_up = success & (old_levels < level_count - 1)
    move_down = (
        completed_episode
        & (~success)
        & (old_levels > 0)
    )

    # Use Isaac Lab's terrain importer to perform the actual per-environment
    # level/origin update.  This keeps the curriculum inside the manager-based
    # terrain workflow rather than modifying robot state from the policy loop.
    terrain.update_env_origins(
        env_ids,
        move_up=move_up,
        move_down=move_down,
    )

    return {
        "level_mean": float(
            terrain.terrain_levels.float().mean().item()
        ),
        "reset_success_rate": float(
            success.float().mean().item()
        ),
    }
