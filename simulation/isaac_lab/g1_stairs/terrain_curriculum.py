"""Per-environment stair-height curriculum for geometry locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .clean_climb import VIOLATIONS
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
    require_episode_success: bool = False,
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
    if not isinstance(require_episode_success, bool):
        raise ValueError("require_episode_success must be boolean")

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
    if require_episode_success:
        runtime = getattr(env, "_g1_geometry_episode", None)
        if runtime is None:
            raise RuntimeError(
                "strict curriculum promotion requires geometry episode runtime"
            )
        from .geometry_episode import Result
        success &= runtime.tracker.reason[env_ids] == Result.SUCCESS

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


# --- Patch 57: clean-climb curriculum ---------------------------------------
#
# Every env has a fixed role by env id: one in four walks flat (level 0, never
# moves), one in four reviews a height it has mastered (a new one at every
# reset), two in four climb at their own frontier. A frontier env moves up one
# height after a clean climb (clean_climb: plan complete, every step on the
# first touchdown, no toe strike, no nose landing, no failure, upright) and
# down one after an episode without a completed climb. A climb with a fault
# keeps the height. The frontier stays within ``max_levels_ahead`` of the
# highest mastered height; a height is mastered when its windowed clean rate
# reaches ``mastered_clean_rate`` (every height below it too); the mastered
# height never goes down. Isaac Lab's update_env_origins is not used: it
# sends envs that solve the last row to a random row.

CLEAN_CLIMB_ROLES = ("flat", "review", "frontier", "frontier")
FLAT, REVIEW, FRONTIER = 0, 1, 2
CLEAN_CLIMB_WINDOW = 500.0      # episodes per height in the clean-rate window
CLEAN_CLIMB_MIN_ATTEMPTS = 100  # window episodes before a height can count as mastered
REVIEW_WEIGHT_FLOOR = 0.1      # review weight of a height at clean rate 1 (patch 62)
CLEAN_CLIMB_KEYS = {"enabled", "type", "initial_level", "mastered_level",
                    "mastered_clean_rate", "max_levels_ahead"}
# Episode outcome counters (stair episodes, windowed), logged as shares.
_OUTCOMES = ("episodes", "clean", *VIOLATIONS, "failed", "not_climbed")


def validate_clean_climb_cfg(cfg, heights) -> dict:
    """Normalised ``terrain.curriculum`` for ``type: clean_climb``."""
    unknown = set(cfg) - CLEAN_CLIMB_KEYS
    if unknown:
        raise ValueError(f"unknown clean_climb curriculum keys: {sorted(unknown)}")
    missing = CLEAN_CLIMB_KEYS - set(cfg)
    if missing:
        raise ValueError(f"clean_climb curriculum needs keys {sorted(missing)}")
    heights = tuple(float(h) for h in heights)
    if len(heights) < 2 or heights[0] != 0.0:
        raise ValueError("clean_climb curriculum needs riser_heights_m starting at 0.0 (flat) "
                         "and at least one stair height")
    top = len(heights) - 1
    mastered = int(cfg["mastered_level"])
    initial = int(cfg["initial_level"])
    ahead = int(cfg["max_levels_ahead"])
    rate = float(cfg["mastered_clean_rate"])
    if not 1 <= mastered <= top:
        raise ValueError("terrain.curriculum.mastered_level must index a stair height")
    if not 1 <= ahead <= top:
        raise ValueError("terrain.curriculum.max_levels_ahead must be in [1, stair heights]")
    if not 1 <= initial <= min(top, mastered + ahead):
        raise ValueError("terrain.curriculum.initial_level must be a stair height "
                         "at most max_levels_ahead above mastered_level")
    if not 0.0 < rate <= 1.0:
        raise ValueError("terrain.curriculum.mastered_clean_rate must be in (0, 1]")
    return {"initial_level": initial, "mastered_level": mastered,
            "mastered_clean_rate": rate, "max_levels_ahead": ahead}


class CleanClimbState:
    """Roles, mastered height and windowed per-height statistics."""

    def __init__(self, num_envs, levels, mastered_level, device):
        ids = torch.arange(num_envs, device=device)
        self.role = torch.tensor([{"flat": FLAT, "review": REVIEW, "frontier": FRONTIER}[r]
                                  for r in CLEAN_CLIMB_ROLES], device=device)[ids % len(CLEAN_CLIMB_ROLES)]
        self.assigned = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.mastered = torch.tensor(int(mastered_level), device=device)
        self.mastered_initial = int(mastered_level)
        self.attempts = torch.zeros(levels, device=device)
        self.clean = torch.zeros(levels, device=device)
        self.outcomes = torch.zeros(len(_OUTCOMES), device=device)
        self.heights = None             # riser height per level, set by clean_climb_levels


def _window(values, total, limit):
    """Scale ``values`` so that ``total`` stays <= limit (sliding window)."""
    return values * (limit / total.clamp_min(limit))


def compute_mastered(attempts, clean, current, initial, rate, minimum=CLEAN_CLIMB_MIN_ATTEMPTS):
    """Highest level L such that every stair level 1..L is mastered: enough
    window episodes and a clean rate >= ``rate``, or no data yet and L <=
    ``initial``. Never below ``current`` (tensor scalar)."""
    levels = attempts.numel()
    index = torch.arange(levels, device=attempts.device)
    known = attempts >= minimum
    good = torch.where(known, clean >= rate * attempts, index <= initial)
    good[0] = True
    # First failing level; everything below it is mastered.
    failing = torch.where(~good, index, torch.full_like(index, levels))
    reached = failing.min() - 1
    return torch.maximum(current, reached.clamp(1, levels - 1))


def review_weights(attempts, clean, mastered, minimum=CLEAN_CLIMB_MIN_ATTEMPTS,
                   floor=REVIEW_WEIGHT_FLOOR):
    """[levels] review sampling weight: 1 - windowed clean rate + ``floor`` on
    levels 1..mastered (a level with fewer than ``minimum`` window episodes
    counts as rate 0), 0 elsewhere.

    Patch 62: uniform review let the low heights slip while the frontier
    moved up (run 20261111_151604 model_44200 vs 20261111_084902 model_35200,
    same robots: 6 cm strict 10/10 -> 4/10, 8 cm 9/10 -> 4/10, tread heels
    2.5 cm shorter at 4-8 cm; training clean 6/8 cm 0.83/0.85 -> 0.61/0.63
    while 14 cm became mastered). Weak heights now get more review episodes;
    the floor keeps every mastered height in practice."""
    rate = torch.where(attempts >= minimum, clean / attempts.clamp_min(1e-6),
                       torch.zeros_like(attempts))
    level = torch.arange(attempts.shape[0], device=attempts.device)
    keep = (level >= 1) & (level <= mastered)
    return torch.where(keep, (1.0 - rate).clamp_min(0.0) + floor, torch.zeros_like(rate))


def next_levels(level, role, success, climbed, completed, mastered, top, ahead, weights=None):
    """New level per env from its role and episode outcome.

    flat: 0. review: drawn from 1..mastered, uniformly or by ``weights``
    ([levels], zero outside 1..mastered). frontier: +1 after a clean climb,
    -1 after an episode without a completed climb, within
    [1, min(top, mastered + ahead)].
    """
    cap = torch.clamp(mastered + ahead, max=top)
    step = success.long() - (completed & ~climbed).long()
    frontier = torch.minimum(torch.clamp(level + step, min=1), cap)
    if weights is None:
        review = torch.randint(0, 1 << 30, level.shape, device=level.device) % mastered + 1
    else:
        review = torch.multinomial(weights, level.numel(), replacement=True).view_as(level)
    return torch.where(role == FLAT, torch.zeros_like(level),
                       torch.where(role == REVIEW, review, frontier))


def clean_climb_levels(
    env,
    env_ids: Sequence[int],
    initial_level: int,
    mastered_level: int,
    mastered_clean_rate: float,
    max_levels_ahead: int,
    asset_name: str = "robot",
    minimum_upright_gravity_z: float = 0.75,
) -> dict[str, float]:
    """Curriculum term (patch 57); runs first in ``_reset_idx``, so the
    episode's clock state, terminations and robot pose are still those of
    the episode that just ended."""
    terrain = env.scene.terrain
    levels = int(terrain.max_terrain_level)
    device = terrain.terrain_levels.device
    state = getattr(env, "_g1_clean_climb", None)
    if state is None:
        state = CleanClimbState(env.num_envs, levels, mastered_level, device)
        state.heights = torch.tensor(stair_cfg(env).riser_heights, dtype=torch.float32, device=device)
        env._g1_clean_climb = state
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=device)[env_ids]
    else:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    top = levels - 1
    old = terrain.terrain_levels[ids].clone()
    role = state.role[ids]
    # Envs reset for the first time (the start of training) only take their
    # role's starting height: they have no episode to judge.
    first = ~state.assigned[ids]
    judged = ~first & (env.episode_length_buf[ids] > 0) & (old > 0)
    clock = getattr(env, "_g1_rolling_clock", None)
    if clock is None:
        if bool(judged.any()):
            raise RuntimeError("clean_climb curriculum needs footstep_clock (rolling clock state)")
        climbed = violations = None
    else:
        climbed = clock.climbed[ids] & judged
        violations = clock.violations[ids] & judged[:, None]
    if climbed is None:
        climbed = torch.zeros_like(judged)
        violations = torch.zeros((ids.numel(), len(VIOLATIONS)), dtype=torch.bool, device=device)
    failed = env.termination_manager.terminated[ids] & judged
    upright = -env.scene[asset_name].data.projected_gravity_b[ids, 2] >= minimum_upright_gravity_z
    success = judged & climbed & ~violations.any(dim=-1) & ~failed & upright

    # Windowed statistics per height (frontier and review episodes alike).
    attempts = torch.zeros(levels, device=device).scatter_add_(0, old, judged.float())
    clean = torch.zeros(levels, device=device).scatter_add_(0, old, success.float())
    state.attempts += attempts
    state.clean += clean
    scale = CLEAN_CLIMB_WINDOW / state.attempts.clamp_min(CLEAN_CLIMB_WINDOW)
    state.attempts *= scale
    state.clean *= scale
    outcome = torch.stack((
        judged.float().sum(), success.float().sum(),
        *(violations[:, k].float().sum() for k in range(len(VIOLATIONS))),
        failed.float().sum(), (judged & ~climbed).float().sum()))
    state.outcomes += outcome
    state.outcomes = _window(state.outcomes, state.outcomes[0], CLEAN_CLIMB_WINDOW * levels)
    state.mastered = compute_mastered(state.attempts, state.clean, state.mastered,
                                      state.mastered_initial, mastered_clean_rate)

    new = next_levels(old, role, success, climbed, judged, state.mastered, top, max_levels_ahead,
                      review_weights(state.attempts, state.clean, state.mastered))
    start = torch.where(role == FLAT, torch.zeros_like(old),
                        torch.where(role == REVIEW, new,
                                    torch.full_like(old, min(initial_level, top))))
    new = torch.where(first, start, new)
    state.assigned[ids] = True
    terrain.terrain_levels[ids] = new
    terrain.env_origins[ids] = terrain.terrain_origins[new, terrain.terrain_types[ids]]

    # Cumulative counters read by isaac_training_metrics (same names as
    # stair_platform_levels).
    total, successes = _counter_buffers(env, levels, device)
    total += attempts.round().long()
    successes += clean.round().long()
    return clean_climb_log(env, state)


def clean_climb_log(env, state) -> dict[str, float]:
    """Logged values (one GPU sync)."""
    terrain = env.scene.terrain
    heights = state.heights
    frontier = heights[terrain.terrain_levels[state.role == FRONTIER]]
    if frontier.numel() == 0:               # fewer than three envs
        frontier = heights[:1]
    packed = torch.cat((
        torch.stack((frontier.mean(), frontier.max(), heights[state.mastered])),
        state.attempts, state.clean, state.outcomes)).tolist()
    levels = state.attempts.numel()
    attempts = packed[3:3 + levels]
    clean = packed[3 + levels:3 + 2 * levels]
    outcomes = dict(zip(_OUTCOMES, packed[3 + 2 * levels:]))
    log = {"frontier_cm_mean": 100.0 * packed[0], "frontier_cm_max": 100.0 * packed[1],
           "mastered_cm": 100.0 * packed[2]}
    episodes = outcomes.pop("episodes")
    if episodes > 0:
        for name, value in outcomes.items():
            log[f"share_{name}"] = value / episodes
    for level in range(1, levels):
        if attempts[level] >= 1.0:
            cm = round(100.0 * float(stair_cfg(env).riser_heights[level]))   # exact config value
            log[f"clean_h{cm:02d}cm"] = clean[level] / attempts[level]
    return log
