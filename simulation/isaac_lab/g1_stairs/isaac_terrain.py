"""Isaac Lab terrain adapter for the simulator-independent stair contract.

Import this module only inside an Isaac Lab Python environment.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .geometry import StaircaseSpec, effective_platform_length


def _difficulty_level(
    difficulty: float,
    bins: int,
) -> int:
    """Map Isaac Lab difficulty to one exact terrain row."""
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    if bins < 1:
        raise ValueError("bins must be >= 1")

    return min(
        int(np.floor(float(difficulty) * bins)),
        bins - 1,
    )


def _riser_height(
    difficulty: float,
    cfg: "ThreeStepStairsCfg",
) -> float:
    """Resolve the exact riser represented by this terrain row."""
    if cfg.riser_heights is not None:
        heights = tuple(float(value) for value in cfg.riser_heights)
        if not heights:
            raise ValueError("riser_heights must not be empty")
        if any(value < 0.0 or value > 0.20 for value in heights):
            raise ValueError("riser_heights must be in [0.0, 0.20]")

        level = _difficulty_level(difficulty, len(heights))
        return heights[level]

    low, high = cfg.riser_height_range

    if abs(high - low) <= 1.0e-12:
        return float(low)

    bins = int(cfg.difficulty_bins)
    if bins < 2:
        raise ValueError(
            "difficulty_bins must be >= 2 for variable riser heights"
        )

    # Isaac Lab adds a small random difficulty offset inside each
    # curriculum row. Quantize back to the row so the terrain level
    # and physical height match exactly.
    level = _difficulty_level(difficulty, bins)
    alpha = level / float(bins - 1)
    return float(low + alpha * (high - low))


def three_step_stairs(
    difficulty: float, cfg: "ThreeStepStairsCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build an exact flat or ascending terrain row."""
    riser = _riser_height(difficulty, cfg)

    # A zero-riser row is an actual flat tile, not a degenerate staircase.
    # Keep the same robot-relative start origin as the stair rows so the
    # locomotion and geometry contracts remain directly comparable.
    if abs(riser) <= 1.0e-12:
        foundation_depth = 0.10
        mesh = trimesh.creation.box(
            extents=(cfg.size[0], cfg.size[1], foundation_depth)
        )
        mesh.apply_translation(
            (
                0.5 * cfg.size[0],
                0.5 * cfg.size[1],
                -0.5 * foundation_depth,
            )
        )
        start_x = cfg.approach_length - cfg.start_clearance
        origin = np.asarray((start_x, 0.5 * cfg.size[1], 0.0))
        return [mesh], origin
    # Isaac Lab places terrain tiles according to cfg.size, so every generated
    # sub-terrain must cover the full tile length.  The nominal three-step
    # course already does this exactly:
    #
    #   2.00 m approach + 3 * 0.25 m tread + 3.00 m platform = 5.75 m
    #
    # Reducing step_count without extending the platform previously left an
    # uncovered X gap at the end of the tile.  Keep the tile size fixed and
    # extend the upper platform to consume the remaining length.
    # Showcase only: each course and its robot start move together by a
    # random (dx, dy) inside a larger tile (irregular field). Everything at
    # runtime is relative to the robot start (env origin), so the robot sees
    # the same course as in training.
    dx = dy = 0.0
    if cfg.showcase_jitter is not None:
        jx, jy = (float(v) for v in cfg.showcase_jitter)
        rng = np.random.default_rng(abs(hash((int(cfg.showcase_seed), round(float(difficulty), 12)))))
        dx, dy = float(rng.uniform(0.0, jx)), float(rng.uniform(-jy, jy))
    platform_length = effective_platform_length(
        terrain_size_x=cfg.size[0] - dx,
        approach_length=cfg.approach_length,
        step_count=cfg.step_count,
        tread_depth=cfg.tread_depth,
        minimum_platform_length=cfg.platform_length,
    )

    # Showcase tiles may be wider than the staircase (stair_width): plain
    # floor on both sides keeps neighbouring courses visibly apart. The
    # height scan reaches 0.5 m sideways, inside a 1.2 m staircase.
    width = cfg.size[1] if cfg.stair_width is None else float(cfg.stair_width)
    if not 0.0 < width <= cfg.size[1] + 1e-9:
        raise ValueError("stair_width must be in (0, tile width]")
    if abs(dy) + 0.5 * width > 0.5 * cfg.size[1] + 1e-9:
        raise ValueError("showcase_jitter moves the staircase out of its tile")
    spec = StaircaseSpec(
        riser_height=riser,
        tread_depth=cfg.tread_depth,
        step_count=cfg.step_count,
        width=width,
        approach_length=cfg.approach_length,
        platform_length=platform_length,
        start_clearance=cfg.start_clearance,
    )

    if not np.isclose(spec.course_length, cfg.size[0] - dx, atol=1e-9):
        raise ValueError(
            "generated stair course must exactly fill the terrain tile: "
            f"course_length={spec.course_length}, terrain_size_x={cfg.size[0]}"
        )
    meshes = []
    if width < cfg.size[1] - 1e-9 or dx > 0.0:
        floor = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.10))
        floor.apply_translation((0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.05))
        meshes.append(floor)
    for box in spec.collision_boxes():
        mesh = trimesh.creation.box(extents=box.size)
        # SubTerrainBaseCfg requires the mesh to span (0, 0) to cfg.size.
        mesh.apply_translation((box.center[0] + dx, box.center[1] + 0.5 * cfg.size[1] + dy, box.center[2]))
        meshes.append(mesh)
    # Isaac Lab uses the returned origin to place the robot.  Start it halfway
    # along the approach, facing the first riser in +X.
    start_x, _, _ = spec.fixed_start_xy_yaw
    origin = np.asarray((start_x + dx, 0.5 * cfg.size[1] + dy, 0.0))
    return meshes, origin


@configclass
class ThreeStepStairsCfg(SubTerrainBaseCfg):
    """Configuration for the exact three-step course and its curriculum."""

    function = three_step_stairs
    size: tuple[float, float] = MISSING
    riser_height_range: tuple[float, float] = (0.05, 0.20)
    # Optional exact per-row heights. When set, this takes precedence over
    # riser_height_range and may include 0.0 to represent a true flat tile.
    riser_heights: tuple[float, ...] | None = None
    # Staircase width when narrower than the tile (showcase only); None =
    # the full tile width, as in training.
    stair_width: float | None = None
    # Showcase only: random (forward 0..x, sideways +-y) offset per course,
    # seeded by showcase_seed and the tile's difficulty draw. None in training.
    showcase_jitter: tuple[float, float] | None = None
    showcase_seed: int = 0
    difficulty_bins: int = 8
    tread_depth: float = 0.25
    step_count: int = 3
    approach_length: float = 2.0
    platform_length: float = 3.0
    start_clearance: float = 1.0
