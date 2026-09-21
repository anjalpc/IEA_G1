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


def three_step_stairs(
    difficulty: float, cfg: "ThreeStepStairsCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build an ascending staircase; difficulty controls the riser height."""
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    riser = cfg.riser_height_range[0] + difficulty * (
        cfg.riser_height_range[1] - cfg.riser_height_range[0]
    )
    # Isaac Lab places terrain tiles according to cfg.size, so every generated
    # sub-terrain must cover the full tile length.  The nominal three-step
    # course already does this exactly:
    #
    #   2.00 m approach + 3 * 0.25 m tread + 3.00 m platform = 5.75 m
    #
    # Reducing step_count without extending the platform previously left an
    # uncovered X gap at the end of the tile.  Keep the tile size fixed and
    # extend the upper platform to consume the remaining length.
    platform_length = effective_platform_length(
        terrain_size_x=cfg.size[0],
        approach_length=cfg.approach_length,
        step_count=cfg.step_count,
        tread_depth=cfg.tread_depth,
        minimum_platform_length=cfg.platform_length,
    )

    spec = StaircaseSpec(
        riser_height=riser,
        tread_depth=cfg.tread_depth,
        step_count=cfg.step_count,
        width=cfg.size[1],
        approach_length=cfg.approach_length,
        platform_length=platform_length,
        start_clearance=cfg.start_clearance,
    )

    if not np.isclose(spec.course_length, cfg.size[0], atol=1e-9):
        raise ValueError(
            "generated stair course must exactly fill the terrain tile: "
            f"course_length={spec.course_length}, terrain_size_x={cfg.size[0]}"
        )
    meshes = []
    for box in spec.collision_boxes():
        mesh = trimesh.creation.box(extents=box.size)
        # SubTerrainBaseCfg requires the mesh to span (0, 0) to cfg.size.
        mesh.apply_translation((box.center[0], box.center[1] + 0.5 * cfg.size[1], box.center[2]))
        meshes.append(mesh)
    # Isaac Lab uses the returned origin to place the robot.  Start it halfway
    # along the approach, facing the first riser in +X.
    start_x, _, _ = spec.fixed_start_xy_yaw
    origin = np.asarray((start_x, 0.5 * cfg.size[1], 0.0))
    return meshes, origin


@configclass
class ThreeStepStairsCfg(SubTerrainBaseCfg):
    """Configuration for the exact three-step course and its curriculum."""

    function = three_step_stairs
    size: tuple[float, float] = MISSING
    riser_height_range: tuple[float, float] = (0.05, 0.20)
    tread_depth: float = 0.25
    step_count: int = 3
    approach_length: float = 2.0
    platform_length: float = 3.0
    start_clearance: float = 1.0
