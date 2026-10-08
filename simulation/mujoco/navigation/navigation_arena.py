"""Deterministic project-owned geometry for navigation integration."""

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class ArenaBox:
    """Describe one named world-frame MuJoCo box."""

    name: str
    position_xyz: tuple[float, float, float]
    half_size_xyz: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


NAVIGATION_ARENA_BOXES = (
    ArenaBox(
        name="navigation_wall_north",
        position_xyz=(0.0, 3.0, 1.0),
        half_size_xyz=(3.0, 0.1, 1.0),
        rgba=(0.35, 0.42, 0.50, 1.0),
    ),
    ArenaBox(
        name="navigation_wall_east",
        position_xyz=(3.5, 0.5, 1.0),
        half_size_xyz=(0.1, 2.5, 1.0),
        rgba=(0.42, 0.50, 0.35, 1.0),
    ),
    ArenaBox(
        name="navigation_pillar_southwest",
        position_xyz=(-1.75, -1.25, 0.9),
        half_size_xyz=(0.35, 0.35, 0.9),
        rgba=(0.52, 0.38, 0.30, 1.0),
    ),
)


def add_navigation_arena(spec):
    """Attach deterministic navigation obstacles to an editable model."""
    for box in NAVIGATION_ARENA_BOXES:
        if spec.geom(box.name) is not None:
            raise ValueError(
                f"navigation arena geometry already exists: {box.name}"
            )

        spec.worldbody.add_geom(
            name=box.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box.position_xyz,
            size=box.half_size_xyz,
            rgba=box.rgba,
        )

    return spec
