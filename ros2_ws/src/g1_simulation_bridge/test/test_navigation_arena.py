"""Contract for the project-owned MuJoCo navigation arena."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest


SIMULATION_ROOT = (
    Path(__file__).parents[4]
    / "simulation"
    / "mujoco"
)
VENDOR_SCENE = (
    SIMULATION_ROOT
    / "vendor/unitree_g1/scene_29dof.xml"
)
ARENA_MODULE = (
    SIMULATION_ROOT
    / "navigation/navigation_arena.py"
)
sys.path.insert(0, str(SIMULATION_ROOT))

from model_composition import (  # noqa: E402
    build_combined_model,
)

from g1_simulation_bridge.mujoco_raycast import (  # noqa: E402
    MuJoCoRayAdapter,
)


EXPECTED_GEOMETRIES = {
    "navigation_wall_north": {
        "position": (0.0, 3.0, 1.0),
        "half_size": (3.0, 0.1, 1.0),
    },
    "navigation_wall_east": {
        "position": (3.5, 0.5, 1.0),
        "half_size": (0.1, 2.5, 1.0),
    },
    "navigation_pillar_southwest": {
        "position": (-1.75, -1.25, 0.9),
        "half_size": (0.35, 0.35, 0.9),
    },
}


@pytest.fixture(scope="module")
def arena():
    model = build_combined_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def geom_id(model, name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )


def test_arena_is_project_owned_not_vendored():
    assert ARENA_MODULE.is_file()

    vendor_source = VENDOR_SCENE.read_text(
        encoding="utf-8"
    )
    for name in EXPECTED_GEOMETRIES:
        assert name not in vendor_source


def test_combined_model_contains_named_arena_geometry(
    arena,
):
    model, _data = arena

    for name in EXPECTED_GEOMETRIES:
        assert geom_id(model, name) >= 0, name


def test_arena_geometry_has_deterministic_dimensions(
    arena,
):
    model, _data = arena

    for name, expected in EXPECTED_GEOMETRIES.items():
        identifier = geom_id(model, name)
        assert identifier >= 0, name

        np.testing.assert_allclose(
            model.geom_pos[identifier],
            expected["position"],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model.geom_size[identifier],
            expected["half_size"],
            rtol=0.0,
            atol=1.0e-12,
        )


def test_horizontal_lidar_rays_hit_asymmetric_walls(
    arena,
):
    model, data = arena
    adapter = MuJoCoRayAdapter(
        model=model,
        data=data,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
    )

    intersections = adapter.cast(
        origin_world=(0.0, 0.0, 1.7),
        directions_world=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
    )

    assert intersections[0] is not None
    assert intersections[0].geom_name == (
        "navigation_wall_east"
    )
    assert intersections[0].distance == pytest.approx(
        3.4,
        abs=1.0e-9,
    )

    assert intersections[1] is not None
    assert intersections[1].geom_name == (
        "navigation_wall_north"
    )
    assert intersections[1].distance == pytest.approx(
        2.9,
        abs=1.0e-9,
    )
