"""ROS-free contract for the MuJoCo lidar raycast adapter."""

import math

import mujoco
import numpy as np
import pytest

from g1_simulation_bridge.mujoco_raycast import (
    MuJoCoRayAdapter,
    MuJoCoRaycastError,
)


SCENE_XML = """
<mujoco model="lidar_test">
  <worldbody>
    <geom
      name="floor"
      type="plane"
      size="10 10 0.1"
    />
    <geom
      name="obstacle"
      type="sphere"
      pos="3 0 1"
      size="0.1"
    />
    <body
      name="pelvis"
      pos="0 0 1"
    >
      <body name="torso_link" />
    </body>
  </worldbody>
</mujoco>
"""


def scene():
    model = mujoco.MjModel.from_xml_string(
        SCENE_XML
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_reads_named_torso_pose_and_known_scene_hits():
    model, data = scene()
    adapter = MuJoCoRayAdapter(
        model=model,
        data=data,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
    )

    torso_pose = adapter.torso_pose()
    intersections = adapter.cast(
        origin_world=(0.0, 0.0, 1.0),
        directions_world=(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
        ),
    )

    np.testing.assert_allclose(
        torso_pose.position_xyz,
        (0.0, 0.0, 1.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torso_pose.orientation_wxyz,
        (1.0, 0.0, 0.0, 0.0),
        rtol=0.0,
        atol=1.0e-12,
    )

    assert intersections[0].geom_name == "obstacle"
    assert intersections[0].distance == pytest.approx(
        2.9,
        abs=1.0e-12,
    )

    assert intersections[1].geom_name == "floor"
    assert intersections[1].distance == pytest.approx(
        1.0,
        abs=1.0e-12,
    )

    assert intersections[2] is None


def test_rejects_missing_torso_body_name():
    model, data = scene()

    with pytest.raises(
        MuJoCoRaycastError,
        match="torso body",
    ):
        MuJoCoRayAdapter(
            model=model,
            data=data,
            torso_body_name="missing_torso",
            robot_root_body_name="pelvis",
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_rejects_nonfinite_native_intersection_distance(
    invalid_distance,
):
    model, data = scene()

    def invalid_raycast(
        model,
        data,
        origin_world,
        direction_world,
    ):
        del model
        del data
        del origin_world
        del direction_world
        return invalid_distance, 0

    adapter = MuJoCoRayAdapter(
        model=model,
        data=data,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
        raycast=invalid_raycast,
    )

    with pytest.raises(
        MuJoCoRaycastError,
        match="finite",
    ):
        adapter.cast(
            origin_world=(0.0, 0.0, 1.0),
            directions_world=(
                (1.0, 0.0, 0.0),
            ),
        )

SELF_FILTER_SCENE_XML = """
<mujoco model="lidar_self_filter_test">
  <worldbody>
    <geom
      name="environment_obstacle"
      type="sphere"
      pos="3 0 1"
      size="0.1"
    />
    <body
      name="pelvis"
      pos="0 0 1"
    >
      <body name="torso_link">
        <geom
          name="robot_self_geom"
          type="sphere"
          pos="1 0 0"
          size="0.1"
        />
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def self_filter_scene():
    model = mujoco.MjModel.from_xml_string(
        SELF_FILTER_SCENE_XML
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_skips_pelvis_subtree_geometries_before_environment_hit():
    model, data = self_filter_scene()
    adapter = MuJoCoRayAdapter(
        model=model,
        data=data,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
    )

    intersections = adapter.cast(
        origin_world=(0.0, 0.0, 1.0),
        directions_world=(
            (1.0, 0.0, 0.0),
        ),
    )

    assert intersections[0].geom_name == (
        "environment_obstacle"
    )
    assert intersections[0].distance == pytest.approx(
        2.9,
        abs=1.0e-9,
    )


def test_rejects_missing_robot_root_body_name():
    model, data = self_filter_scene()

    with pytest.raises(
        MuJoCoRaycastError,
        match="robot root body",
    ):
        MuJoCoRayAdapter(
            model=model,
            data=data,
            torso_body_name="torso_link",
            robot_root_body_name="missing_pelvis",
        )


def test_bounds_repeated_robot_self_intersections():
    model, data = self_filter_scene()
    self_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "robot_self_geom",
    )
    calls = []

    def repeated_self_hit(
        model,
        data,
        origin_world,
        direction_world,
    ):
        del model
        del data
        calls.append(
            (
                tuple(origin_world),
                tuple(direction_world),
            )
        )
        return 0.1, self_geom_id

    adapter = MuJoCoRayAdapter(
        model=model,
        data=data,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
        raycast=repeated_self_hit,
    )

    with pytest.raises(
        MuJoCoRaycastError,
        match="self-filter attempts",
    ):
        adapter.cast(
            origin_world=(0.0, 0.0, 1.0),
            directions_world=(
                (1.0, 0.0, 0.0),
            ),
        )

    assert 1 < len(calls) <= 3


def test_rejects_disabled_self_filter():
    model, data = scene()

    with pytest.raises(
        MuJoCoRaycastError,
        match="robot root body",
    ):
        MuJoCoRayAdapter(
            model=model,
            data=data,
            torso_body_name="torso_link",
            robot_root_body_name=None,
        )


def test_accepts_unnamed_environment_geometry():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco model="unnamed_environment">
          <worldbody>
            <geom
              type="sphere"
              pos="3 0 1"
              size="0.1"
            />
            <body
              name="pelvis"
              pos="0 0 1"
            >
              <body name="torso_link" />
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    adapter = MuJoCoRayAdapter(
        model=model,
        data=data,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
    )

    intersections = adapter.cast(
        origin_world=(0.0, 0.0, 1.0),
        directions_world=(
            (1.0, 0.0, 0.0),
        ),
    )

    assert intersections[0].distance == pytest.approx(
        2.9,
        abs=1.0e-12,
    )
    assert intersections[0].geom_id == 0
    assert intersections[0].geom_name is None
