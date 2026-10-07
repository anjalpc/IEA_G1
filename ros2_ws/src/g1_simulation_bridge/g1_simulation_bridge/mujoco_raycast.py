"""ROS-free MuJoCo raycast boundary for simulated lidar."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np


_NORMALIZATION_TOLERANCE = 1.0e-9
_SELF_HIT_ADVANCE_METRES = 1.0e-7


class MuJoCoRaycastError(RuntimeError):
    """Report an invalid model contract or raycast result."""


@dataclass(frozen=True)
class TorsoPose:
    """World pose read directly from MuJoCo using wxyz orientation."""

    position_xyz: tuple
    orientation_wxyz: tuple


@dataclass(frozen=True)
class RayIntersection:
    """One validated MuJoCo geometry intersection."""

    distance: float
    geom_id: int
    geom_name: str


def _finite_vector(name, values):
    try:
        vector = np.asarray(
            tuple(values),
            dtype=float,
        )
    except (TypeError, ValueError):
        raise MuJoCoRaycastError(
            f"{name} must be numeric"
        ) from None

    if vector.shape != (3,):
        raise MuJoCoRaycastError(
            f"{name} must contain exactly 3 values"
        )
    if not np.all(np.isfinite(vector)):
        raise MuJoCoRaycastError(
            f"{name} must be finite"
        )
    return vector


def _validated_directions(values):
    try:
        directions = np.asarray(
            tuple(values),
            dtype=float,
        )
    except (TypeError, ValueError):
        raise MuJoCoRaycastError(
            "ray directions must be numeric"
        ) from None

    if (
        directions.ndim != 2
        or directions.shape[0] == 0
        or directions.shape[1] != 3
    ):
        raise MuJoCoRaycastError(
            "ray directions must have shape (N, 3)"
        )
    if not np.all(np.isfinite(directions)):
        raise MuJoCoRaycastError(
            "ray directions must be finite"
        )

    norms = np.linalg.norm(
        directions,
        axis=1,
    )
    if not np.allclose(
        norms,
        np.ones(len(norms)),
        rtol=0.0,
        atol=_NORMALIZATION_TOLERANCE,
    ):
        raise MuJoCoRaycastError(
            "ray directions must be normalized"
        )
    return directions


def _native_raycast(
    model,
    data,
    origin_world,
    direction_world,
):
    geom_id = np.full(
        1,
        -1,
        dtype=np.int32,
    )
    distance = mujoco.mj_ray(
        model,
        data,
        origin_world,
        direction_world,
        None,
        1,
        -1,
        geom_id,
    )
    return float(distance), int(geom_id[0])


def _body_belongs_to_subtree(
    model,
    *,
    body_id,
    root_body_id,
):
    while True:
        if body_id == root_body_id:
            return True
        if body_id == 0:
            return False
        body_id = int(
            model.body_parentid[body_id]
        )


def _robot_geometry_ids(
    model,
    *,
    root_body_name,
):
    root_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        root_body_name,
    )
    if root_body_id < 0:
        raise MuJoCoRaycastError(
            "robot root body not found: "
            f"{root_body_name}"
        )

    return frozenset(
        geom_id
        for geom_id in range(model.ngeom)
        if _body_belongs_to_subtree(
            model,
            body_id=int(
                model.geom_bodyid[geom_id]
            ),
            root_body_id=root_body_id,
        )
    )


class MuJoCoRayAdapter:
    """Read a named torso pose and cast validated world-frame rays."""

    def __init__(
        self,
        *,
        model,
        data,
        torso_body_name,
        robot_root_body_name=None,
        raycast=None,
    ):
        self._model = model
        self._data = data
        self._raycast = (
            _native_raycast
            if raycast is None
            else raycast
        )
        if robot_root_body_name is None:
            raise MuJoCoRaycastError(
                "robot root body name is required"
            )
        self._robot_geom_ids = (
            _robot_geometry_ids(
                model,
                root_body_name=(
                    robot_root_body_name
                ),
            )
        )

        torso_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            torso_body_name,
        )
        if torso_body_id < 0:
            raise MuJoCoRaycastError(
                "torso body not found: "
                f"{torso_body_name}"
            )
        self._torso_body_id = torso_body_id

    def torso_pose(self):
        """Return the current world pose of the configured torso body."""
        position = self._data.xpos[
            self._torso_body_id
        ]
        orientation = self._data.xquat[
            self._torso_body_id
        ]

        if (
            not np.all(np.isfinite(position))
            or not np.all(np.isfinite(orientation))
        ):
            raise MuJoCoRaycastError(
                "torso pose must be finite"
            )

        orientation_norm = float(
            np.linalg.norm(orientation)
        )
        if not math.isclose(
            orientation_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise MuJoCoRaycastError(
                "torso orientation must be normalized"
            )

        return TorsoPose(
            position_xyz=tuple(
                float(value)
                for value in position
            ),
            orientation_wxyz=tuple(
                float(value)
                for value in orientation
            ),
        )

    def cast(
        self,
        *,
        origin_world,
        directions_world,
    ):
        """Cast rays while excluding robot-owned intersections."""
        origin = _finite_vector(
            "ray origin",
            origin_world,
        )
        directions = _validated_directions(
            directions_world
        )
        maximum_attempts = (
            2 * len(self._robot_geom_ids)
            + 1
        )

        intersections = []
        for direction in directions:
            ray_origin = origin.copy()
            travelled = 0.0

            for _attempt in range(
                maximum_attempts
            ):
                distance, geom_id = self._raycast(
                    self._model,
                    self._data,
                    ray_origin,
                    direction,
                )
                distance = float(distance)
                geom_id = int(geom_id)

                if not math.isfinite(distance):
                    raise MuJoCoRaycastError(
                        "raycast distance must be finite"
                    )

                if distance < 0.0 and geom_id < 0:
                    intersections.append(None)
                    break

                if distance <= 0.0:
                    raise MuJoCoRaycastError(
                        "raycast hit distance must be positive"
                    )
                if (
                    geom_id < 0
                    or geom_id >= self._model.ngeom
                ):
                    raise MuJoCoRaycastError(
                        "raycast geometry id is invalid"
                    )

                geom_name = mujoco.mj_id2name(
                    self._model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                if not geom_name:
                    raise MuJoCoRaycastError(
                        "raycast geometry must have a name"
                    )

                total_distance = (
                    travelled + distance
                )
                if (
                    geom_id
                    not in self._robot_geom_ids
                ):
                    intersections.append(
                        RayIntersection(
                            distance=total_distance,
                            geom_id=geom_id,
                            geom_name=geom_name,
                        )
                    )
                    break

                advance = (
                    distance
                    + _SELF_HIT_ADVANCE_METRES
                )
                ray_origin = (
                    ray_origin
                    + direction * advance
                )
                travelled += advance
            else:
                raise MuJoCoRaycastError(
                    "ray exceeded bounded "
                    "self-filter attempts"
                )

        return tuple(intersections)
