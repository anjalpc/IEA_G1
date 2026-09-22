"""Whole-body state representation for the validated stair reference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

from .reference_motion import load_reference_motion


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_reference_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def quat_mul_wxyz(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Hamilton product for w,x,y,z quaternions."""

    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def source_to_task_positions(
    positions: np.ndarray,
) -> np.ndarray:
    """Source frame -> task frame: X=-Y, Y=X, Z=Z."""

    p = np.asarray(
        positions,
        dtype=np.float64,
    )

    return np.stack(
        (
            -p[..., 1],
            p[..., 0],
            p[..., 2],
        ),
        axis=-1,
    )


def source_to_task_quaternion(
    quaternion: np.ndarray,
) -> np.ndarray:
    """Apply the +90 degree world-Z source-to-task rotation."""

    q = quat_normalize(quaternion)

    half = math.pi / 4.0

    q_world_rotation = np.asarray(
        [
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        ],
        dtype=np.float64,
    )

    return quat_normalize(
        quat_mul_wxyz(
            q_world_rotation,
            q,
        )
    )


def angular_velocity_from_quaternions(
    q0: np.ndarray,
    q1: np.ndarray,
    dt: float,
) -> np.ndarray:
    """World-frame angular velocity estimated from consecutive poses."""

    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    q_delta = quat_normalize(
        quat_mul_wxyz(
            q1,
            quat_conjugate_wxyz(q0),
        )
    )

    w = float(
        np.clip(
            q_delta[0],
            -1.0,
            1.0,
        )
    )

    angle = 2.0 * math.acos(w)

    sin_half = math.sqrt(
        max(
            1.0 - w * w,
            0.0,
        )
    )

    if (
        sin_half < 1.0e-8
        or angle < 1.0e-8
    ):
        return np.zeros(
            3,
            dtype=np.float64,
        )

    axis = q_delta[1:] / sin_half

    if angle > math.pi:
        angle -= 2.0 * math.pi

    return axis * (angle / dt)


@dataclass(frozen=True)
class ReferenceStateData:
    frequency_hz: float

    root_pos_task: np.ndarray
    root_quat_task: np.ndarray
    root_lin_vel_task: np.ndarray
    root_ang_vel_task: np.ndarray

    dof_pos: np.ndarray
    dof_vel: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.dof_pos.shape[0])


@lru_cache(maxsize=8)
def _load_reference_state_cached(
    path_string: str,
) -> ReferenceStateData:
    path = Path(path_string)

    motion = load_reference_motion(path)

    # pelvis is tracked body index 0 in the frozen reference contract
    source_pos = np.asarray(
        motion.body_pos_w[:, 0],
        dtype=np.float64,
    )

    source_quat = np.asarray(
        motion.body_quat_w[:, 0],
        dtype=np.float64,
    )

    root_pos_task = source_to_task_positions(
        source_pos
    )

    root_quat_task = np.stack(
        [
            source_to_task_quaternion(q)
            for q in source_quat
        ],
        axis=0,
    )

    dt = 1.0 / float(
        motion.frequency_hz
    )

    root_lin_vel_task = np.zeros_like(
        root_pos_task,
        dtype=np.float64,
    )

    root_lin_vel_task[:-1] = (
        root_pos_task[1:]
        - root_pos_task[:-1]
    ) / dt

    root_lin_vel_task[-1] = (
        root_lin_vel_task[-2]
    )

    root_ang_vel_task = np.zeros(
        (motion.frame_count, 3),
        dtype=np.float64,
    )

    for index in range(
        motion.frame_count - 1
    ):
        root_ang_vel_task[index] = (
            angular_velocity_from_quaternions(
                root_quat_task[index],
                root_quat_task[index + 1],
                dt,
            )
        )

    root_ang_vel_task[-1] = (
        root_ang_vel_task[-2]
    )

    return ReferenceStateData(
        frequency_hz=float(
            motion.frequency_hz
        ),
        root_pos_task=root_pos_task,
        root_quat_task=root_quat_task,
        root_lin_vel_task=root_lin_vel_task,
        root_ang_vel_task=root_ang_vel_task,
        dof_pos=np.asarray(
            motion.dof_pos,
            dtype=np.float32,
        ),
        dof_vel=np.asarray(
            motion.dof_vel,
            dtype=np.float32,
        ),
    )


def load_reference_state(
    value: str | Path,
) -> ReferenceStateData:
    path = resolve_reference_path(
        value
    ).resolve()

    return _load_reference_state_cached(
        str(path)
    )
