"""Read-only physics measurements for planned footsteps.

Provides body-contact support flags and collision-envelope sole positions.
Support force is NOT proof of valid tread contact; callers must inspect the
course, collision and touchdown semantics before using it for advancement.
"""
from __future__ import annotations

import math
import torch

from .contact_sensor_contract import FOOT_BODY_NAMES
from .motion_reference import sole_state


def measured_foot_state(env, *, support_force_n: float):
    """Return course-local sole-bottom-envelope centers [N,2,3] and support [N,2].

    Both robot and sensor body orders are resolved explicitly as [left, right].
    This function has no side effects and never advances a footstep target.
    """
    if not math.isfinite(support_force_n) or support_force_n <= 0:
        raise ValueError("support_force_n must be finite and positive")
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    robot_names = list(robot.body_names)
    sensor_names = list(sensor.body_names)
    missing = [name for name in FOOT_BODY_NAMES
               if name not in robot_names or name not in sensor_names]
    if missing:
        raise RuntimeError(f"foot bodies absent from robot/contact sensor: {missing}")
    body_ids = [robot_names.index(name) for name in FOOT_BODY_NAMES]
    sensor_ids = [sensor_names.index(name) for name in FOOT_BODY_NAMES]
    data = robot.data
    centre, _bounds, _velocity = sole_state(
        data.body_pos_w[:, body_ids],
        data.body_quat_w[:, body_ids],
        data.body_lin_vel_w[:, body_ids],
        data.body_ang_vel_w[:, body_ids],
    )
    sole_xyz = centre - env.scene.env_origins[:, None, :]
    force_z = sensor.data.net_forces_w[:, sensor_ids, 2]
    if (sole_xyz.shape != (env.num_envs, 2, 3)
            or force_z.shape != (env.num_envs, 2)):
        raise ValueError("unexpected foot state/sensor shapes")
    if (not torch.isfinite(sole_xyz).all()
            or not torch.isfinite(force_z).all()):
        raise ValueError("non-finite foot state/contact measurements")
    return sole_xyz, force_z >= support_force_n
