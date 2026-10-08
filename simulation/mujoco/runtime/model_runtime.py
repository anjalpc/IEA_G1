"""Shared model initialization and safety-harness behavior."""

from __future__ import annotations

import mujoco
import numpy as np


class ElasticHarness:
    """Unitree-style overhead elastic tether for controller bring-up."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.body_id < 0:
            raise RuntimeError("G1 model does not contain torso_link")
        self.enabled = True
        self.anchor = np.array([0.0, 0.0, 3.0])
        self.length = 0.0
        self.stiffness = 200.0
        self.damping = 100.0

    def key(self, keycode: int) -> None:
        if keycode == ord("9"):
            self.enabled = not self.enabled
            print(
                f"HARNESS: {'attached' if self.enabled else 'released'}",
                flush=True,
            )
        elif keycode == ord("8"):
            self.length += 0.10
            print(f"HARNESS LENGTH: {self.length:.2f} m", flush=True)
        elif keycode == ord("7"):
            self.length = max(0.0, self.length - 0.10)
            print(f"HARNESS LENGTH: {self.length:.2f} m", flush=True)

    def apply(self, data: mujoco.MjData) -> None:
        data.xfrc_applied[self.body_id] = 0.0
        if not self.enabled:
            return
        # Match Unitree's native implementation: calculate the tether from the
        # floating-base translation, then apply that force at torso_link.
        delta = self.anchor - data.qpos[:3]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-9:
            return
        direction = delta / distance
        linear_velocity = data.qvel[:3]
        along_tether = float(np.dot(linear_velocity, direction))
        magnitude = self.stiffness * (distance - self.length)
        magnitude -= self.damping * along_tether
        data.xfrc_applied[self.body_id, :3] = magnitude * direction


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Initialize the integrated G1 model in its supported start pose."""
    data.qpos[:] = model.qpos0
    data.qpos[2] = 0.80
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
