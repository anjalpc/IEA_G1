"""Bind the simulation-only planar carrier to MuJoCo."""

import math

from navigation.planar_carrier import (
    PlanarCarrier,
    PlanarPose,
)


class MuJoCoPlanarCarrier:
    """Apply planar carrier state to a named MuJoCo free joint."""

    def __init__(
        self,
        *,
        mujoco,
        model,
        data,
        harness,
        command_timeout_seconds,
        max_forward_mps,
        max_lateral_mps,
        max_yaw_rate_rps,
        allow_lateral,
        joint_name="floating_base_joint",
    ):
        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._harness = harness

        identifier = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if identifier < 0:
            raise ValueError(
                "planar carrier free joint was not found"
            )
        if (
            model.jnt_type[identifier]
            != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(
                "planar carrier joint must be free"
            )

        self._qpos = int(
            model.jnt_qposadr[identifier]
        )
        self._qvel = int(
            model.jnt_dofadr[identifier]
        )

        if not hasattr(harness, "anchor"):
            raise ValueError(
                "planar carrier requires a harness anchor"
            )
        if len(harness.anchor) < 2:
            raise ValueError(
                "harness anchor must contain x and y"
            )

        quaternion = tuple(
            float(value)
            for value in data.qpos[
                self._qpos + 3:self._qpos + 7
            ]
        )
        yaw = self._yaw_from_wxyz(quaternion)

        self._carrier = PlanarCarrier(
            command_timeout_seconds=(
                command_timeout_seconds
            ),
            max_forward_mps=max_forward_mps,
            max_lateral_mps=max_lateral_mps,
            max_yaw_rate_rps=max_yaw_rate_rps,
            allow_lateral=allow_lateral,
            initial_pose=PlanarPose(
                x=float(data.qpos[self._qpos]),
                y=float(data.qpos[self._qpos + 1]),
                yaw=yaw,
            ),
        )

    @staticmethod
    def _yaw_from_wxyz(quaternion):
        """Extract planar yaw from a MuJoCo wxyz quaternion."""
        w, x, y, z = quaternion
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @property
    def pose(self):
        """Return the current pure carrier pose."""
        return self._carrier.pose

    @property
    def stopped(self):
        """Return whether planar carrier motion is stopped."""
        return self._carrier.stopped

    def command(
        self,
        *,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
        simulation_time,
    ):
        """Accept one validated simulation-time command."""
        self._carrier.command(
            forward_mps=forward_mps,
            lateral_mps=lateral_mps,
            yaw_rate_rps=yaw_rate_rps,
            simulation_time=simulation_time,
        )

    def _apply(self):
        pose = self._carrier.pose
        velocity = self._carrier.velocity
        half_yaw = pose.yaw / 2.0

        self._data.qpos[self._qpos] = pose.x
        self._data.qpos[self._qpos + 1] = pose.y

        self._data.qpos[self._qpos + 3] = (
            math.cos(half_yaw)
        )
        self._data.qpos[self._qpos + 4] = 0.0
        self._data.qpos[self._qpos + 5] = 0.0
        self._data.qpos[self._qpos + 6] = (
            math.sin(half_yaw)
        )

        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        self._data.qvel[self._qvel] = (
            velocity.forward_mps * cos_yaw
            - velocity.lateral_mps * sin_yaw
        )
        self._data.qvel[self._qvel + 1] = (
            velocity.forward_mps * sin_yaw
            + velocity.lateral_mps * cos_yaw
        )
        self._data.qvel[self._qvel + 3] = 0.0
        self._data.qvel[self._qvel + 4] = 0.0
        self._data.qvel[self._qvel + 5] = (
            velocity.yaw_rate_rps
        )

        self._harness.anchor[0] = pose.x
        self._harness.anchor[1] = pose.y

        self._mujoco.mj_forward(
            self._model,
            self._data,
        )

    def after_step(self, *, simulation_time):
        """Advance and apply state after a physics step."""
        pose = self._carrier.advance(
            simulation_time=simulation_time
        )
        self._apply()
        return pose

    def stop(self):
        """Stop and immediately expose zero planar velocity."""
        self._carrier.stop()
        self._apply()
