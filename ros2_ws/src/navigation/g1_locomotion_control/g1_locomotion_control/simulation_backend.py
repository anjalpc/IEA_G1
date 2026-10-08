"""Arbiter-owned backend for simulation planar velocity."""

import math

from geometry_msgs.msg import Twist


class SimulationBackendError(RuntimeError):
    """Stable simulation-backend boundary error."""


class SimulationBackend:
    """Publish approved velocity to the MuJoCo bridge."""

    def __init__(self, publisher):
        self._publisher = publisher
        self.active = False
        self.stopped = True
        self.faulted = False

    def activate(self):
        """Prepare without producing a motion command."""
        if self.active:
            return
        self.active = True
        self.stopped = True
        self.faulted = False

    @staticmethod
    def _message(
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        try:
            values = tuple(
                float(value)
                for value in (
                    forward_mps,
                    lateral_mps,
                    yaw_rate_rps,
                )
            )
        except (TypeError, ValueError) as error:
            raise SimulationBackendError(
                "SIMULATION_COMMAND_FAILED"
            ) from error

        if not all(math.isfinite(value) for value in values):
            raise SimulationBackendError(
                "SIMULATION_COMMAND_FAILED"
            )

        message = Twist()
        message.linear.x = values[0]
        message.linear.y = values[1]
        message.linear.z = 0.0
        message.angular.x = 0.0
        message.angular.y = 0.0
        message.angular.z = values[2]
        return message

    def command_velocity(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        """Publish one already-approved planar velocity."""
        if not self.active:
            raise SimulationBackendError(
                "SIMULATION_BACKEND_INACTIVE"
            )
        if self.faulted:
            raise SimulationBackendError(
                "SIMULATION_BACKEND_UNAVAILABLE"
            )

        message = self._message(
            forward_mps,
            lateral_mps,
            yaw_rate_rps,
        )
        try:
            self._publisher.publish(message)
        except Exception as error:
            self.faulted = True
            self.stopped = False
            raise SimulationBackendError(
                "SIMULATION_COMMAND_FAILED"
            ) from error

        self.stopped = (
            message.linear.x == 0.0
            and message.linear.y == 0.0
            and message.angular.z == 0.0
        )

    def stop(self):
        """Publish explicit zero whenever active."""
        if not self.active:
            return

        message = self._message(0.0, 0.0, 0.0)
        try:
            self._publisher.publish(message)
        except Exception as error:
            self.faulted = True
            self.stopped = False
            raise SimulationBackendError(
                "SIMULATION_STOP_FAILED"
            ) from error

        self.stopped = True

    def healthy(self):
        """Require an active backend and live bridge listener."""
        if not self.active or self.faulted:
            return False
        try:
            return (
                self._publisher.get_subscription_count()
                > 0
            )
        except Exception:
            return False

    def deactivate(self):
        """Deactivate only after a confirmed stop."""
        if not self.active:
            return
        if not self.stopped:
            raise SimulationBackendError(
                "SIMULATION_BACKEND_NOT_STOPPED"
            )
        self.active = False
