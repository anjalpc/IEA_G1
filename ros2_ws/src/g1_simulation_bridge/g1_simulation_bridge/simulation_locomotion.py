"""Trusted arbiter-to-MuJoCo locomotion ingress."""

from geometry_msgs.msg import Twist
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from g1_robot_interfaces.locomotion import (
    PlanarVelocityLimits,
    validate_planar_velocity,
)


LATERAL_TOLERANCE = 1e-9
SIMULATION_COMMAND_TOPIC = (
    "/g1/simulation/locomotion_command"
)


class SimulationLocomotionController:
    """Validate trusted commands and drive the planar carrier."""

    def __init__(
        self,
        *,
        node,
        carrier,
        data,
        velocity_limits,
        allow_lateral,
    ):
        if not isinstance(
            velocity_limits,
            PlanarVelocityLimits,
        ):
            raise ValueError(
                "velocity_limits must be PlanarVelocityLimits"
            )
        if not isinstance(allow_lateral, bool):
            raise ValueError("allow_lateral must be boolean")

        self._node = node
        self._carrier = carrier
        self._data = data
        self._velocity_limits = velocity_limits
        self._allow_lateral = allow_lateral

        self.command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.command_subscription = (
            node.create_subscription(
                Twist,
                SIMULATION_COMMAND_TOPIC,
                self.command_callback,
                self.command_qos,
            )
        )

    def command_callback(self, message):
        """Validate and apply one simulation command."""
        try:
            forward, lateral, yaw_rate = (
                validate_planar_velocity(
                    linear_x=message.linear.x,
                    linear_y=message.linear.y,
                    linear_z=message.linear.z,
                    angular_x=message.angular.x,
                    angular_y=message.angular.y,
                    angular_z=message.angular.z,
                    limits=self._velocity_limits,
                )
            )
            if (
                not self._allow_lateral
                and abs(lateral) > LATERAL_TOLERANCE
            ):
                raise ValueError(
                    "lateral simulation velocity is disabled"
                )

            self._carrier.command(
                forward_mps=forward,
                lateral_mps=(
                    lateral if self._allow_lateral else 0.0
                ),
                yaw_rate_rps=yaw_rate,
                simulation_time=float(self._data.time),
            )
        except ValueError:
            self._carrier.stop()
            self._node.get_logger().error(
                "Rejected unsafe simulation locomotion command"
            )

    def after_step(self):
        """Advance the carrier using completed simulation time."""
        self._carrier.after_step(
            simulation_time=float(self._data.time)
        )

    def stop(self):
        """Stop the simulation carrier immediately."""
        self._carrier.stop()
