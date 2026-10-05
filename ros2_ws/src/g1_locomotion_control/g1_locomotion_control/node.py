"""ROS wrapper for the hardware-independent locomotion arbiter."""

import math

import rclpy

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from g1_locomotion_interfaces.msg import (
    LocomotionCommand,
    LocomotionState,
)
from g1_locomotion_interfaces.srv import (
    AcquireLocomotion,
    ReleaseLocomotion,
)
from g1_robot_interfaces.locomotion import PlanarVelocityLimits

from g1_locomotion_control.arbiter import LocomotionArbiter
from g1_locomotion_control.backend import RecordingBackend


class LocomotionArbiterNode(Node):
    """Expose the pure arbiter on the application ROS domain."""

    def __init__(
        self,
        *,
        backend=None,
        parameter_overrides=None,
        lease_factory=None,
        monotonic=None,
    ):
        super().__init__(
            "g1_locomotion_arbiter",
            parameter_overrides=parameter_overrides,
        )

        self.declare_parameter("backend", "recording")
        self.declare_parameter(
            "command_timeout_seconds",
            0.250,
        )
        self.declare_parameter("max_forward_mps", -1.0)
        self.declare_parameter("max_lateral_mps", -1.0)
        self.declare_parameter("max_yaw_rate_rps", -1.0)
        self.declare_parameter("watchdog_rate_hz", 50.0)
        self.declare_parameter("state_publish_rate_hz", 10.0)

        backend_name = str(
            self.get_parameter("backend").value
        )
        if backend_name != "recording":
            raise ValueError(
                "only the recording backend is available"
            )

        limits = PlanarVelocityLimits(
            forward_mps=self.get_parameter(
                "max_forward_mps"
            ).value,
            lateral_mps=self.get_parameter(
                "max_lateral_mps"
            ).value,
            yaw_rate_rps=self.get_parameter(
                "max_yaw_rate_rps"
            ).value,
        )
        timeout = self.get_parameter(
            "command_timeout_seconds"
        ).value

        watchdog_rate = float(
            self.get_parameter("watchdog_rate_hz").value
        )
        state_rate = float(
            self.get_parameter("state_publish_rate_hz").value
        )
        for name, value in (
            ("watchdog_rate_hz", watchdog_rate),
            ("state_publish_rate_hz", state_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        self.backend = backend or RecordingBackend()
        arbiter_arguments = {
            "backend": self.backend,
            "velocity_limits": limits,
            "command_timeout_seconds": timeout,
            "lease_factory": lease_factory,
        }
        if monotonic is not None:
            arbiter_arguments["monotonic"] = monotonic
        self.arbiter = LocomotionArbiter(**arbiter_arguments)

        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_publisher = self.create_publisher(
            LocomotionState,
            "/g1/locomotion/state",
            state_qos,
        )

        self.command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.command_subscription = self.create_subscription(
            LocomotionCommand,
            "/g1/locomotion/command",
            self.command_callback,
            self.command_qos,
        )
        self.acquire_service = self.create_service(
            AcquireLocomotion,
            "/g1/locomotion/acquire",
            self.acquire_callback,
        )
        self.release_service = self.create_service(
            ReleaseLocomotion,
            "/g1/locomotion/release",
            self.release_callback,
        )
        self.watchdog_timer = self.create_timer(
            1.0 / watchdog_rate,
            self.watchdog_callback,
        )
        self.state_timer = self.create_timer(
            1.0 / state_rate,
            self.publish_state,
        )
        self._destroyed = False
        self.publish_state()

    def acquire_callback(self, request, response):
        """Map an acquire service request into the pure arbiter."""
        result = self.arbiter.acquire(
            request.requested_owner,
            request.requester_id,
        )
        response.granted = result.granted
        response.lease_id = result.lease_id
        response.active_owner = result.active_owner
        response.reason = result.reason
        self.publish_state()
        return response

    def release_callback(self, request, response):
        """Map a release service request into the pure arbiter."""
        result = self.arbiter.release(
            request.owner,
            request.lease_id,
        )
        response.released = result.released
        response.active_owner = result.active_owner
        response.reason = result.reason
        self.publish_state()
        return response

    def command_callback(self, message):
        """Validate and dispatch one leased velocity command."""
        twist = message.twist
        self.arbiter.command(
            owner=message.owner,
            lease_id=message.lease_id,
            linear_x=twist.linear.x,
            linear_y=twist.linear.y,
            linear_z=twist.linear.z,
            angular_x=twist.angular.x,
            angular_y=twist.angular.y,
            angular_z=twist.angular.z,
        )
        self.publish_state()

    def watchdog_callback(self):
        """Check command freshness and publish safety transitions."""
        if self.arbiter.check_watchdog():
            self.publish_state()

    def state_message(self):
        """Build public ROS state without private lease data."""
        snapshot = self.arbiter.snapshot()
        message = LocomotionState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.active_owner = snapshot.active_owner
        message.active_backend = (
            LocomotionState.BACKEND_SIMULATION
        )
        message.ready = snapshot.ready
        message.stopped = snapshot.stopped
        message.emergency_stop = snapshot.emergency_stop
        message.command_age_seconds = (
            snapshot.command_age_seconds
        )
        message.reason = snapshot.reason
        return message

    def publish_state(self):
        """Publish the current public arbiter state."""
        self.state_publisher.publish(self.state_message())

    def destroy_node(self):
        """Safely shut down the arbiter before ROS resources."""
        if not self._destroyed:
            self._destroyed = True
            self.arbiter.shutdown()
            self.publish_state()
        return super().destroy_node()


def main(args=None):
    """Run the recording-only locomotion arbiter node."""
    rclpy.init(args=args)
    node = LocomotionArbiterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
