"""ROS wrapper for the hardware-independent locomotion arbiter."""

import rclpy

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from g1_locomotion_interfaces.msg import LocomotionState
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
    ):
        super().__init__(
            "g1_locomotion_arbiter",
            parameter_overrides=parameter_overrides,
        )

        self.declare_parameter(
            "command_timeout_seconds",
            0.250,
        )
        self.declare_parameter("max_forward_mps", -1.0)
        self.declare_parameter("max_lateral_mps", -1.0)
        self.declare_parameter("max_yaw_rate_rps", -1.0)

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

        self.backend = backend or RecordingBackend()
        self.arbiter = LocomotionArbiter(
            backend=self.backend,
            velocity_limits=limits,
            command_timeout_seconds=timeout,
            lease_factory=lease_factory,
        )

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
