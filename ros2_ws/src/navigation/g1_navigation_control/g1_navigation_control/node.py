"""ROS adapter from Nav2 velocity to leased locomotion."""

import math
import time

import rclpy

from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from g1_locomotion_interfaces.msg import LocomotionCommand
from g1_locomotion_interfaces.srv import (
    AcquireLocomotion,
    ReleaseLocomotion,
)
from g1_robot_interfaces.locomotion import (
    OWNER_NAVIGATION,
    PlanarVelocityLimits,
)

from g1_navigation_control.command_gate import (
    NavigationCommandGate,
)
from g1_navigation_control.messages import (
    build_locomotion_command,
)


class NavigationCommandAdapterNode(Node):
    """Request navigation authority and forward safe commands."""

    def __init__(
        self,
        *,
        parameter_overrides=None,
        monotonic=time.monotonic,
    ):
        super().__init__(
            "g1_navigation_command_adapter",
            parameter_overrides=parameter_overrides,
        )

        self.declare_parameter(
            "requester_id",
            "g1-navigation-control",
        )
        self.declare_parameter("max_forward_mps", 0.15)
        self.declare_parameter("max_lateral_mps", 0.01)
        self.declare_parameter("max_yaw_rate_rps", 0.40)
        self.declare_parameter("allow_lateral", False)
        self.declare_parameter(
            "command_timeout_seconds",
            0.25,
        )
        self.declare_parameter("watchdog_rate_hz", 20.0)

        self.requester_id = str(
            self.get_parameter("requester_id").value
        )
        if not self.requester_id.strip():
            raise ValueError("requester_id must be nonempty")

        timeout = float(
            self.get_parameter(
                "command_timeout_seconds"
            ).value
        )
        watchdog_rate = float(
            self.get_parameter("watchdog_rate_hz").value
        )
        for name, value in (
            ("command_timeout_seconds", timeout),
            ("watchdog_rate_hz", watchdog_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        self._monotonic = monotonic
        self._command_timeout_seconds = timeout

        self.gate = NavigationCommandGate(
            limits=PlanarVelocityLimits(
                forward_mps=self.get_parameter(
                    "max_forward_mps"
                ).value,
                lateral_mps=self.get_parameter(
                    "max_lateral_mps"
                ).value,
                yaw_rate_rps=self.get_parameter(
                    "max_yaw_rate_rps"
                ).value,
            ),
            allow_lateral=bool(
                self.get_parameter("allow_lateral").value
            ),
        )

        self.raw_command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.locomotion_command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.command_publisher = self.create_publisher(
            LocomotionCommand,
            "/g1/locomotion/command",
            self.locomotion_command_qos,
        )
        self.raw_command_subscription = (
            self.create_subscription(
                Twist,
                "/g1/navigation/cmd_vel_raw",
                self.command_callback,
                self.raw_command_qos,
            )
        )
        self.acquire_client = self.create_client(
            AcquireLocomotion,
            "/g1/locomotion/acquire",
        )
        self.release_client = self.create_client(
            ReleaseLocomotion,
            "/g1/locomotion/release",
        )
        self.watchdog_timer = self.create_timer(
            1.0 / watchdog_rate,
            self.watchdog_callback,
        )

        self._acquire_future = None
        self._release_future = None
        self._last_command_at = None
        self._destroyed = False

    def request_navigation_lease(self):
        """Request navigation ownership without blocking ROS."""
        if (
            self.gate.active
            or self._acquire_future is not None
            or self._release_future is not None
        ):
            return
        if not self.acquire_client.service_is_ready():
            return

        request = AcquireLocomotion.Request()
        request.requested_owner = OWNER_NAVIGATION
        request.requester_id = self.requester_id

        self._acquire_future = self.acquire_client.call_async(
            request
        )
        self._acquire_future.add_done_callback(
            self.acquire_response_callback
        )

    def acquire_response_callback(self, future):
        """Activate only a valid granted navigation lease."""
        self._acquire_future = None
        try:
            response = future.result()
            if (
                response.granted
                and response.active_owner == OWNER_NAVIGATION
                and response.lease_id
                and not self.gate.active
            ):
                self.gate.grant(response.lease_id)
                self._last_command_at = self._monotonic()
        except Exception:
            self.get_logger().warning(
                "Navigation lease request failed"
            )

    def release_response_callback(self, future):
        """Finish a release attempt without exposing details."""
        self._release_future = None
        try:
            future.result()
        except Exception:
            self.get_logger().warning(
                "Navigation lease release failed"
            )

    def publish_command(self, command):
        """Publish one already validated leased command."""
        self.command_publisher.publish(
            build_locomotion_command(
                command=command,
                stamp=self.get_clock().now().to_msg(),
            )
        )

    def stop_and_release(self):
        """Publish zero, clear local authority, and release."""
        stop = self.gate.stop()
        if stop is None:
            return

        lease_id = stop.lease_id
        self.publish_command(stop)
        self.gate.clear()
        self._last_command_at = None

        if not self.release_client.service_is_ready():
            return

        request = ReleaseLocomotion.Request()
        request.owner = OWNER_NAVIGATION
        request.lease_id = lease_id

        try:
            self._release_future = (
                self.release_client.call_async(request)
            )
            self._release_future.add_done_callback(
                self.release_response_callback
            )
        except Exception:
            self._release_future = None
            self.get_logger().warning(
                "Navigation lease release failed"
            )

    def command_callback(self, message):
        """Validate and forward one Nav2 velocity command."""
        if not self.gate.active:
            self.request_navigation_lease()
            return

        try:
            command = self.gate.translate(
                linear_x=message.linear.x,
                linear_y=message.linear.y,
                linear_z=message.linear.z,
                angular_x=message.angular.x,
                angular_y=message.angular.y,
                angular_z=message.angular.z,
            )
        except ValueError:
            self.get_logger().error(
                "Rejected unsafe navigation velocity"
            )
            self.stop_and_release()
            return

        if command is None:
            return

        self.publish_command(command)
        self._last_command_at = self._monotonic()

    def watchdog_callback(self):
        """Release navigation when its command stream stalls."""
        if (
            not self.gate.active
            or self._last_command_at is None
        ):
            return

        age = self._monotonic() - self._last_command_at
        if age >= self._command_timeout_seconds:
            self.get_logger().warning(
                "Navigation command stream timed out"
            )
            self.stop_and_release()



    def destroy_node(self):
        """Stop and release navigation before ROS teardown."""
        if self._destroyed:
            return True

        self._destroyed = True
        self.stop_and_release()
        return super().destroy_node()


def main(args=None):
    """Run the Nav2 command adapter."""
    rclpy.init(args=args)
    node = NavigationCommandAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
