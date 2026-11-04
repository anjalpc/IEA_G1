"""ROS authority boundary for qualified physical odometry."""

import copy
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from tf2_ros import TransformBroadcaster

from .physical_authority import (
    PhysicalAuthorityBoundary,
    PhysicalAuthorityLimits,
)
from .planar_transform import (
    odometry_to_base_footprint_transform,
    odometry_to_pelvis_transform,
)


STAGED_ODOMETRY_TOPIC = (
    "/g1/navigation/physical/odometry_staged"
)
ODOMETRY_TOPIC = "/g1/state/odometry"


def _canonical_base_odometry(source):
    """Project qualified pelvis state onto canonical base odometry."""

    base_transform = odometry_to_base_footprint_transform(
        source
    )
    canonical = copy.deepcopy(source)
    canonical.child_frame_id = base_transform.child_frame_id

    canonical.pose.pose.position.x = (
        base_transform.transform.translation.x
    )
    canonical.pose.pose.position.y = (
        base_transform.transform.translation.y
    )
    canonical.pose.pose.position.z = 0.0
    canonical.pose.pose.orientation = (
        base_transform.transform.rotation
    )

    canonical.twist.twist.linear.z = 0.0
    canonical.twist.twist.angular.x = 0.0
    canonical.twist.twist.angular.y = 0.0
    return canonical


class PhysicalAuthorityNode(Node):
    """Publish canonical physical odometry only after qualification."""

    def __init__(
        self,
        *,
        parameter_overrides=None,
        monotonic=None,
    ):
        super().__init__(
            "g1_physical_odometry_authority",
            parameter_overrides=parameter_overrides,
        )

        defaults = {
            "warmup_samples": 20,
            "stale_after_seconds": 0.25,
            "maximum_translation_step": 0.15,
            "maximum_rotation_step_degrees": 15.0,
            "maximum_translation_rate": 0.5,
            "maximum_rotation_rate_degrees": 60.0,
            "maximum_motion_gap_seconds": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.boundary = PhysicalAuthorityBoundary(
            limits=PhysicalAuthorityLimits(
                warmup_samples=self.get_parameter(
                    "warmup_samples"
                ).value,
                stale_after_seconds=self.get_parameter(
                    "stale_after_seconds"
                ).value,
                maximum_translation_step=self.get_parameter(
                    "maximum_translation_step"
                ).value,
                maximum_rotation_step=math.radians(
                    self.get_parameter(
                        "maximum_rotation_step_degrees"
                    ).value
                ),
                maximum_translation_rate=self.get_parameter(
                    "maximum_translation_rate"
                ).value,
                maximum_rotation_rate=math.radians(
                    self.get_parameter(
                        "maximum_rotation_rate_degrees"
                    ).value
                ),
                maximum_motion_gap_seconds=self.get_parameter(
                    "maximum_motion_gap_seconds"
                ).value,
            ),
            monotonic=monotonic or time.monotonic,
        )

        self.input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.odometry_publisher = self.create_publisher(
            Odometry,
            ODOMETRY_TOPIC,
            self.output_qos,
        )
        self.transform_broadcaster = TransformBroadcaster(
            self
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            STAGED_ODOMETRY_TOPIC,
            self.odometry_callback,
            self.input_qos,
        )

    def odometry_callback(self, message):
        """Publish one qualified canonical odometry update and TF."""

        result = self.boundary.observe(message)
        if result.odometry is None:
            return

        canonical = _canonical_base_odometry(
            result.odometry
        )
        self.odometry_publisher.publish(canonical)
        self.transform_broadcaster.sendTransform([
            odometry_to_base_footprint_transform(
                result.odometry
            ),
            odometry_to_pelvis_transform(
                result.odometry
            ),
        ])


def main(args=None):
    """Run the physical odometry authority boundary."""

    rclpy.init(args=args)
    node = PhysicalAuthorityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
