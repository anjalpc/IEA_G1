"""ROS adapter for fail-closed GLIM input qualification."""

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
from sensor_msgs.msg import Imu, PointCloud2

from .glim_health import GlimHealthBoundary, GlimHealthLimits


CLOUD_TOPIC = "/g1/sensors/lidar/points"
IMU_TOPIC = "/g1/sensors/lidar/imu_raw"
GLIM_ODOMETRY_TOPIC = "/glim_ros/odom_corrected"
VALIDATED_ODOMETRY_TOPIC = (
    "/g1/navigation/glim/odometry_validated"
)


def _stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class GlimHealthNode(Node):
    """Release unchanged GLIM odometry only while all inputs are healthy."""

    def __init__(self, *, parameter_overrides=None, monotonic=None):
        super().__init__(
            "g1_glim_health",
            parameter_overrides=parameter_overrides,
        )

        defaults = {
            "odometry_stale_after_seconds": 0.35,
            "cloud_stale_after_seconds": 0.35,
            "imu_stale_after_seconds": 0.10,
            "warmup_samples": 20,
            "maximum_step_translation_m": 0.15,
            "maximum_step_rotation_degrees": 15.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        limits = GlimHealthLimits(
            odometry_stale_after_seconds=self.get_parameter(
                "odometry_stale_after_seconds"
            ).value,
            cloud_stale_after_seconds=self.get_parameter(
                "cloud_stale_after_seconds"
            ).value,
            imu_stale_after_seconds=self.get_parameter(
                "imu_stale_after_seconds"
            ).value,
            warmup_samples=self.get_parameter(
                "warmup_samples"
            ).value,
            maximum_step_translation_m=self.get_parameter(
                "maximum_step_translation_m"
            ).value,
            maximum_step_rotation_rad=math.radians(
                self.get_parameter(
                    "maximum_step_rotation_degrees"
                ).value
            ),
        )
        self.boundary = GlimHealthBoundary(
            limits=limits,
            monotonic=monotonic or time.monotonic,
        )

        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odometry_input_qos = QoSProfile(
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
            VALIDATED_ODOMETRY_TOPIC,
            self.output_qos,
        )
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            CLOUD_TOPIC,
            self.cloud_callback,
            self.sensor_qos,
        )
        self.imu_subscription = self.create_subscription(
            Imu,
            IMU_TOPIC,
            self.imu_callback,
            self.sensor_qos,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            GLIM_ODOMETRY_TOPIC,
            self.odometry_callback,
            self.odometry_input_qos,
        )

    def cloud_callback(self, message):
        """Record one raw MID-360 cloud arrival."""

        self.boundary.observe_cloud(
            stamp=_stamp_seconds(message.header.stamp),
            frame_id=message.header.frame_id,
        )

    def imu_callback(self, message):
        """Record one raw MID-360 IMU arrival."""

        self.boundary.observe_imu(
            stamp=_stamp_seconds(message.header.stamp),
            frame_id=message.header.frame_id,
        )

    def odometry_callback(self, message):
        """Release an unchanged optimized sample only when healthy."""

        pose = message.pose.pose
        self.boundary.observe_odometry(
            stamp=_stamp_seconds(message.header.stamp),
            frame_id=message.header.frame_id,
            child_frame_id=message.child_frame_id,
            position=(
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ),
            orientation=(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
        )
        if self.boundary.snapshot().ready:
            self.odometry_publisher.publish(message)


def main(args=None):
    """Run the isolated GLIM health adapter."""

    rclpy.init(args=args)
    node = GlimHealthNode()
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
