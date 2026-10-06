"""ROS wrapper for validated navigation state publication."""

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
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from .gateway import StateValidationGateway
from .planar_transform import (
    odometry_to_base_footprint_transform,
)


IMU_RAW_TOPIC = "/g1/state/imu_raw"
ODOMETRY_RAW_TOPIC = "/g1/state/odometry_raw"
ODOMETRY_TOPIC = "/g1/state/odometry"


class StateEstimationNode(Node):
    """Publish odometry only while raw state is validated."""

    def __init__(
        self,
        *,
        parameter_overrides=None,
        monotonic=None,
    ):
        super().__init__(
            "g1_state_estimation",
            parameter_overrides=parameter_overrides,
        )

        self.declare_parameter(
            "stale_after_seconds",
            0.1,
        )
        stale_after_seconds = self.get_parameter(
            "stale_after_seconds"
        ).value

        self.gateway = StateValidationGateway(
            stale_after_seconds=stale_after_seconds,
            monotonic=monotonic or time.monotonic,
        )

        self.input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
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
        self.imu_subscription = self.create_subscription(
            Imu,
            IMU_RAW_TOPIC,
            self.imu_callback,
            self.input_qos,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            ODOMETRY_RAW_TOPIC,
            self.odometry_callback,
            self.input_qos,
        )

    def imu_callback(self, message):
        """Validate and record one raw IMU message."""

        self.gateway.observe_imu(message)

    def odometry_callback(self, message):
        """Publish raw odometry only when the gateway releases it."""

        update = self.gateway.observe_odometry(message)
        if update.odometry is not None:
            self.odometry_publisher.publish(
                update.odometry
            )
            self.transform_broadcaster.sendTransform(
                odometry_to_base_footprint_transform(
                    update.odometry
                )
            )


def main(args=None):
    """Run the validated state-estimation boundary."""

    rclpy.init(args=args)
    node = StateEstimationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
