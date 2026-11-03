"""ROS wrapper for staged planar GLIM odometry."""

import rclpy

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from .glim_planar import GlimPlanarProjector


VALIDATED_ODOMETRY_TOPIC = (
    "/g1/navigation/glim/odometry_validated"
)
PLANAR_ODOMETRY_TOPIC = (
    "/g1/navigation/glim/odometry_planar"
)


class GlimPlanarNode(Node):
    """Project qualified GLIM poses onto an isolated robot plane."""

    def __init__(self):
        super().__init__("g1_glim_planar")

        self.projector = GlimPlanarProjector()
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
            PLANAR_ODOMETRY_TOPIC,
            self.output_qos,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            VALIDATED_ODOMETRY_TOPIC,
            self.odometry_callback,
            self.input_qos,
        )

    def odometry_callback(self, message):
        """Publish one start-relative planar sample."""

        self.odometry_publisher.publish(
            self.projector.project(message)
        )


def main(args=None):
    """Run the staged planar GLIM adapter."""

    rclpy.init(args=args)
    node = GlimPlanarNode()
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
