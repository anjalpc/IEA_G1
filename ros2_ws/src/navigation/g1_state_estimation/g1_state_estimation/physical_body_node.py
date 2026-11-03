"""ROS wrapper for guarded physical-body odometry staging."""

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
from sensor_msgs.msg import Imu

from .pelvis_imu_health import (
    PelvisImuHealthBoundary,
    PelvisImuHealthLimits,
)
from .physical_body import PhysicalBodyProjector


PLANAR_ODOMETRY_TOPIC = (
    "/g1/navigation/glim/odometry_planar"
)
PELVIS_IMU_TOPIC = "/g1/state/imu_raw"
STAGED_ODOMETRY_TOPIC = (
    "/g1/navigation/physical/odometry_staged"
)
MAX_PENDING_ODOMETRY = 20
MAX_IMU_HISTORY = 100


def _stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class PhysicalBodyNode(Node):
    """Stage calibrated body odometry without navigation authority."""

    def __init__(
        self,
        *,
        parameter_overrides=None,
        monotonic=None,
    ):
        super().__init__(
            "g1_physical_body",
            parameter_overrides=parameter_overrides,
        )

        defaults = {
            "pelvis_height_m": 0.785,
            "imu_stale_after_seconds": 0.10,
            "imu_warmup_samples": 20,
            "minimum_acceleration_norm": 4.0,
            "maximum_acceleration_norm": 15.0,
            "maximum_angular_velocity_norm": 10.0,
            "maximum_sync_offset_seconds": 0.05,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        maximum_sync_offset = self.get_parameter(
            "maximum_sync_offset_seconds"
        ).value
        if (
            not math.isfinite(maximum_sync_offset)
            or maximum_sync_offset <= 0.0
        ):
            raise ValueError(
                "maximum sync offset must be finite and positive"
            )
        self.maximum_sync_offset_seconds = maximum_sync_offset

        self.projector = PhysicalBodyProjector(
            pelvis_height_m=self.get_parameter(
                "pelvis_height_m"
            ).value,
        )
        self.imu_health = PelvisImuHealthBoundary(
            limits=PelvisImuHealthLimits(
                stale_after_seconds=self.get_parameter(
                    "imu_stale_after_seconds"
                ).value,
                warmup_samples=self.get_parameter(
                    "imu_warmup_samples"
                ).value,
                minimum_acceleration_norm=self.get_parameter(
                    "minimum_acceleration_norm"
                ).value,
                maximum_acceleration_norm=self.get_parameter(
                    "maximum_acceleration_norm"
                ).value,
                maximum_angular_velocity_norm=self.get_parameter(
                    "maximum_angular_velocity_norm"
                ).value,
            ),
            monotonic=monotonic or time.monotonic,
        )
        self.latest_imu = None
        self.imu_history = []
        self.pending_odometry = []

        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odometry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.odometry_publisher = self.create_publisher(
            Odometry,
            STAGED_ODOMETRY_TOPIC,
            self.odometry_qos,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            PLANAR_ODOMETRY_TOPIC,
            self.odometry_callback,
            self.odometry_qos,
        )
        self.imu_subscription = self.create_subscription(
            Imu,
            PELVIS_IMU_TOPIC,
            self.imu_callback,
            self.sensor_qos,
        )

    def imu_callback(self, message):
        """Qualify and retain one atomic pelvis-IMU sample."""

        self.imu_health.observe(
            stamp=_stamp_seconds(message.header.stamp),
            frame_id=message.header.frame_id,
            orientation=(
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
                message.orientation.w,
            ),
            angular_velocity=(
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ),
            linear_acceleration=(
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ),
            orientation_covariance=tuple(
                message.orientation_covariance
            ),
            angular_velocity_covariance=tuple(
                message.angular_velocity_covariance
            ),
            linear_acceleration_covariance=tuple(
                message.linear_acceleration_covariance
            ),
        )
        self.latest_imu = message
        self.imu_history.append(message)
        self.imu_history = self.imu_history[
            -MAX_IMU_HISTORY:
        ]
        self._publish_pending_odometry()

    def odometry_callback(self, message):
        """Retain qualified odometry until matching IMU arrives."""

        if not self.imu_health.snapshot().ready:
            return

        self.pending_odometry.append(message)
        self.pending_odometry = self.pending_odometry[
            -MAX_PENDING_ODOMETRY:
        ]
        self._publish_pending_odometry()

    def _publish_pending_odometry(self):
        """Publish odometry with the nearest timestamped IMU."""

        if not self.imu_health.snapshot().ready:
            return
        if not self.imu_history:
            return

        stamped_imus = [
            (_stamp_seconds(message.header.stamp), message)
            for message in self.imu_history
        ]
        stamped_imus = [
            (stamp, message)
            for stamp, message in stamped_imus
            if math.isfinite(stamp)
        ]
        if not stamped_imus:
            self.pending_odometry = []
            return

        newest_imu_stamp = stamped_imus[-1][0]
        remaining = []

        for message in self.pending_odometry:
            odometry_stamp = _stamp_seconds(
                message.header.stamp
            )
            if not math.isfinite(odometry_stamp):
                continue

            imu_stamp, matched_imu = min(
                stamped_imus,
                key=lambda item: abs(
                    item[0] - odometry_stamp
                ),
            )
            offset = imu_stamp - odometry_stamp

            if abs(offset) <= self.maximum_sync_offset_seconds:
                try:
                    output = self.projector.project(
                        message,
                        matched_imu,
                    )
                except ValueError:
                    continue
                self.odometry_publisher.publish(output)
            elif (
                newest_imu_stamp
                < odometry_stamp
                - self.maximum_sync_offset_seconds
            ):
                remaining.append(message)

        self.pending_odometry = remaining


def main(args=None):
    """Run the guarded physical-body staging adapter."""

    rclpy.init(args=args)
    node = PhysicalBodyNode()
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
