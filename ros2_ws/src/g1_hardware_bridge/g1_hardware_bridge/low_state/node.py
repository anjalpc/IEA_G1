"""Publish validated physical LowState on application ROS."""

import math
import signal
import threading

import rclpy
from rclpy.node import Node

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, JointState

from g1_hardware_bridge.low_state.monitor import (
    LowStateMonitor,
)
from g1_hardware_bridge.low_state.protocol import (
    JOINT_NAMES,
)
from g1_hardware_bridge.native_runtime import (
    native_dds_environment,
    network_interface,
)


JOINT_STATES_TOPIC = "/joint_states"
IMU_TOPIC = "/g1/state/imu_raw"
IMU_FRAME = "imu_in_pelvis"


def joint_state_qos():
    """Return the physical joint-state publisher QoS."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def imu_qos():
    """Return the physical raw-IMU publisher QoS."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def build_joint_state_message(sample, stamp):
    """Build one standard measured waist JointState message."""
    message = JointState()
    message.header.stamp = stamp
    message.name = JOINT_NAMES
    message.position = sample["positions"]
    return message


def build_imu_message(sample, stamp):
    """Build one raw measured pelvis Imu message."""
    message = Imu()
    message.header.stamp = stamp
    message.header.frame_id = IMU_FRAME

    w, x, y, z = sample["orientation_wxyz"]
    message.orientation.x = x
    message.orientation.y = y
    message.orientation.z = z
    message.orientation.w = w

    angular_x, angular_y, angular_z = (
        sample["angular_velocity_xyz"]
    )
    message.angular_velocity.x = angular_x
    message.angular_velocity.y = angular_y
    message.angular_velocity.z = angular_z

    linear_x, linear_y, linear_z = (
        sample["linear_acceleration_xyz"]
    )
    message.linear_acceleration.x = linear_x
    message.linear_acceleration.y = linear_y
    message.linear_acceleration.z = linear_z

    message.orientation_covariance = [0.0] * 9
    message.angular_velocity_covariance = [0.0] * 9
    message.linear_acceleration_covariance = [0.0] * 9
    return message


class HardwareStateNode(Node):
    """Bridge validated native LowState telemetry to application ROS."""

    def __init__(
        self,
        *,
        monitor_factory=LowStateMonitor,
        environment_factory=native_dds_environment,
        shutdown_callback=rclpy.shutdown,
    ):
        super().__init__("g1_hardware_state")

        self._shutdown_callback = shutdown_callback

        self.declare_parameter(
            "network_interface",
            network_interface(),
        )
        self.declare_parameter(
            "state_publish_period_s",
            0.02,
        )

        interface = str(
            self.get_parameter(
                "network_interface"
            ).value
        )
        period_seconds = float(
            self.get_parameter(
                "state_publish_period_s"
            ).value
        )

        if not interface.strip():
            raise ValueError(
                "network_interface must not be empty"
            )
        if (
            not math.isfinite(period_seconds)
            or period_seconds <= 0.0
        ):
            raise ValueError(
                "state_publish_period_s must be "
                "finite and positive"
            )

        self._joint_state_publisher = (
            self.create_publisher(
                JointState,
                JOINT_STATES_TOPIC,
                joint_state_qos(),
            )
        )
        self._imu_publisher = self.create_publisher(
            Imu,
            IMU_TOPIC,
            imu_qos(),
        )

        self._monitor = monitor_factory(
            interface=interface,
            period_seconds=period_seconds,
            environment=environment_factory(),
            sample_callback=self._publish_sample,
            availability_callback=(
                self._source_availability_changed
            ),
        )
        self._monitor.start()

        self.get_logger().warning(
            "Physical LowState source ready"
        )

    def _publish_sample(self, sample):
        stamp = self.get_clock().now().to_msg()
        joint_message = build_joint_state_message(
            sample,
            stamp,
        )
        imu_message = build_imu_message(
            sample,
            stamp,
        )
        self._joint_state_publisher.publish(
            joint_message
        )
        self._imu_publisher.publish(
            imu_message
        )

    def _source_availability_changed(self, available):
        if available:
            self.get_logger().info(
                "Physical LowState helper available"
            )
        else:
            self.get_logger().error(
                "Physical LowState helper unavailable"
            )
            self._shutdown_callback()

    def stop_monitor(self):
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.stop()


def main(args=None):
    """Run with explicit helper cleanup on every shutdown path."""
    if SignalHandlerOptions is None:
        rclpy.init(args=args)
    else:
        rclpy.init(
            args=args,
            signal_handler_options=SignalHandlerOptions.NO,
        )

    node = HardwareStateNode()
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    previous_int = signal.signal(
        signal.SIGINT,
        request_stop,
    )
    previous_term = signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    try:
        while (
            rclpy.ok()
            and not stop_requested.is_set()
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.2,
            )
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_monitor()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        signal.signal(
            signal.SIGINT,
            previous_int,
        )
        signal.signal(
            signal.SIGTERM,
            previous_term,
        )


if __name__ == "__main__":
    main()
