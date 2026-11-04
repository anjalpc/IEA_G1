"""Publish validated physical LowState on application ROS."""

import math
import signal
import threading
import time

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
        shutdown_callback=None,
        monotonic_clock=time.monotonic,
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
        self.declare_parameter(
            "state_sample_timeout_s",
            0.5,
        )
        self.declare_parameter(
            "state_helper_restart_delay_s",
            3.0,
        )
        self.declare_parameter(
            "state_helper_restart_limit",
            3,
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
        sample_timeout = float(
            self.get_parameter(
                "state_sample_timeout_s"
            ).value
        )
        restart_delay = float(
            self.get_parameter(
                "state_helper_restart_delay_s"
            ).value
        )
        restart_limit = int(
            self.get_parameter(
                "state_helper_restart_limit"
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

        if (
            not math.isfinite(sample_timeout)
            or sample_timeout <= 0.0
        ):
            raise ValueError(
                "state_sample_timeout_s must be "
                "finite and positive"
            )
        if (
            not math.isfinite(restart_delay)
            or restart_delay <= 0.0
        ):
            raise ValueError(
                "state_helper_restart_delay_s must be "
                "finite and positive"
            )
        if restart_limit < 1:
            raise ValueError(
                "state_helper_restart_limit must be positive"
            )

        self._sample_lock = threading.Lock()
        self._pending_sample = None
        self._monotonic_clock = monotonic_clock
        self._sample_timeout = sample_timeout
        self._restart_delay = restart_delay
        self._restart_limit = restart_limit
        self._last_sample_time = monotonic_clock()
        self._restart_due_time = None
        self._restart_attempts = 0
        self._stopping = False
        self._shutdown_requested = False

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
            sample_callback=self._queue_sample,
            availability_callback=(
                self._source_availability_changed
            ),
        )
        self._publish_timer = self.create_timer(
            period_seconds,
            self._publish_pending_sample,
        )
        self._monitor.start()

        self.get_logger().warning(
            "Physical LowState source ready"
        )

    def _queue_sample(self, sample):
        with self._sample_lock:
            self._pending_sample = sample
            self._last_sample_time = self._monotonic_clock()
            self._restart_due_time = None
            self._restart_attempts = 0

    def _publish_pending_sample(self):
        self._check_source_health()

        with self._sample_lock:
            sample = self._pending_sample
            self._pending_sample = None

        if sample is not None:
            self._publish_sample(sample)

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

    def _check_source_health(self):
        now = self._monotonic_clock()
        action = None
        attempt = None

        with self._sample_lock:
            if self._stopping:
                return

            if self._restart_due_time is not None:
                if now < self._restart_due_time:
                    return

                if self._restart_attempts >= self._restart_limit:
                    self._stopping = True
                    self._shutdown_requested = True
                    action = "shutdown"
                else:
                    self._restart_attempts += 1
                    attempt = self._restart_attempts
                    self._last_sample_time = now
                    self._restart_due_time = None
                    action = "start"
            elif (
                now - self._last_sample_time
                > self._sample_timeout
            ):
                self._restart_due_time = (
                    now + self._restart_delay
                )
                action = "stop"

        if action == "stop":
            self.get_logger().warning(
                "Physical LowState samples stale; "
                "stopping helper before recovery"
            )
            self._monitor.stop()
        elif action == "start":
            self.get_logger().warning(
                "Restarting Physical LowState helper "
                f"(attempt {attempt}/{self._restart_limit})"
            )
            try:
                self._monitor.start()
            except Exception as error:
                self.get_logger().error(
                    "Physical LowState helper restart failed: "
                    f"{type(error).__name__}"
                )
                with self._sample_lock:
                    self._restart_due_time = (
                        self._monotonic_clock()
                        + self._restart_delay
                    )
        elif action == "shutdown":
            self.get_logger().error(
                "Physical LowState recovery exhausted; "
                "shutting down the supported stack"
            )
            if self._shutdown_callback is not None:
                self._shutdown_callback()

    @property
    def shutdown_requested(self):
        return self._shutdown_requested

    def _source_availability_changed(self, available):
        if available:
            self.get_logger().info(
                "Physical LowState helper available"
            )
        else:
            self.get_logger().warning(
                "Physical LowState helper unavailable; "
                "recovery scheduled"
            )
            with self._sample_lock:
                if (
                    not self._stopping
                    and self._restart_due_time is None
                ):
                    self._restart_due_time = (
                        self._monotonic_clock()
                        + self._restart_delay
                    )

    def stop_monitor(self):
        with self._sample_lock:
            self._stopping = True
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
            and not node.shutdown_requested
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
