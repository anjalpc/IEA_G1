"""Publish raw simulation state using ROS navigation messages."""

from nav_msgs.msg import Odometry
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu

from g1_simulation_bridge.state_messages import (
    build_clock_message,
    build_imu_message,
    build_odometry_message,
)


CLOCK_TOPIC = "/clock"
IMU_RAW_TOPIC = "/g1/state/imu_raw"
ODOMETRY_RAW_TOPIC = "/g1/state/odometry_raw"


def _stream_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class SimulationStatePublisher:
    """Publish clock every step and raw state at its due cadence."""

    def __init__(
        self,
        *,
        node,
        state_reader,
        schedule,
        imu_frame_id,
        odom_frame_id,
        child_frame_id,
        covariance_variance,
    ):
        self._state_reader = state_reader
        self._schedule = schedule
        self._imu_frame_id = imu_frame_id
        self._odom_frame_id = odom_frame_id
        self._child_frame_id = child_frame_id
        self._covariance_variance = covariance_variance

        self._clock_publisher = node.create_publisher(
            Clock,
            CLOCK_TOPIC,
            _stream_qos(),
        )
        self._imu_publisher = node.create_publisher(
            Imu,
            IMU_RAW_TOPIC,
            _stream_qos(),
        )
        self._odometry_publisher = node.create_publisher(
            Odometry,
            ODOMETRY_RAW_TOPIC,
            _stream_qos(),
        )

    def after_step(self, *, data, simulation_time):
        """Publish work due after one completed physics step."""

        decision = self._schedule.after_step(
            simulation_time
        )
        self._clock_publisher.publish(
            build_clock_message(decision.clock_stamp)
        )

        if not decision.publish_state:
            return

        sample = self._state_reader.sample(data)

        imu_message = build_imu_message(
            sample,
            decision.clock_stamp,
            frame_id=self._imu_frame_id,
            covariance_variance=(
                self._covariance_variance
            ),
        )
        odometry_message = build_odometry_message(
            sample,
            decision.clock_stamp,
            odom_frame_id=self._odom_frame_id,
            child_frame_id=self._child_frame_id,
            covariance_variance=(
                self._covariance_variance
            ),
        )

        self._imu_publisher.publish(imu_message)
        self._odometry_publisher.publish(
            odometry_message
        )
