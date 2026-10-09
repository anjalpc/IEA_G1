"""ROS contract for the physical waist joint-state source."""

from builtin_interfaces.msg import Time
import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    ReliabilityPolicy,
)

from g1_hardware_bridge.low_state import (
    node as hardware_state_node,
)
from g1_hardware_bridge.low_state.node import (
    IMU_FRAME,
    IMU_TOPIC,
    JOINT_STATES_TOPIC,
    build_imu_message,
    build_joint_state_message,
    imu_qos,
    joint_state_qos,
)


def test_builds_standard_measured_joint_state():
    stamp = Time(
        sec=12,
        nanosec=345678901,
    )
    message = build_joint_state_message(
        {
            "tick": 123,
            "positions": [0.12, -0.13, 0.14],
        },
        stamp,
    )

    assert message.header.stamp == stamp
    assert list(message.name) == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    assert list(message.position) == [
        0.12,
        -0.13,
        0.14,
    ]
    assert list(message.velocity) == []
    assert list(message.effort) == []


def test_uses_reliable_volatile_depth_one_qos():
    qos = joint_state_qos()

    assert JOINT_STATES_TOPIC == "/joint_states"
    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 1
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.VOLATILE


def test_builds_raw_pelvis_imu_message():
    stamp = Time(
        sec=12,
        nanosec=345678901,
    )
    message = build_imu_message(
        {
            "orientation_wxyz": [
                0.182574,
                0.365148,
                0.547723,
                0.730297,
            ],
            "angular_velocity_xyz": [0.1, -0.2, 0.3],
            "linear_acceleration_xyz": [0.4, 0.5, 9.6],
        },
        stamp,
    )

    assert message.header.stamp == stamp
    assert message.header.frame_id == "imu_in_pelvis"
    assert message.orientation.x == 0.365148
    assert message.orientation.y == 0.547723
    assert message.orientation.z == 0.730297
    assert message.orientation.w == 0.182574
    assert message.angular_velocity.x == 0.1
    assert message.angular_velocity.y == -0.2
    assert message.angular_velocity.z == 0.3
    assert message.linear_acceleration.x == 0.4
    assert message.linear_acceleration.y == 0.5
    assert message.linear_acceleration.z == 9.6
    assert list(message.orientation_covariance) == [0.0] * 9
    assert list(message.angular_velocity_covariance) == [0.0] * 9
    assert list(message.linear_acceleration_covariance) == [0.0] * 9


def test_uses_best_effort_volatile_depth_one_imu_qos():
    qos = imu_qos()

    assert IMU_TOPIC == "/g1/state/imu_raw"
    assert IMU_FRAME == "imu_in_pelvis"
    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 1
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE


def test_node_supervises_monitor_and_publishes_callbacks(
    monkeypatch,
):
    monitors = []
    shutdown_requests = []

    class RecordingMonitor:
        def __init__(self, **options):
            self.options = options
            self.started = False
            self.stopped = False
            monitors.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    class PublisherRecorder:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class FixedNow:
        @staticmethod
        def to_msg():
            return Time(
                sec=7,
                nanosec=8000000,
            )

    class FixedClock:
        @staticmethod
        def now():
            return FixedNow()

    monkeypatch.setenv(
        "G1_NETWORK_INTERFACE",
        "eth-test",
    )

    rclpy.init()
    node = None
    try:
        node = hardware_state_node.HardwareStateNode(
            monitor_factory=RecordingMonitor,
            environment_factory=lambda: {
                "ROS_DOMAIN_ID": "0",
            },
            shutdown_callback=(
                lambda: shutdown_requests.append(True)
            ),
        )

        assert len(monitors) == 1
        monitor = monitors[0]
        assert monitor.started is True
        assert monitor.options["interface"] == "eth-test"
        assert monitor.options["period_seconds"] == 0.02
        assert monitor.options["environment"] == {
            "ROS_DOMAIN_ID": "0",
        }
        assert callable(
            monitor.options["sample_callback"]
        )
        assert callable(
            monitor.options["availability_callback"]
        )

        monitor.options[
            "availability_callback"
        ](False)
        assert shutdown_requests == [True]

        assert (
            node._joint_state_publisher.topic_name
            == JOINT_STATES_TOPIC
        )
        assert (
            node._imu_publisher.topic_name
            == IMU_TOPIC
        )

        joint_recorder = PublisherRecorder()
        imu_recorder = PublisherRecorder()
        node._joint_state_publisher = joint_recorder
        node._imu_publisher = imu_recorder
        node.get_clock = lambda: FixedClock()

        monitor.options["sample_callback"]({
            "tick": 123,
            "positions": [0.12, -0.13, 0.14],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "angular_velocity_xyz": [0.1, -0.2, 0.3],
            "linear_acceleration_xyz": [0.4, 0.5, 9.6],
        })

        assert len(joint_recorder.messages) == 1
        assert len(imu_recorder.messages) == 1

        joint_message = joint_recorder.messages[0]
        imu_message = imu_recorder.messages[0]

        expected_stamp = Time(
            sec=7,
            nanosec=8000000,
        )
        assert joint_message.header.stamp == expected_stamp
        assert imu_message.header.stamp == expected_stamp
        assert list(joint_message.position) == [
            0.12,
            -0.13,
            0.14,
        ]
        assert imu_message.header.frame_id == IMU_FRAME
        assert imu_message.orientation.w == 1.0
        assert imu_message.angular_velocity.x == 0.1
        assert imu_message.linear_acceleration.z == 9.6

        node.stop_monitor()
        assert monitor.stopped is True
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
