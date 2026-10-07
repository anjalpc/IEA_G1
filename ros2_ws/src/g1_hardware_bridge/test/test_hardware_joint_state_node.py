"""ROS contract for the physical waist joint-state source."""

from builtin_interfaces.msg import Time
import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    ReliabilityPolicy,
)

from g1_hardware_bridge.joint_state import (
    node as joint_state_node,
)
from g1_hardware_bridge.joint_state.node import (
    JOINT_STATES_TOPIC,
    build_joint_state_message,
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
        node = joint_state_node.HardwareJointStateNode(
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

        recorder = PublisherRecorder()
        node._joint_state_publisher = recorder
        node.get_clock = lambda: FixedClock()

        monitor.options["sample_callback"]({
            "tick": 123,
            "positions": [0.12, -0.13, 0.14],
        })

        assert len(recorder.messages) == 1
        message = recorder.messages[0]
        assert message.header.stamp == Time(
            sec=7,
            nanosec=8000000,
        )
        assert list(message.position) == [
            0.12,
            -0.13,
            0.14,
        ]

        node.stop_monitor()
        assert monitor.stopped is True
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
