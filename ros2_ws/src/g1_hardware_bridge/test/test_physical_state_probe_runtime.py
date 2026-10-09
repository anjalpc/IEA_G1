"""Authority boundary for the read-only physical-state probe."""

import json
from pathlib import Path
from types import SimpleNamespace

from g1_hardware_bridge.state_probe.helper import run_probe
from g1_hardware_bridge.state_probe.protocol import OUTPUT_PREFIX


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HELPER = (
    PACKAGE_ROOT
    / "g1_hardware_bridge"
    / "state_probe"
    / "helper.py"
)


def test_helper_has_no_command_or_publisher_authority():
    source = HELPER.read_text(encoding="utf-8")

    forbidden = (
        "ChannelPublisher",
        "LowCmd_",
        "MotorCmd",
        "SportClient",
        "Move(",
        "StopMove(",
    )

    assert not any(token in source for token in forbidden)


def test_helper_does_not_initialize_dds_at_import_time():
    source = HELPER.read_text(encoding="utf-8")

    assert "ChannelFactoryInitialize(" not in source
    assert "ChannelSubscriber(" not in source


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def monotonic(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class FakeSubscriber:
    def __init__(self, topic, message_type, messages, created):
        self.topic = topic
        self.message_type = message_type
        self.messages = messages
        created.append((topic, message_type))

    def Init(self, callback):
        for message in self.messages:
            callback(message)


def test_run_probe_is_bounded_read_only_and_emits_once():
    clock = FakeClock()
    initialized = []
    created = []
    emitted = []

    imu = SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.0, 0.0, 0.0],
        accelerometer=[0.0, 0.0, 9.81],
    )
    messages = {
        "rt/lowstate": [
            SimpleNamespace(tick=1, imu_state=imu),
            SimpleNamespace(tick=2, imu_state=imu),
        ],
    }
    low_state_type = type("LowState", (), {})

    run_probe(
        interface="  test-interface  ",
        duration_seconds=1.0,
        minimum_samples=2,
        candidate_types={
            "rt/lowstate": low_state_type,
            "rt/secondary_imu": None,
            "rt/sportmodestate": None,
        },
        channel_factory_initialize=lambda domain, interface: (
            initialized.append((domain, interface))
        ),
        subscriber_factory=lambda topic, message_type: FakeSubscriber(
            topic,
            message_type,
            messages[topic],
            created,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        emit=emitted.append,
    )

    assert initialized == [(0, "test-interface")]
    assert created == [("rt/lowstate", low_state_type)]
    assert len(emitted) == 1
    assert emitted[0].startswith(OUTPUT_PREFIX)

    result = json.loads(emitted[0][len(OUTPUT_PREFIX):])
    assert result["requested_duration_seconds"] == 1.0
    assert result["actual_elapsed_seconds"] == 1.0
    assert result["topics"]["rt/lowstate"]["reason"] == "VALID"
    assert (
        result["topics"]["rt/secondary_imu"]["reason"]
        == "TYPE_UNAVAILABLE"
    )
    assert (
        result["topics"]["rt/sportmodestate"]["reason"]
        == "TYPE_UNAVAILABLE"
    )


def test_run_probe_contains_malformed_callback():
    clock = FakeClock()
    emitted = []

    run_probe(
        interface="test-interface",
        duration_seconds=1.0,
        minimum_samples=1,
        candidate_types={
            "rt/lowstate": type("LowState", (), {}),
            "rt/secondary_imu": None,
            "rt/sportmodestate": None,
        },
        channel_factory_initialize=lambda _domain, _interface: None,
        subscriber_factory=lambda topic, message_type: FakeSubscriber(
            topic,
            message_type,
            [SimpleNamespace(tick=1)],
            [],
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        emit=emitted.append,
    )

    assert len(emitted) == 1
    assert "imu_state" not in emitted[0]

    result = json.loads(emitted[0][len(OUTPUT_PREFIX):])
    assert (
        result["topics"]["rt/lowstate"]["reason"]
        == "FIELDS_MISSING"
    )


def test_run_probe_contains_dds_initialization_failure():
    clock = FakeClock()
    emitted = []

    def fail_initialization(_domain, _interface):
        raise RuntimeError("private network identifier")

    run_probe(
        interface="test-interface",
        duration_seconds=1.0,
        minimum_samples=1,
        candidate_types={
            "rt/lowstate": type("LowState", (), {}),
            "rt/secondary_imu": None,
            "rt/sportmodestate": None,
        },
        channel_factory_initialize=fail_initialization,
        subscriber_factory=lambda _topic, _type: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        emit=emitted.append,
    )

    assert len(emitted) == 1
    assert "private network identifier" not in emitted[0]

    result = json.loads(emitted[0][len(OUTPUT_PREFIX):])
    assert (
        result["topics"]["rt/lowstate"]["reason"]
        == "RUNTIME_FAILURE"
    )


def test_run_probe_sanitizes_nonfinite_sample():
    clock = FakeClock()
    emitted = []
    imu = SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[float("nan"), 0.0, 0.0],
        accelerometer=[0.0, 0.0, 9.81],
    )

    run_probe(
        interface="test-interface",
        duration_seconds=1.0,
        minimum_samples=1,
        candidate_types={
            "rt/lowstate": type("LowState", (), {}),
            "rt/secondary_imu": None,
            "rt/sportmodestate": None,
        },
        channel_factory_initialize=lambda _domain, _interface: None,
        subscriber_factory=lambda topic, message_type: FakeSubscriber(
            topic,
            message_type,
            [SimpleNamespace(tick=1, imu_state=imu)],
            [],
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        emit=emitted.append,
    )

    result = json.loads(emitted[0][len(OUTPUT_PREFIX):])
    assert (
        result["topics"]["rt/lowstate"]["reason"]
        == "NONFINITE_SAMPLE"
    )
    assert "nan" not in emitted[0].lower()


def test_run_probe_contains_subscriber_failure():
    clock = FakeClock()
    emitted = []

    def fail_subscriber(_topic, _message_type):
        raise RuntimeError("private DDS detail")

    run_probe(
        interface="test-interface",
        duration_seconds=1.0,
        minimum_samples=1,
        candidate_types={
            "rt/lowstate": type("LowState", (), {}),
            "rt/secondary_imu": None,
            "rt/sportmodestate": None,
        },
        channel_factory_initialize=lambda _domain, _interface: None,
        subscriber_factory=fail_subscriber,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        emit=emitted.append,
    )

    assert len(emitted) == 1
    assert "private DDS detail" not in emitted[0]

    result = json.loads(emitted[0][len(OUTPUT_PREFIX):])
    assert (
        result["topics"]["rt/lowstate"]["reason"]
        == "RUNTIME_FAILURE"
    )
