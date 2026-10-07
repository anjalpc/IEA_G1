"""Lifecycle contract for the native LowState helper."""

from types import SimpleNamespace

import pytest

from g1_hardware_bridge.joint_state import helper
from g1_hardware_bridge.joint_state.protocol import (
    decode_sample,
)


class ImmediateStop:
    def __init__(self):
        self.waits = []

    def wait(self, timeout):
        self.waits.append(timeout)
        return True


class FakeSubscriber:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.callback = None
        self.instances.append(self)

    def Init(self, callback):
        self.callback = callback


def low_state():
    motors = [
        SimpleNamespace(q=0.01 * index)
        for index in range(29)
    ]
    return SimpleNamespace(
        tick=123,
        motor_state=motors,
    )


def test_initializes_read_only_domain_zero_subscriber():
    initialization = []
    lines = []
    stop = ImmediateStop()
    message_type = object()
    FakeSubscriber.instances.clear()

    helper.run_native(
        interface="eth-test",
        period_seconds=0.02,
        stop_event=stop,
        channel_factory_initialize=(
            lambda domain, interface: initialization.append(
                (domain, interface)
            )
        ),
        subscriber_factory=FakeSubscriber,
        message_type=message_type,
        monotonic=lambda: 1.0,
        emit=lines.append,
    )

    assert initialization == [(0, "eth-test")]
    assert len(FakeSubscriber.instances) == 1

    subscriber = FakeSubscriber.instances[0]
    assert subscriber.topic == "rt/lowstate"
    assert subscriber.message_type is message_type
    assert callable(subscriber.callback)
    assert stop.waits == [0.2]

    assert subscriber.callback(low_state()) is True
    assert decode_sample(lines[0]) == {
        "tick": 123,
        "positions": [0.12, 0.13, 0.14],
    }


@pytest.mark.parametrize(
    "interface",
    ["", "   ", None],
)
def test_rejects_invalid_interface_before_initialization(
    interface,
):
    initialization = []

    with pytest.raises(
        ValueError,
        match="interface",
    ):
        helper.run_native(
            interface=interface,
            period_seconds=0.02,
            stop_event=ImmediateStop(),
            channel_factory_initialize=(
                lambda *args: initialization.append(args)
            ),
            subscriber_factory=FakeSubscriber,
            message_type=object(),
        )

    assert initialization == []


def test_rejects_invalid_period_before_initialization():
    initialization = []

    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        helper.run_native(
            interface="eth-test",
            period_seconds=0.0,
            stop_event=ImmediateStop(),
            channel_factory_initialize=(
                lambda *args: initialization.append(args)
            ),
            subscriber_factory=FakeSubscriber,
            message_type=object(),
        )

    assert initialization == []


def test_parses_native_helper_arguments():
    options = helper.parse_args([
        "eth-test",
        "--period-seconds",
        "0.03",
    ])

    assert options.interface == "eth-test"
    assert options.period_seconds == 0.03


def test_main_loads_sdk_bindings_lazily(monkeypatch):
    initialization = object()
    subscriber = object()
    message_type = object()
    calls = []

    monkeypatch.setattr(
        helper,
        "_unitree_bindings",
        lambda: (
            initialization,
            subscriber,
            message_type,
        ),
    )
    monkeypatch.setattr(
        helper,
        "run_native",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        helper.signal,
        "signal",
        lambda *_args: None,
    )

    helper.main([
        "eth-test",
        "--period-seconds",
        "0.03",
    ])

    assert len(calls) == 1
    call = calls[0]
    assert call["interface"] == "eth-test"
    assert call["period_seconds"] == 0.03
    assert (
        call["channel_factory_initialize"]
        is initialization
    )
    assert call["subscriber_factory"] is subscriber
    assert call["message_type"] is message_type
    assert hasattr(call["stop_event"], "wait")
