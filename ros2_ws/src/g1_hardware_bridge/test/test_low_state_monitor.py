"""Contract for the physical LowState helper monitor."""

import io
import subprocess

import pytest

from g1_hardware_bridge.low_state.monitor import (
    LowStateMonitor,
)


class FakeProcess:
    def __init__(
        self,
        lines="",
        running=True,
        resist=False,
    ):
        self.stdout = io.StringIO(lines)
        self.returncode = None if running else 0
        self.resist = resist
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.resist:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        self.waited = True
        if self.returncode is None:
            raise subprocess.TimeoutExpired(
                "joint-state monitor",
                1.0,
            )
        return self.returncode


def monitor(**overrides):
    options = {
        "interface": "eth-test",
        "period_seconds": 0.02,
        "environment": {
            "ROS_DOMAIN_ID": "0",
        },
        "sample_callback": lambda sample: None,
    }
    options.update(overrides)
    return LowStateMonitor(**options)


def test_valid_line_reaches_sample_callback():
    samples = []
    instance = monitor(
        sample_callback=samples.append
    )

    accepted = instance.consume_line(
        'LOW_STATE={'
        '"angular_velocity_xyz":[0.4,0.5,0.6],'
        '"linear_acceleration_xyz":[0.7,0.8,9.6],'
        '"orientation_wxyz":[1.0,0.0,0.0,0.0],'
        '"positions":[0.1,-0.2,0.3],'
        '"tick":7}\n'
    )

    assert accepted is True
    assert samples == [{
        "tick": 7,
        "positions": [0.1, -0.2, 0.3],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "angular_velocity_xyz": [0.4, 0.5, 0.6],
        "linear_acceleration_xyz": [0.7, 0.8, 9.6],
    }]


@pytest.mark.parametrize(
    "line",
    [
        "native startup log\n",
        "LOW_STATE=not-json\n",
        "LOW_STATE={}\n",
        (
            'LOW_STATE={"tick":1,'
            '"positions":[0.0,0.0]}\n'
        ),
    ],
)
def test_noise_and_malformed_lines_are_ignored(line):
    samples = []
    instance = monitor(
        sample_callback=samples.append
    )

    assert instance.consume_line(line) is False
    assert samples == []


def test_start_uses_isolated_helper_command():
    calls = []
    availability = []
    process = FakeProcess(running=False)

    def factory(command, **options):
        calls.append((command, options))
        return process

    instance = monitor(
        process_factory=factory,
        availability_callback=availability.append,
    )
    instance.start()
    instance._reader.join(timeout=1.0)

    command, options = calls[0]
    assert command[1:3] == [
        "-m",
        "g1_hardware_bridge.low_state.helper",
    ]
    assert command[3:] == [
        "eth-test",
        "--period-seconds",
        "0.02",
    ]
    assert options["env"]["ROS_DOMAIN_ID"] == "0"
    assert options["stdout"] == subprocess.PIPE
    assert options["stderr"] is None
    assert availability == [True, False]
    assert process.waited is True


def test_rejects_second_running_helper():
    process = FakeProcess()
    instance = monitor(
        process_factory=(
            lambda *_args, **_kwargs: process
        )
    )
    instance.start()

    with pytest.raises(
        RuntimeError,
        match="already running",
    ):
        instance.start()

    instance.stop()


def test_stop_terminates_and_reaps_helper():
    process = FakeProcess()
    availability = []
    instance = monitor(
        process_factory=(
            lambda *_args, **_kwargs: process
        ),
        availability_callback=availability.append,
    )

    instance.start()
    instance.stop()

    assert process.terminated is True
    assert process.killed is False
    assert process.waited is True
    assert availability[-1] is False


def test_stop_escalates_when_helper_resists():
    process = FakeProcess(resist=True)
    instance = monitor(
        process_factory=(
            lambda *_args, **_kwargs: process
        )
    )

    instance.start()
    instance.stop(timeout=0.01)

    assert process.terminated is True
    assert process.killed is True
    assert process.waited is True


def test_stop_is_idempotent_before_start():
    monitor().stop()
