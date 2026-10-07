"""Read physical G1 waist state from native Unitree telemetry."""

import argparse
import signal
import sys
import threading
import time

from g1_hardware_bridge.joint_state.protocol import (
    JointStateGate,
    encode_sample,
)


ERROR_PREFIX = "JOINT_STATE_ERROR="


class LowStateReader:
    """Convert native LowState callbacks into minimal protocol lines."""

    def __init__(
        self,
        *,
        gate=None,
        emit=None,
        error_stream=sys.stderr,
    ):
        self.gate = gate or JointStateGate()
        self.emit = emit or self._emit_stdout
        self.error_stream = error_stream

    @staticmethod
    def _emit_stdout(line):
        print(line, flush=True)

    def receive_state(self, message):
        """Validate and emit one advancing due LowState sample."""
        try:
            motor_positions = [
                motor.q
                for motor in message.motor_state
            ]
            sample = self.gate.offer(
                message.tick,
                motor_positions,
            )
        except (AttributeError, TypeError, ValueError):
            print(
                ERROR_PREFIX + "invalid sample",
                file=self.error_stream,
                flush=True,
            )
            return False

        if sample is None:
            return False

        self.emit(
            encode_sample(sample)
        )
        return True


def parse_args(args=None):
    """Parse the isolated native helper arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("interface")
    parser.add_argument(
        "--period-seconds",
        type=float,
        default=0.02,
    )
    return parser.parse_args(args)


def _unitree_bindings():
    """Load robot-local SDK2 bindings only inside the helper."""
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowState_,
    )

    return (
        ChannelFactoryInitialize,
        ChannelSubscriber,
        LowState_,
    )


def run_native(
    *,
    interface,
    period_seconds,
    stop_event,
    channel_factory_initialize,
    subscriber_factory,
    message_type,
    monotonic=time.monotonic,
    emit=None,
):
    """Run one injected, read-only native LowState subscriber."""
    if (
        not isinstance(interface, str)
        or not interface.strip()
    ):
        raise ValueError(
            "network interface must not be empty"
        )

    gate = JointStateGate(
        period_seconds=period_seconds,
        monotonic=monotonic,
    )

    channel_factory_initialize(
        0,
        interface.strip(),
    )

    reader = LowStateReader(
        gate=gate,
        emit=emit,
    )
    subscriber = subscriber_factory(
        "rt/lowstate",
        message_type,
    )
    subscriber.Init(reader.receive_state)

    while not stop_event.wait(0.2):
        pass


def main(args=None):
    """Run until the supervising parent requests termination."""
    options = parse_args(args)
    (
        channel_factory_initialize,
        subscriber_factory,
        message_type,
    ) = _unitree_bindings()

    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        stop_event.set()

    previous_int = signal.signal(
        signal.SIGINT,
        request_stop,
    )
    previous_term = signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    try:
        run_native(
            interface=options.interface,
            period_seconds=options.period_seconds,
            stop_event=stop_event,
            channel_factory_initialize=(
                channel_factory_initialize
            ),
            subscriber_factory=subscriber_factory,
            message_type=message_type,
        )
    finally:
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
