"""CLI and lazy binding resolution for physical state inspection."""

import argparse
import importlib
import math
import threading
import time

from g1_hardware_bridge.state_probe.protocol import (
    REASON_FIELDS_MISSING,
    REASON_NONFINITE_SAMPLE,
    REASON_RUNTIME_FAILURE,
    ProbeSummary,
    encode_summary,
    normalize_observation,
)


CANDIDATE_BINDINGS = {
    "rt/lowstate": (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "LowState_",
    ),
    "rt/secondary_imu": (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "IMUState_",
    ),
    "rt/sportmodestate": (
        "unitree_sdk2py.idl.unitree_go.msg.dds_",
        "SportModeState_",
    ),
}


def _nonempty_string(value):
    value = str(value).strip()
    if not value:
        raise argparse.ArgumentTypeError(
            "value must be nonempty"
        )
    return value


def _positive_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "value must be a number"
        ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and positive"
        )
    return value


def _positive_integer(value):
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "value must be an integer"
        ) from error
    if value <= 0:
        raise argparse.ArgumentTypeError(
            "value must be positive"
        )
    return value


def extract_candidate_observation(
    topic,
    message,
    callback_sequence,
):
    """Extract only sanitized metadata from an approved message."""
    if topic == "rt/lowstate":
        imu = message.imu_state
        progress = message.tick
        fields = {
            "quaternion": imu.quaternion,
            "gyroscope": imu.gyroscope,
            "accelerometer": imu.accelerometer,
        }
    elif topic == "rt/secondary_imu":
        progress = callback_sequence
        fields = {
            "quaternion": message.quaternion,
            "gyroscope": message.gyroscope,
            "accelerometer": message.accelerometer,
        }
    elif topic == "rt/sportmodestate":
        progress = callback_sequence
        fields = {
            "position": message.position,
            "velocity": message.velocity,
        }
    else:
        raise ValueError("state-probe topic is not approved")

    return normalize_observation(
        topic=topic,
        progress=progress,
        fields=fields,
    )


def run_probe(
    interface,
    duration_seconds,
    minimum_samples,
    candidate_types,
    channel_factory_initialize,
    subscriber_factory,
    monotonic,
    sleep,
    emit,
):
    """Observe approved DDS state streams for one bounded interval."""
    interface = str(interface).strip()
    if not interface:
        raise ValueError("interface must be nonempty")

    summary = ProbeSummary(minimum_samples=minimum_samples)
    callback_sequences = {
        topic: 0
        for topic in CANDIDATE_BINDINGS
    }
    subscribers = []
    summary_lock = threading.Lock()
    started_at = monotonic()

    for topic in CANDIDATE_BINDINGS:
        if candidate_types.get(topic) is None:
            summary.mark_unavailable(topic)

    def finish():
        with summary_lock:
            result = summary.finalize(
                requested_duration_seconds=duration_seconds,
                actual_elapsed_seconds=max(
                    0.0,
                    monotonic() - started_at,
                ),
            )
        emit(encode_summary(result))
        return result

    try:
        channel_factory_initialize(0, interface)
    except Exception:
        for topic in CANDIDATE_BINDINGS:
            if candidate_types.get(topic) is not None:
                summary.mark_failure(
                    topic,
                    REASON_RUNTIME_FAILURE,
                )
        return finish()

    for topic in CANDIDATE_BINDINGS:
        message_type = candidate_types.get(topic)
        if message_type is None:
            continue

        def callback(message, observed_topic=topic):
            with summary_lock:
                callback_sequences[observed_topic] += 1
                try:
                    observation = extract_candidate_observation(
                        observed_topic,
                        message,
                        callback_sequences[observed_topic],
                    )
                    summary.observe(
                        observation,
                        observed_at=max(
                            0.0,
                            monotonic() - started_at,
                        ),
                    )
                except (AttributeError, TypeError):
                    summary.mark_failure(
                        observed_topic,
                        REASON_FIELDS_MISSING,
                    )
                except ValueError as error:
                    reason = REASON_FIELDS_MISSING
                    if "finite" in str(error):
                        reason = REASON_NONFINITE_SAMPLE
                    summary.mark_failure(observed_topic, reason)
                except Exception:
                    summary.mark_failure(
                        observed_topic,
                        REASON_RUNTIME_FAILURE,
                    )

        try:
            subscriber = subscriber_factory(topic, message_type)
            subscriber.Init(callback)
            subscribers.append(subscriber)
        except Exception:
            with summary_lock:
                summary.mark_failure(
                    topic,
                    REASON_RUNTIME_FAILURE,
                )

    try:
        sleep(duration_seconds)
    except Exception:
        with summary_lock:
            for topic in CANDIDATE_BINDINGS:
                if candidate_types.get(topic) is not None:
                    summary.mark_failure(
                        topic,
                        REASON_RUNTIME_FAILURE,
                    )

    return finish()


def parse_args(args=None):
    """Parse explicit, bounded physical-state probe arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--interface",
        required=True,
        type=_nonempty_string,
    )
    parser.add_argument(
        "--duration-seconds",
        default=10.0,
        type=_positive_float,
    )
    parser.add_argument(
        "--minimum-samples",
        default=20,
        type=_positive_integer,
    )
    return parser.parse_args(args)


def resolve_channel_api(import_module=importlib.import_module):
    """Resolve Unitree channel primitives only when explicitly called."""
    module = import_module("unitree_sdk2py.core.channel")
    return (
        module.ChannelFactoryInitialize,
        module.ChannelSubscriber,
    )


def resolve_candidate_types(import_module=importlib.import_module):
    """Resolve approved message types without importing command APIs."""
    resolved = {}
    modules = {}

    for topic, (module_name, type_name) in CANDIDATE_BINDINGS.items():
        try:
            if module_name not in modules:
                modules[module_name] = import_module(module_name)
            resolved[topic] = getattr(
                modules[module_name],
                type_name,
            )
        except (AttributeError, ImportError, KeyError):
            resolved[topic] = None

    return resolved


def run_native(
    args=None,
    import_module=importlib.import_module,
    monotonic=time.monotonic,
    sleep=time.sleep,
    emit=print,
):
    """Run the explicit, time-bounded native state probe."""
    options = parse_args(args)
    candidate_types = resolve_candidate_types(
        import_module=import_module,
    )

    try:
        channel_initialize, subscriber_factory = (
            resolve_channel_api(import_module=import_module)
        )
    except Exception:
        def channel_initialize(_domain, _interface):
            raise RuntimeError("Unitree channel API unavailable")

        def subscriber_factory(_topic, _message_type):
            raise RuntimeError("Unitree channel API unavailable")

    return run_probe(
        interface=options.interface,
        duration_seconds=options.duration_seconds,
        minimum_samples=options.minimum_samples,
        candidate_types=candidate_types,
        channel_factory_initialize=channel_initialize,
        subscriber_factory=subscriber_factory,
        monotonic=monotonic,
        sleep=sleep,
        emit=emit,
    )


def main(args=None):
    """Run the manual physical-state probe console command."""
    run_native(args=args)
    return 0
