"""Pure validation and sanitization for physical state observations."""

import json
import math


APPROVED_TOPICS = (
    "rt/lowstate",
    "rt/secondary_imu",
    "rt/sportmodestate",
)
OUTPUT_PREFIX = "PHYSICAL_STATE_PROBE="

REASON_VALID = "VALID"
REASON_NO_SAMPLES = "NO_SAMPLES"
REASON_TYPE_UNAVAILABLE = "TYPE_UNAVAILABLE"
REASON_FIELDS_MISSING = "FIELDS_MISSING"
REASON_NONFINITE_SAMPLE = "NONFINITE_SAMPLE"
REASON_SOURCE_NOT_ADVANCING = "SOURCE_NOT_ADVANCING"
REASON_INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
REASON_RUNTIME_FAILURE = "RUNTIME_FAILURE"


def _validated_progress(progress):
    if isinstance(progress, bool) or not isinstance(progress, int):
        raise TypeError("state-probe progress must be an integer")
    if progress < 0:
        raise ValueError("state-probe progress must be nonnegative")
    return progress


def _validated_fields(fields):
    if not isinstance(fields, dict):
        raise TypeError("state-probe fields must be a mapping")
    if not fields:
        raise ValueError("state-probe fields must not be empty")

    lengths = {}
    for name, values in fields.items():
        if not isinstance(name, str) or not name:
            raise ValueError("state-probe field names must be nonempty")
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, (list, tuple))
        ):
            raise TypeError(
                "state-probe field values must be sequences"
            )
        if not values:
            raise ValueError(
                "state-probe field values must not be empty"
            )

        numeric_values = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(
                "state-probe field values must be finite"
            )
        lengths[name] = len(numeric_values)

    return {
        name: lengths[name]
        for name in sorted(lengths)
    }


def normalize_observation(topic, progress, fields):
    """Return sanitized metadata for one physical-state observation."""
    if topic not in APPROVED_TOPICS:
        raise ValueError("state-probe topic is not approved")

    return {
        "topic": topic,
        "progress": _validated_progress(progress),
        "field_lengths": _validated_fields(fields),
        "finite": True,
    }


_TOPIC_FIELDS = {
    "available",
    "sample_count",
    "cadence_hz",
    "progress_advanced",
    "field_lengths",
    "reason",
}
_SUMMARY_FIELDS = {
    "schema_version",
    "requested_duration_seconds",
    "actual_elapsed_seconds",
    "topics",
}


def _validated_positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _validated_nonnegative_time(value, label):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} time must be finite and nonnegative")
    return value


class ProbeSummary:
    """Aggregate only sanitized metadata from candidate state streams."""

    def __init__(self, minimum_samples):
        self.minimum_samples = _validated_positive_integer(
            minimum_samples,
            "minimum samples",
        )
        self._topics = {
            topic: {
                "available": True,
                "sample_count": 0,
                "first_time": None,
                "last_time": None,
                "last_progress": None,
                "progress_advanced": False,
                "field_lengths": None,
                "fields_consistent": True,
                "failure_reason": None,
            }
            for topic in APPROVED_TOPICS
        }

    def mark_unavailable(self, topic):
        """Mark one approved candidate message type as unavailable."""
        if topic not in APPROVED_TOPICS:
            raise ValueError("state-probe topic is not approved")
        state = self._topics[topic]
        state["available"] = False

    def mark_failure(self, topic, reason):
        """Record one sanitized failure reason for an approved topic."""
        if topic not in APPROVED_TOPICS:
            raise ValueError("state-probe topic is not approved")
        if reason not in {
            REASON_FIELDS_MISSING,
            REASON_NONFINITE_SAMPLE,
            REASON_RUNTIME_FAILURE,
        }:
            raise ValueError("state-probe failure reason is invalid")
        self._topics[topic]["failure_reason"] = reason

    def observe(self, observation, observed_at):
        """Consume sanitized observation metadata without retaining values."""
        required = {
            "topic",
            "progress",
            "field_lengths",
            "finite",
        }
        if not isinstance(observation, dict):
            raise TypeError("state-probe observation must be a mapping")
        if set(observation) != required:
            raise ValueError("state-probe observation fields are invalid")

        topic = observation["topic"]
        if topic not in APPROVED_TOPICS:
            raise ValueError("state-probe topic is not approved")
        if observation["finite"] is not True:
            raise ValueError("state-probe observation must be finite")

        progress = _validated_progress(observation["progress"])
        observed_at = _validated_nonnegative_time(
            observed_at,
            "observation",
        )
        field_lengths = dict(observation["field_lengths"])

        state = self._topics[topic]
        if not state["available"]:
            raise ValueError(
                "state-probe unavailable topic cannot accept observations"
            )

        if state["field_lengths"] is None:
            state["field_lengths"] = field_lengths
        elif state["field_lengths"] != field_lengths:
            state["fields_consistent"] = False

        if (
            state["last_progress"] is not None
            and progress != state["last_progress"]
        ):
            state["progress_advanced"] = True

        state["sample_count"] += 1
        state["last_progress"] = progress
        if state["first_time"] is None:
            state["first_time"] = observed_at
        state["last_time"] = observed_at

    def _topic_result(self, topic):
        state = self._topics[topic]
        count = state["sample_count"]

        cadence_hz = None
        if (
            count >= 2
            and state["last_time"] > state["first_time"]
        ):
            cadence_hz = round(
                (count - 1)
                / (state["last_time"] - state["first_time"]),
                6,
            )

        if not state["available"]:
            reason = REASON_TYPE_UNAVAILABLE
        elif state["failure_reason"] is not None:
            reason = state["failure_reason"]
        elif count == 0:
            reason = REASON_NO_SAMPLES
        elif not state["fields_consistent"]:
            reason = REASON_FIELDS_MISSING
        elif count < self.minimum_samples:
            reason = REASON_INSUFFICIENT_SAMPLES
        elif not state["progress_advanced"] and count >= 2:
            reason = REASON_SOURCE_NOT_ADVANCING
        else:
            reason = REASON_VALID

        return {
            "available": state["available"],
            "sample_count": count,
            "cadence_hz": cadence_hz,
            "progress_advanced": state["progress_advanced"],
            "field_lengths": state["field_lengths"] or {},
            "reason": reason,
        }

    def finalize(
        self,
        requested_duration_seconds,
        actual_elapsed_seconds,
    ):
        """Return one complete sanitized probe summary."""
        requested = _validated_nonnegative_time(
            requested_duration_seconds,
            "requested duration",
        )
        actual = _validated_nonnegative_time(
            actual_elapsed_seconds,
            "actual elapsed",
        )
        if requested == 0.0:
            raise ValueError(
                "requested duration time must be positive"
            )

        return {
            "schema_version": 1,
            "requested_duration_seconds": requested,
            "actual_elapsed_seconds": actual,
            "topics": {
                topic: self._topic_result(topic)
                for topic in APPROVED_TOPICS
            },
        }


def _validated_summary(summary):
    if not isinstance(summary, dict):
        raise TypeError("state-probe summary must be a mapping")
    if set(summary) != _SUMMARY_FIELDS:
        raise ValueError("state-probe summary fields are invalid")
    if summary["schema_version"] != 1:
        raise ValueError("state-probe schema version is invalid")

    topics = summary["topics"]
    if not isinstance(topics, dict):
        raise TypeError("state-probe topics must be a mapping")
    if set(topics) != set(APPROVED_TOPICS):
        raise ValueError("state-probe topic fields are invalid")

    for topic in APPROVED_TOPICS:
        result = topics[topic]
        if not isinstance(result, dict):
            raise TypeError("state-probe topic result must be a mapping")
        if set(result) != _TOPIC_FIELDS:
            raise ValueError("state-probe topic result fields are invalid")

    return summary


def encode_summary(summary):
    """Encode one validated sanitized result for external inspection."""
    payload = _validated_summary(summary)
    return OUTPUT_PREFIX + json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
