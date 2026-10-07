"""Pure validation and protocol for physical waist telemetry."""

import json
import math
import time


JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
JOINT_INDICES = (12, 13, 14)
OUTPUT_PREFIX = "JOINT_STATE="

_REQUIRED_MOTOR_COUNT = 29
_PROTOCOL_FIELDS = {
    "tick",
    "positions",
}


def _validated_tick(tick):
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise TypeError("joint-state tick must be an integer")
    return tick


def _validated_waist_positions(positions):
    if not isinstance(positions, (list, tuple)):
        raise TypeError(
            "joint-state positions must be a sequence"
        )
    if len(positions) != len(JOINT_NAMES):
        raise ValueError(
            "joint-state positions must contain three values"
        )

    values = tuple(
        float(value)
        for value in positions
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "joint-state positions must be finite"
        )
    return values


def normalize_sample(tick, motor_positions):
    """Return one validated minimal waist-state sample."""
    tick = _validated_tick(tick)

    if len(motor_positions) < _REQUIRED_MOTOR_COUNT:
        raise ValueError(
            "LowState motor_state must contain at least 29 entries"
        )

    positions = _validated_waist_positions([
        motor_positions[index]
        for index in JOINT_INDICES
    ])

    return {
        "tick": tick,
        "positions": list(positions),
    }


def _validated_protocol_sample(sample):
    if not isinstance(sample, dict):
        raise TypeError(
            "joint-state payload must be a JSON object"
        )
    if set(sample) != _PROTOCOL_FIELDS:
        raise ValueError(
            "joint-state payload fields are invalid"
        )

    return {
        "tick": _validated_tick(sample["tick"]),
        "positions": list(
            _validated_waist_positions(
                sample["positions"]
            )
        ),
    }


def encode_sample(sample):
    """Encode one validated sample for the parent monitor."""
    payload = _validated_protocol_sample(sample)
    return OUTPUT_PREFIX + json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_sample(line):
    """Decode one prefixed helper line, or ignore unrelated output."""
    stripped = str(line).strip()
    if not stripped.startswith(OUTPUT_PREFIX):
        return None

    payload = json.loads(
        stripped[len(OUTPUT_PREFIX):]
    )
    return _validated_protocol_sample(payload)


class JointStateGate:
    """Deduplicate and rate-limit advancing physical telemetry."""

    def __init__(
        self,
        period_seconds=0.02,
        monotonic=time.monotonic,
    ):
        period_seconds = float(period_seconds)
        if (
            not math.isfinite(period_seconds)
            or period_seconds <= 0.0
        ):
            raise ValueError(
                "joint-state period must be finite and positive"
            )

        if monotonic is None or not callable(monotonic):
            raise TypeError(
                "monotonic clock must be callable"
            )

        self.period_seconds = period_seconds
        self.monotonic = monotonic
        self._last_tick = None
        self._last_publish_time = None

    def offer(self, tick, motor_positions):
        """Return an accepted sample or None when intentionally skipped."""
        sample = normalize_sample(
            tick,
            motor_positions,
        )

        if sample["tick"] == self._last_tick:
            return None
        self._last_tick = sample["tick"]

        now = float(self.monotonic())
        if not math.isfinite(now):
            raise ValueError(
                "monotonic time must be finite"
            )

        if (
            self._last_publish_time is not None
            and now - self._last_publish_time
            < self.period_seconds
        ):
            return None

        self._last_publish_time = now
        return sample
