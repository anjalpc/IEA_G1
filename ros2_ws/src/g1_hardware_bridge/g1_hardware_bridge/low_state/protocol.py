"""Pure validation and protocol for physical LowState telemetry."""

import json
import math
import time


JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
JOINT_INDICES = (12, 13, 14)
OUTPUT_PREFIX = "LOW_STATE="

_REQUIRED_MOTOR_COUNT = 29
_PROTOCOL_FIELDS = {
    "tick",
    "positions",
    "orientation_wxyz",
    "angular_velocity_xyz",
    "linear_acceleration_xyz",
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


def _validated_vector(name, values, expected_length):
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values"
        )

    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} values must be finite")
    return normalized


def _validated_orientation_wxyz(values):
    orientation = _validated_vector(
        "orientation_wxyz",
        values,
        4,
    )
    norm = math.sqrt(
        sum(value * value for value in orientation)
    )
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise ValueError(
            "orientation_wxyz must have unit length"
        )
    return orientation


def normalize_atomic_sample(
    *,
    tick,
    motor_positions,
    orientation_wxyz,
    angular_velocity_xyz,
    linear_acceleration_xyz,
):
    """Return one validated atomic physical LowState sample."""
    sample = normalize_sample(
        tick,
        motor_positions,
    )
    sample["orientation_wxyz"] = list(
        _validated_orientation_wxyz(
            orientation_wxyz
        )
    )
    sample["angular_velocity_xyz"] = list(
        _validated_vector(
            "angular_velocity_xyz",
            angular_velocity_xyz,
            3,
        )
    )
    sample["linear_acceleration_xyz"] = list(
        _validated_vector(
            "linear_acceleration_xyz",
            linear_acceleration_xyz,
            3,
        )
    )
    return sample


def _validated_protocol_sample(sample):
    if not isinstance(sample, dict):
        raise TypeError(
            "LowState payload must be a JSON object"
        )
    if set(sample) != _PROTOCOL_FIELDS:
        raise ValueError(
            "LowState payload fields are invalid"
        )

    return {
        "tick": _validated_tick(sample["tick"]),
        "positions": list(
            _validated_waist_positions(
                sample["positions"]
            )
        ),
        "orientation_wxyz": list(
            _validated_orientation_wxyz(
                sample["orientation_wxyz"]
            )
        ),
        "angular_velocity_xyz": list(
            _validated_vector(
                "angular_velocity_xyz",
                sample["angular_velocity_xyz"],
                3,
            )
        ),
        "linear_acceleration_xyz": list(
            _validated_vector(
                "linear_acceleration_xyz",
                sample["linear_acceleration_xyz"],
                3,
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


class LowStateGate:
    """Validate, deduplicate, and rate-limit physical LowState."""

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
                "LowState period must be finite and positive"
            )

        if monotonic is None or not callable(monotonic):
            raise TypeError(
                "monotonic clock must be callable"
            )

        self.period_seconds = period_seconds
        self.monotonic = monotonic
        self._last_tick = None
        self._last_publish_time = None

    def offer(
        self,
        *,
        tick,
        motor_positions,
        orientation_wxyz,
        angular_velocity_xyz,
        linear_acceleration_xyz,
    ):
        """Return an accepted atomic sample or None when skipped."""
        sample = normalize_atomic_sample(
            tick=tick,
            motor_positions=motor_positions,
            orientation_wxyz=orientation_wxyz,
            angular_velocity_xyz=angular_velocity_xyz,
            linear_acceleration_xyz=linear_acceleration_xyz,
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
