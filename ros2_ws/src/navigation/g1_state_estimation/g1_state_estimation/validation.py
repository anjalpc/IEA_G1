"""Validate state-estimation inputs at the navigation boundary."""

from dataclasses import dataclass
import math


STATE_VALID = "STATE_VALID"
TIMESTAMP_REQUIRED = "TIMESTAMP_REQUIRED"
INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
FRAME_ID_REQUIRED = "FRAME_ID_REQUIRED"
NONFINITE_STATE = "NONFINITE_STATE"
INVALID_QUATERNION = "INVALID_QUATERNION"
COVARIANCE_REQUIRED = "COVARIANCE_REQUIRED"

QUATERNION_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ValidationResult:
    """Sanitized result returned by every state validator."""

    valid: bool
    reason: str


def _validate_timestamp(stamp):
    if stamp.sec == 0 and stamp.nanosec == 0:
        return ValidationResult(False, TIMESTAMP_REQUIRED)

    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
        return ValidationResult(False, INVALID_TIMESTAMP)

    return None


def _has_frame_id(frame_id):
    return isinstance(frame_id, str) and bool(frame_id.strip())


def _components(vector):
    return (vector.x, vector.y, vector.z)


def _quaternion_components(quaternion):
    return (
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    )


def _all_finite(values):
    return all(math.isfinite(value) for value in values)


def _has_unit_quaternion(quaternion):
    components = _quaternion_components(quaternion)
    norm = math.sqrt(sum(value * value for value in components))
    return math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=QUATERNION_NORM_TOLERANCE,
    )


def _has_covariance(values):
    return (
        values[0] != -1.0
        and any(value != 0.0 for value in values)
    )


def _invalid(reason):
    return ValidationResult(False, reason)


def _valid():
    return ValidationResult(True, STATE_VALID)


def validate_imu(message):
    """Validate an IMU message before health evaluation."""

    timestamp_result = _validate_timestamp(message.header.stamp)
    if timestamp_result is not None:
        return timestamp_result

    if not _has_frame_id(message.header.frame_id):
        return _invalid(FRAME_ID_REQUIRED)

    state_values = (
        *_quaternion_components(message.orientation),
        *_components(message.angular_velocity),
        *_components(message.linear_acceleration),
        *message.orientation_covariance,
        *message.angular_velocity_covariance,
        *message.linear_acceleration_covariance,
    )
    if not _all_finite(state_values):
        return _invalid(NONFINITE_STATE)

    if not _has_unit_quaternion(message.orientation):
        return _invalid(INVALID_QUATERNION)

    covariance_sets = (
        message.orientation_covariance,
        message.angular_velocity_covariance,
        message.linear_acceleration_covariance,
    )
    if not all(_has_covariance(values) for values in covariance_sets):
        return _invalid(COVARIANCE_REQUIRED)

    return _valid()


def validate_odometry(message):
    """Validate an odometry message before health evaluation."""

    timestamp_result = _validate_timestamp(message.header.stamp)
    if timestamp_result is not None:
        return timestamp_result

    if not _has_frame_id(message.header.frame_id):
        return _invalid(FRAME_ID_REQUIRED)
    if not _has_frame_id(message.child_frame_id):
        return _invalid(FRAME_ID_REQUIRED)

    pose = message.pose.pose
    twist = message.twist.twist
    state_values = (
        *_components(pose.position),
        *_quaternion_components(pose.orientation),
        *_components(twist.linear),
        *_components(twist.angular),
        *message.pose.covariance,
        *message.twist.covariance,
    )
    if not _all_finite(state_values):
        return _invalid(NONFINITE_STATE)

    if not _has_unit_quaternion(pose.orientation):
        return _invalid(INVALID_QUATERNION)

    covariance_sets = (
        message.pose.covariance,
        message.twist.covariance,
    )
    if not all(_has_covariance(values) for values in covariance_sets):
        return _invalid(COVARIANCE_REQUIRED)

    return _valid()
