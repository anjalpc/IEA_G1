"""Contract-first tests for state-estimation input validation."""

import math

import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from g1_state_estimation.validation import (
    validate_imu,
    validate_odometry,
)


def populated_covariance(size):
    covariance = [0.0] * size
    width = 3 if size == 9 else 6
    for index in range(width):
        covariance[index * width + index] = 0.01
    return covariance


def valid_imu():
    message = Imu()
    message.header.stamp.sec = 123
    message.header.frame_id = "imu_link"
    message.orientation.w = 1.0
    message.orientation_covariance = populated_covariance(9)
    message.angular_velocity_covariance = populated_covariance(9)
    message.linear_acceleration_covariance = populated_covariance(9)
    return message


def valid_odometry():
    message = Odometry()
    message.header.stamp.sec = 123
    message.header.frame_id = "odom"
    message.child_frame_id = "base_link"
    message.pose.pose.orientation.w = 1.0
    message.pose.covariance = populated_covariance(36)
    message.twist.covariance = populated_covariance(36)
    return message


def test_accepts_well_formed_imu():
    result = validate_imu(valid_imu())

    assert result.valid is True
    assert result.reason == "STATE_VALID"


def test_accepts_well_formed_odometry():
    result = validate_odometry(valid_odometry())

    assert result.valid is True
    assert result.reason == "STATE_VALID"


def test_rejects_unset_imu_timestamp():
    message = valid_imu()
    message.header.stamp.sec = 0

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "TIMESTAMP_REQUIRED"


def test_rejects_unset_odometry_timestamp():
    message = valid_odometry()
    message.header.stamp.sec = 0

    result = validate_odometry(message)

    assert result.valid is False
    assert result.reason == "TIMESTAMP_REQUIRED"


@pytest.mark.parametrize(
    "sec,nanosec",
    [
        (-1, 0),
        (1, 1_000_000_000),
    ],
)
def test_rejects_invalid_imu_timestamp(sec, nanosec):
    message = valid_imu()
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = nanosec

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "INVALID_TIMESTAMP"


@pytest.mark.parametrize(
    "sec,nanosec",
    [
        (-1, 0),
        (1, 1_000_000_000),
    ],
)
def test_rejects_invalid_odometry_timestamp(sec, nanosec):
    message = valid_odometry()
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = nanosec

    result = validate_odometry(message)

    assert result.valid is False
    assert result.reason == "INVALID_TIMESTAMP"


def test_rejects_missing_imu_frame():
    message = valid_imu()
    message.header.frame_id = "   "

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "FRAME_ID_REQUIRED"


@pytest.mark.parametrize(
    "field",
    ["header.frame_id", "child_frame_id"],
)
def test_rejects_missing_odometry_frames(field):
    message = valid_odometry()

    target = message
    parts = field.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], "")

    result = validate_odometry(message)

    assert result.valid is False
    assert result.reason == "FRAME_ID_REQUIRED"


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("orientation", "x", math.nan),
        ("angular_velocity", "z", math.inf),
        ("linear_acceleration", "x", -math.inf),
    ],
)
def test_rejects_nonfinite_imu_state(target, field, value):
    message = valid_imu()
    setattr(getattr(message, target), field, value)

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "NONFINITE_STATE"


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("pose.pose.position", "x", math.nan),
        ("pose.pose.orientation", "y", math.inf),
        ("twist.twist.linear", "x", -math.inf),
    ],
)
def test_rejects_nonfinite_odometry_state(target, field, value):
    message = valid_odometry()

    value_target = message
    for part in target.split("."):
        value_target = getattr(value_target, part)
    setattr(value_target, field, value)

    result = validate_odometry(message)

    assert result.valid is False
    assert result.reason == "NONFINITE_STATE"


@pytest.mark.parametrize(
    "components",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 2.0),
    ],
)
def test_rejects_invalid_imu_quaternion(components):
    message = valid_imu()
    (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) = components

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "INVALID_QUATERNION"


@pytest.mark.parametrize(
    "components",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 2.0),
    ],
)
def test_rejects_invalid_odometry_quaternion(components):
    message = valid_odometry()
    (
        message.pose.pose.orientation.x,
        message.pose.pose.orientation.y,
        message.pose.pose.orientation.z,
        message.pose.pose.orientation.w,
    ) = components

    result = validate_odometry(message)

    assert result.valid is False
    assert result.reason == "INVALID_QUATERNION"


@pytest.mark.parametrize(
    "field",
    [
        "orientation_covariance",
        "angular_velocity_covariance",
        "linear_acceleration_covariance",
    ],
)
def test_rejects_missing_imu_covariance(field):
    message = valid_imu()
    setattr(message, field, [0.0] * 9)

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "COVARIANCE_REQUIRED"


@pytest.mark.parametrize(
    "field",
    ["pose.covariance", "twist.covariance"],
)
def test_rejects_missing_odometry_covariance(field):
    message = valid_odometry()

    target = message
    parts = field.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], [0.0] * 36)

    result = validate_odometry(message)

    assert result.valid is False
    assert result.reason == "COVARIANCE_REQUIRED"


@pytest.mark.parametrize(
    "field",
    [
        "orientation_covariance",
        "angular_velocity_covariance",
        "linear_acceleration_covariance",
    ],
)
def test_rejects_nonfinite_imu_covariance(field):
    message = valid_imu()
    covariance = list(getattr(message, field))
    covariance[0] = math.nan
    setattr(message, field, covariance)

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "NONFINITE_STATE"


@pytest.mark.parametrize(
    "field",
    ["pose.covariance", "twist.covariance"],
)
def test_rejects_nonfinite_odometry_covariance(field):
    message = valid_odometry()

    target = message
    parts = field.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)

    covariance = list(getattr(target, parts[-1]))
    covariance[0] = math.inf
    setattr(target, parts[-1], covariance)

    result = validate_odometry(message)

    assert result.valid is False
    assert result.reason == "NONFINITE_STATE"


def test_rejects_unavailable_imu_covariance():
    message = valid_imu()
    covariance = list(message.orientation_covariance)
    covariance[0] = -1.0
    message.orientation_covariance = covariance

    result = validate_imu(message)

    assert result.valid is False
    assert result.reason == "COVARIANCE_REQUIRED"
