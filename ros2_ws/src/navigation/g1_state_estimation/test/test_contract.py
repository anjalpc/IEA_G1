"""Public API and validation-to-health contract tests."""

import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import g1_state_estimation as api
from g1_state_estimation.health import StateHealthMonitor


PUBLIC_API = {
    "HealthSnapshot",
    "HealthState",
    "ObservationQuality",
    "StateHealthMonitor",
    "StateValidationGateway",
    "ValidatedOdometryUpdate",
    "ValidationResult",
    "validate_imu",
    "validate_odometry",
}


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


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


def monitor():
    return StateHealthMonitor(
        stale_after_seconds=0.5,
        monotonic=FakeClock(),
    )


def test_package_exports_only_the_intentional_public_contract():
    assert set(api.__all__) == PUBLIC_API

    for name in PUBLIC_API:
        assert hasattr(api, name)


def test_validated_messages_can_make_navigation_ready():
    instance = monitor()

    instance.observe_imu(api.validate_imu(valid_imu()))
    instance.observe_odometry(
        api.validate_odometry(valid_odometry())
    )
    snapshot = instance.snapshot()

    assert snapshot.state is api.HealthState.VALID
    assert snapshot.ready is True
    assert snapshot.reason == "STATE_VALID"


def test_raw_ros_messages_cannot_bypass_validation():
    instance = monitor()

    with pytest.raises(ValueError, match="validation result"):
        instance.observe_imu(valid_imu())

    with pytest.raises(ValueError, match="validation result"):
        instance.observe_odometry(valid_odometry())


def test_invalid_validated_message_fails_health_closed():
    instance = monitor()
    message = valid_imu()
    message.header.frame_id = ""

    instance.observe_imu(api.validate_imu(message))
    instance.observe_odometry(
        api.validate_odometry(valid_odometry())
    )
    snapshot = instance.snapshot()

    assert snapshot.state is api.HealthState.INVALID
    assert snapshot.ready is False
    assert snapshot.reason == "STATE_INVALID"
