"""Contract tests for the validated odometry gateway."""

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from g1_state_estimation.gateway import (
    StateValidationGateway,
)
from g1_state_estimation.health import HealthState


class FakeClock:
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
    message.header.frame_id = "pelvis"
    message.orientation.w = 1.0
    message.orientation_covariance = populated_covariance(9)
    message.angular_velocity_covariance = (
        populated_covariance(9)
    )
    message.linear_acceleration_covariance = (
        populated_covariance(9)
    )
    return message


def valid_odometry():
    message = Odometry()
    message.header.stamp.sec = 123
    message.header.frame_id = "odom"
    message.child_frame_id = "pelvis"
    message.pose.pose.orientation.w = 1.0
    message.pose.covariance = populated_covariance(36)
    message.twist.covariance = populated_covariance(36)
    return message


def gateway():
    clock = FakeClock()
    instance = StateValidationGateway(
        stale_after_seconds=0.5,
        monotonic=clock,
    )
    return instance, clock


def test_suppresses_odometry_until_imu_is_valid():
    instance, _clock = gateway()

    update = instance.observe_odometry(valid_odometry())

    assert update.odometry is None
    assert update.health.state is HealthState.INITIALIZING
    assert update.health.ready is False


def test_releases_same_odometry_when_both_streams_are_valid():
    instance, _clock = gateway()
    odometry = valid_odometry()

    instance.observe_imu(valid_imu())
    update = instance.observe_odometry(odometry)

    assert update.odometry is odometry
    assert update.health.state is HealthState.VALID
    assert update.health.ready is True


def test_invalid_imu_suppresses_otherwise_valid_odometry():
    instance, _clock = gateway()
    imu = valid_imu()
    imu.orientation.w = 2.0

    instance.observe_imu(imu)
    update = instance.observe_odometry(valid_odometry())

    assert update.odometry is None
    assert update.health.state is HealthState.INVALID
    assert update.health.reason == "STATE_INVALID"


def test_invalid_odometry_is_never_released():
    instance, _clock = gateway()
    odometry = valid_odometry()
    odometry.pose.pose.position.x = float("nan")

    instance.observe_imu(valid_imu())
    update = instance.observe_odometry(odometry)

    assert update.odometry is None
    assert update.health.state is HealthState.INVALID
    assert update.health.reason == "STATE_INVALID"


def test_stale_imu_suppresses_fresh_odometry():
    instance, clock = gateway()

    instance.observe_imu(valid_imu())
    clock.now += 0.501
    update = instance.observe_odometry(valid_odometry())

    assert update.odometry is None
    assert update.health.state is HealthState.STALE
    assert update.health.ready is False


def test_imu_refresh_does_not_republish_old_odometry():
    instance, clock = gateway()

    instance.observe_imu(valid_imu())
    instance.observe_odometry(valid_odometry())
    clock.now += 0.501

    health = instance.observe_imu(valid_imu())

    assert health.state is HealthState.STALE
    assert health.ready is False
