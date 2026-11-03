"""Safety-contract tests for physical odometry authority."""

import math

from nav_msgs.msg import Odometry

from g1_state_estimation.physical_authority import (
    PhysicalAuthorityBoundary,
    PhysicalAuthorityLimits,
)


class FakeClock:
    """Deterministic receipt-time clock."""

    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def odometry(seconds, *, x=0.0, yaw=0.0):
    """Build one staged physical-body odometry sample."""

    message = Odometry()
    message.header.stamp.sec = int(seconds)
    message.header.stamp.nanosec = int(
        round((seconds - int(seconds)) * 1e9)
    )
    message.header.frame_id = "odom"
    message.child_frame_id = "pelvis"
    message.pose.pose.position.x = x
    message.pose.pose.position.z = 0.785
    message.pose.pose.orientation.z = math.sin(yaw * 0.5)
    message.pose.pose.orientation.w = math.cos(yaw * 0.5)
    return message


def boundary(*, warmup_samples=3):
    """Create one strict physical authority boundary."""

    clock = FakeClock()
    instance = PhysicalAuthorityBoundary(
        limits=PhysicalAuthorityLimits(
            warmup_samples=warmup_samples,
            stale_after_seconds=0.25,
            maximum_translation_step=0.15,
            maximum_rotation_step=math.radians(15.0),
        ),
        monotonic=clock,
    )
    return instance, clock


def warm(instance):
    """Supply two valid warmup samples."""

    assert instance.observe(odometry(100.0)).odometry is None
    assert instance.observe(
        odometry(100.1, x=0.01)
    ).odometry is None


def test_releases_only_after_warmup_and_preserves_unknown_covariance():
    instance, _clock = boundary()
    warm(instance)

    source = odometry(100.2, x=0.02)
    result = instance.observe(source)

    assert result.odometry is source
    assert list(result.odometry.pose.covariance) == [0.0] * 36
    assert list(result.odometry.twist.covariance) == [0.0] * 36


def test_rejects_wrong_frames_and_nonunit_orientation():
    instance, _clock = boundary(warmup_samples=1)

    wrong_parent = odometry(100.0)
    wrong_parent.header.frame_id = "map"
    assert instance.observe(wrong_parent).odometry is None

    wrong_child = odometry(100.1)
    wrong_child.child_frame_id = "base_footprint"
    assert instance.observe(wrong_child).odometry is None

    bad_quaternion = odometry(100.2)
    bad_quaternion.pose.pose.orientation.w = 2.0
    assert instance.observe(bad_quaternion).odometry is None


def test_rejects_duplicate_or_reversing_timestamps():
    instance, _clock = boundary(warmup_samples=1)

    assert instance.observe(
        odometry(100.0)
    ).odometry is not None
    assert instance.observe(
        odometry(100.0)
    ).odometry is None
    assert instance.observe(
        odometry(99.9)
    ).odometry is None


def test_rejects_unbounded_translation_and_rotation_steps():
    instance, _clock = boundary(warmup_samples=1)

    assert instance.observe(
        odometry(100.0)
    ).odometry is not None

    translation_jump = odometry(100.1, x=0.151)
    assert instance.observe(translation_jump).odometry is None

    rotation_jump = odometry(
        100.2,
        yaw=math.radians(15.1),
    )
    assert instance.observe(rotation_jump).odometry is None


def test_snapshot_fails_closed_after_stale_timeout():
    instance, clock = boundary(warmup_samples=1)

    assert instance.observe(
        odometry(100.0)
    ).odometry is not None
    assert instance.snapshot().ready is True

    clock.now += 0.251

    assert instance.snapshot().ready is False
