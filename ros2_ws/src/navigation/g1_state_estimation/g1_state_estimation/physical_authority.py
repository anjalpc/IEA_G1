"""Strict qualification for staged physical-body odometry."""

from dataclasses import dataclass
import math


EXPECTED_PARENT_FRAME = "odom"
EXPECTED_CHILD_FRAME = "pelvis"


@dataclass(frozen=True)
class PhysicalAuthorityLimits:
    """Limits for physical odometry authority."""

    warmup_samples: int = 20
    stale_after_seconds: float = 0.25
    maximum_translation_step: float = 0.15
    maximum_rotation_step: float = math.radians(15.0)

    def __post_init__(self):
        if self.warmup_samples < 1:
            raise ValueError("warmup samples must be positive")
        for name in (
            "stale_after_seconds",
            "maximum_translation_step",
            "maximum_rotation_step",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )


@dataclass(frozen=True)
class PhysicalAuthorityResult:
    """Result from one staged odometry observation."""

    odometry: object | None
    reason: str


@dataclass(frozen=True)
class PhysicalAuthoritySnapshot:
    """Current readiness without granting ROS authority."""

    ready: bool
    reason: str


def _stamp_seconds(message):
    return (
        message.header.stamp.sec
        + message.header.stamp.nanosec * 1e-9
    )


def _pose_values(message):
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    return (
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )


def _twist_values(message):
    linear = message.twist.twist.linear
    angular = message.twist.twist.angular
    return (
        linear.x,
        linear.y,
        linear.z,
        angular.x,
        angular.y,
        angular.z,
    )


def _rotation_step(previous, current):
    dot = abs(sum(
        left * right
        for left, right in zip(previous, current)
    ))
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


class PhysicalAuthorityBoundary:
    """Fail closed before canonical odometry and TF publication."""

    def __init__(self, *, limits=None, monotonic):
        self.limits = limits or PhysicalAuthorityLimits()
        self.monotonic = monotonic
        self._healthy_samples = 0
        self._last_receipt = None
        self._last_stamp = None
        self._last_position = None
        self._last_orientation = None

    def _reject(self, reason):
        self._healthy_samples = 0
        return PhysicalAuthorityResult(
            odometry=None,
            reason=reason,
        )

    def observe(self, message):
        """Observe one staged message and release only when qualified."""

        if message.header.frame_id != EXPECTED_PARENT_FRAME:
            return self._reject("PARENT_FRAME_INVALID")
        if message.child_frame_id != EXPECTED_CHILD_FRAME:
            return self._reject("CHILD_FRAME_INVALID")

        stamp = _stamp_seconds(message)
        values = _pose_values(message) + _twist_values(message)
        if not math.isfinite(stamp) or not all(
            math.isfinite(value)
            for value in values
        ):
            return self._reject("NONFINITE_STATE")

        if (
            self._last_stamp is not None
            and stamp <= self._last_stamp
        ):
            return self._reject("TIMESTAMP_NOT_ADVANCING")

        position = values[:3]
        orientation = values[3:7]
        norm = math.sqrt(sum(
            component * component
            for component in orientation
        ))
        if abs(norm - 1.0) > 0.01:
            return self._reject("ORIENTATION_INVALID")

        if self._last_position is not None:
            translation_step = math.sqrt(sum(
                (current - previous) ** 2
                for previous, current in zip(
                    self._last_position,
                    position,
                )
            ))
            if (
                translation_step
                > self.limits.maximum_translation_step
            ):
                return self._reject("TRANSLATION_STEP_EXCEEDED")

        if self._last_orientation is not None:
            rotation_step = _rotation_step(
                self._last_orientation,
                orientation,
            )
            if (
                rotation_step
                > self.limits.maximum_rotation_step
            ):
                return self._reject("ROTATION_STEP_EXCEEDED")

        self._last_stamp = stamp
        self._last_position = position
        self._last_orientation = orientation
        self._last_receipt = self.monotonic()
        self._healthy_samples += 1

        if self._healthy_samples < self.limits.warmup_samples:
            return PhysicalAuthorityResult(
                odometry=None,
                reason="WARMING_UP",
            )

        return PhysicalAuthorityResult(
            odometry=message,
            reason="READY",
        )

    def snapshot(self):
        """Return current readiness, failing closed when stale."""

        if self._last_receipt is None:
            return PhysicalAuthoritySnapshot(
                ready=False,
                reason="WAITING_FOR_ODOMETRY",
            )
        if (
            self.monotonic() - self._last_receipt
            > self.limits.stale_after_seconds
        ):
            return PhysicalAuthoritySnapshot(
                ready=False,
                reason="ODOMETRY_STALE",
            )
        if self._healthy_samples < self.limits.warmup_samples:
            return PhysicalAuthoritySnapshot(
                ready=False,
                reason="WARMING_UP",
            )
        return PhysicalAuthoritySnapshot(
            ready=True,
            reason="READY",
        )
