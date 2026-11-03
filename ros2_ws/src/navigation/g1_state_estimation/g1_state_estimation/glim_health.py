"""Fail-closed health evaluation for GLIM navigation inputs."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GlimHealthLimits:
    """Validated limits for the physical GLIM health boundary."""

    odometry_stale_after_seconds: float
    cloud_stale_after_seconds: float
    imu_stale_after_seconds: float
    warmup_samples: int
    maximum_step_translation_m: float
    maximum_step_rotation_rad: float

    def __post_init__(self):
        positive_values = (
            self.odometry_stale_after_seconds,
            self.cloud_stale_after_seconds,
            self.imu_stale_after_seconds,
            self.maximum_step_translation_m,
            self.maximum_step_rotation_rad,
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in positive_values
        ):
            raise ValueError("GLIM health limits must be finite and positive")
        if not isinstance(self.warmup_samples, int):
            raise ValueError("warmup_samples must be a positive integer")
        if self.warmup_samples <= 0:
            raise ValueError("warmup_samples must be a positive integer")


@dataclass(frozen=True)
class GlimHealthSnapshot:
    """Sanitized readiness result for navigation consumers."""

    ready: bool
    reason: str


@dataclass(frozen=True)
class _Observation:
    valid: bool
    observed_at: float | None


class GlimHealthBoundary:
    """Require live LiDAR, IMU, and plausible corrected odometry."""

    def __init__(self, *, limits, monotonic):
        if not isinstance(limits, GlimHealthLimits):
            raise ValueError("limits must be GlimHealthLimits")

        self._limits = limits
        self._monotonic = monotonic
        self._cloud = None
        self._imu = None
        self._odometry = None
        self._last_odometry_stamp = None
        self._last_position = None
        self._last_orientation = None
        self._warmup_count = 0
        self._odometry_reason = None

    def _now(self):
        try:
            value = self._monotonic()
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _valid_stamp(stamp):
        return math.isfinite(stamp) and stamp > 0.0

    def observe_cloud(self, *, stamp, frame_id):
        """Record one raw MID-360 cloud observation."""

        valid = (
            self._valid_stamp(stamp)
            and frame_id == "mid360_link"
        )
        self._cloud = _Observation(valid, self._now())

    def observe_imu(self, *, stamp, frame_id):
        """Record one raw MID-360 IMU observation."""

        valid = (
            self._valid_stamp(stamp)
            and frame_id == "livox_frame"
        )
        self._imu = _Observation(valid, self._now())

    @staticmethod
    def _finite_vector(values, expected_size):
        return (
            len(values) == expected_size
            and all(math.isfinite(value) for value in values)
        )

    @staticmethod
    def _unit_quaternion(quaternion):
        norm = math.sqrt(sum(value * value for value in quaternion))
        return math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3)

    @staticmethod
    def _translation_step(previous, current):
        return math.sqrt(
            sum(
                (current_value - previous_value) ** 2
                for previous_value, current_value
                in zip(previous, current)
            )
        )

    @staticmethod
    def _rotation_step(previous, current):
        dot = abs(sum(a * b for a, b in zip(previous, current)))
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    def _invalidate_odometry(self, reason):
        self._odometry = _Observation(False, self._now())
        self._odometry_reason = reason
        self._warmup_count = 0

    def observe_odometry(
        self,
        *,
        stamp,
        frame_id,
        child_frame_id,
        position,
        orientation,
    ):
        """Validate one optimized GLIM odometry observation."""

        well_formed = (
            self._valid_stamp(stamp)
            and frame_id == "glim_odom"
            and child_frame_id == "livox_frame"
            and self._finite_vector(position, 3)
            and self._finite_vector(orientation, 4)
            and self._unit_quaternion(orientation)
        )
        if not well_formed:
            self._invalidate_odometry("ODOMETRY_INVALID")
            return

        if (
            self._last_odometry_stamp is not None
            and stamp <= self._last_odometry_stamp
        ):
            self._invalidate_odometry("ODOMETRY_INVALID")
            return

        if self._last_position is not None:
            translation = self._translation_step(
                self._last_position,
                position,
            )
            rotation = self._rotation_step(
                self._last_orientation,
                orientation,
            )
            if (
                translation
                > self._limits.maximum_step_translation_m
                or rotation
                > self._limits.maximum_step_rotation_rad
            ):
                self._invalidate_odometry("ODOMETRY_JUMP")
                self._last_odometry_stamp = stamp
                self._last_position = tuple(position)
                self._last_orientation = tuple(orientation)
                return

        self._odometry = _Observation(True, self._now())
        self._odometry_reason = None
        self._last_odometry_stamp = stamp
        self._last_position = tuple(position)
        self._last_orientation = tuple(orientation)
        self._warmup_count += 1

    @staticmethod
    def _stale(observation, now, timeout):
        return (
            observation.observed_at is None
            or now < observation.observed_at
            or now - observation.observed_at > timeout
        )

    def snapshot(self):
        """Return current readiness without exposing private state."""

        if self._odometry is not None and not self._odometry.valid:
            return GlimHealthSnapshot(False, self._odometry_reason)

        observations = (self._cloud, self._imu, self._odometry)
        if any(observation is None for observation in observations):
            return GlimHealthSnapshot(False, "WAITING_FOR_INPUTS")

        if not self._cloud.valid or not self._imu.valid:
            return GlimHealthSnapshot(False, "INPUT_INVALID")

        now = self._now()
        if now is None:
            return GlimHealthSnapshot(False, "TIME_SOURCE_INVALID")

        stale = (
            self._stale(
                self._cloud,
                now,
                self._limits.cloud_stale_after_seconds,
            )
            or self._stale(
                self._imu,
                now,
                self._limits.imu_stale_after_seconds,
            )
            or self._stale(
                self._odometry,
                now,
                self._limits.odometry_stale_after_seconds,
            )
        )
        if stale:
            return GlimHealthSnapshot(False, "INPUT_STALE")

        if self._warmup_count < self._limits.warmup_samples:
            return GlimHealthSnapshot(False, "WARMING_UP")

        return GlimHealthSnapshot(True, "GLIM_HEALTHY")
