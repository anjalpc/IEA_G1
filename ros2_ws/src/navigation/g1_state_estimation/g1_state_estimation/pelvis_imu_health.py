"""Fail-closed qualification for the physical pelvis IMU."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PelvisImuHealthLimits:
    """Limits for the pinned physical pelvis-IMU source."""

    stale_after_seconds: float
    warmup_samples: int
    minimum_acceleration_norm: float
    maximum_acceleration_norm: float
    maximum_angular_velocity_norm: float

    def __post_init__(self):
        positive = (
            self.stale_after_seconds,
            self.maximum_acceleration_norm,
            self.maximum_angular_velocity_norm,
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in positive
        ):
            raise ValueError(
                "pelvis IMU limits must be finite and positive"
            )
        if (
            not math.isfinite(self.minimum_acceleration_norm)
            or self.minimum_acceleration_norm < 0.0
            or self.minimum_acceleration_norm
            >= self.maximum_acceleration_norm
        ):
            raise ValueError(
                "pelvis IMU acceleration limits are invalid"
            )
        if (
            not isinstance(self.warmup_samples, int)
            or self.warmup_samples <= 0
        ):
            raise ValueError(
                "warmup_samples must be a positive integer"
            )


@dataclass(frozen=True)
class PelvisImuHealthSnapshot:
    """Sanitized readiness result for physical body staging."""

    ready: bool
    reason: str


class PelvisImuHealthBoundary:
    """Require live, advancing, plausible physical pelvis IMU data."""

    def __init__(self, *, limits, monotonic):
        if not isinstance(limits, PelvisImuHealthLimits):
            raise ValueError(
                "limits must be PelvisImuHealthLimits"
            )

        self._limits = limits
        self._monotonic = monotonic
        self._observed_at = None
        self._last_stamp = None
        self._valid = None
        self._warmup_count = 0

    def _now(self):
        try:
            value = self._monotonic()
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _finite_vector(values, size):
        return (
            len(values) == size
            and all(math.isfinite(value) for value in values)
        )

    @staticmethod
    def _unit_quaternion(values):
        norm = math.sqrt(
            sum(value * value for value in values)
        )
        return math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-3,
        )

    @staticmethod
    def _norm(values):
        return math.sqrt(
            sum(value * value for value in values)
        )

    @classmethod
    def _unknown_covariance(cls, values):
        return (
            cls._finite_vector(values, 9)
            and all(value == 0.0 for value in values)
        )

    def _invalidate(self):
        self._valid = False
        self._observed_at = self._now()
        self._warmup_count = 0

    def observe(
        self,
        *,
        stamp,
        frame_id,
        orientation,
        angular_velocity,
        linear_acceleration,
        orientation_covariance,
        angular_velocity_covariance,
        linear_acceleration_covariance,
    ):
        """Validate one atomic physical pelvis-IMU sample."""

        vectors_valid = (
            self._finite_vector(orientation, 4)
            and self._finite_vector(angular_velocity, 3)
            and self._finite_vector(linear_acceleration, 3)
        )
        covariances_valid = (
            self._unknown_covariance(
                orientation_covariance
            )
            and self._unknown_covariance(
                angular_velocity_covariance
            )
            and self._unknown_covariance(
                linear_acceleration_covariance
            )
        )

        well_formed = (
            math.isfinite(stamp)
            and stamp > 0.0
            and frame_id == "imu_in_pelvis"
            and vectors_valid
            and covariances_valid
        )
        if not well_formed:
            self._invalidate()
            return

        if (
            self._last_stamp is not None
            and stamp <= self._last_stamp
        ):
            self._invalidate()
            return

        acceleration_norm = self._norm(
            linear_acceleration
        )
        angular_velocity_norm = self._norm(
            angular_velocity
        )
        plausible = (
            self._unit_quaternion(orientation)
            and acceleration_norm
            >= self._limits.minimum_acceleration_norm
            and acceleration_norm
            <= self._limits.maximum_acceleration_norm
            and angular_velocity_norm
            <= self._limits.maximum_angular_velocity_norm
        )

        self._last_stamp = stamp
        if not plausible:
            self._invalidate()
            return

        self._valid = True
        self._observed_at = self._now()
        self._warmup_count += 1

    def snapshot(self):
        """Return current readiness without exposing private state."""

        if self._valid is None:
            return PelvisImuHealthSnapshot(
                False,
                "WAITING_FOR_PELVIS_IMU",
            )
        if not self._valid:
            return PelvisImuHealthSnapshot(
                False,
                "PELVIS_IMU_INVALID",
            )

        now = self._now()
        if (
            now is None
            or self._observed_at is None
            or now < self._observed_at
        ):
            return PelvisImuHealthSnapshot(
                False,
                "TIME_SOURCE_INVALID",
            )

        if (
            now - self._observed_at
            > self._limits.stale_after_seconds
        ):
            return PelvisImuHealthSnapshot(
                False,
                "PELVIS_IMU_STALE",
            )

        if self._warmup_count < self._limits.warmup_samples:
            return PelvisImuHealthSnapshot(
                False,
                "WARMING_UP",
            )

        return PelvisImuHealthSnapshot(
            True,
            "PELVIS_IMU_HEALTHY",
        )
