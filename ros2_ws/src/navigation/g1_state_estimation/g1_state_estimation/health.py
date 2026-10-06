"""Evaluate whether validated state is ready for navigation."""

from dataclasses import dataclass
from enum import Enum
import math


class HealthState(Enum):
    """Navigation-facing state-estimation health."""

    OFFLINE = "OFFLINE"
    INITIALIZING = "INITIALIZING"
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    INVALID = "INVALID"


class ObservationQuality(Enum):
    """Explicit quality assigned by a validated state source."""

    VALID = "VALID"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class HealthSnapshot:
    """Sanitized navigation-readiness snapshot."""

    state: HealthState
    ready: bool
    reason: str


@dataclass(frozen=True)
class _Observation:
    validation: object
    quality: ObservationQuality
    observed_at: object


class StateHealthMonitor:
    """Track the latest validated IMU and odometry observations."""

    def __init__(self, *, stale_after_seconds, monotonic):
        if (
            not math.isfinite(stale_after_seconds)
            or stale_after_seconds <= 0.0
        ):
            raise ValueError(
                "stale_after_seconds must be finite and positive"
            )

        self._stale_after_seconds = stale_after_seconds
        self._monotonic = monotonic
        self._imu = None
        self._odometry = None

    def _read_monotonic(self):
        try:
            now = self._monotonic()
            if not math.isfinite(now):
                return None
        except (TypeError, ValueError, OverflowError):
            return None

        return now

    def _validate_quality(self, quality):
        if not isinstance(quality, ObservationQuality):
            raise ValueError("unsupported observation quality")

    def observe_imu(
        self,
        validation,
        *,
        quality=ObservationQuality.VALID,
    ):
        """Record the latest IMU validation result."""

        self._validate_quality(quality)
        self._imu = _Observation(
            validation=validation,
            quality=quality,
            observed_at=self._read_monotonic(),
        )

    def observe_odometry(
        self,
        validation,
        *,
        quality=ObservationQuality.VALID,
    ):
        """Record the latest odometry validation result."""

        self._validate_quality(quality)
        self._odometry = _Observation(
            validation=validation,
            quality=quality,
            observed_at=self._read_monotonic(),
        )

    def _time_source_is_valid(self, observations, now):
        if now is None:
            return False

        for observation in observations:
            if observation.observed_at is None:
                return False
            if now < observation.observed_at:
                return False

        return True

    def _is_stale(self, observation, now):
        return (
            now - observation.observed_at
            > self._stale_after_seconds
        )

    def snapshot(self):
        """Return current navigation readiness without private details."""

        if self._imu is None and self._odometry is None:
            return HealthSnapshot(
                state=HealthState.OFFLINE,
                ready=False,
                reason="NO_STATE",
            )

        observations = tuple(
            observation
            for observation in (self._imu, self._odometry)
            if observation is not None
        )

        if any(
            not observation.validation.valid
            for observation in observations
        ):
            return HealthSnapshot(
                state=HealthState.INVALID,
                ready=False,
                reason="STATE_INVALID",
            )

        now = self._read_monotonic()
        if not self._time_source_is_valid(observations, now):
            return HealthSnapshot(
                state=HealthState.INVALID,
                ready=False,
                reason="TIME_SOURCE_INVALID",
            )

        if self._imu is None:
            return HealthSnapshot(
                state=HealthState.INITIALIZING,
                ready=False,
                reason="WAITING_FOR_IMU",
            )

        if self._odometry is None:
            return HealthSnapshot(
                state=HealthState.INITIALIZING,
                ready=False,
                reason="WAITING_FOR_ODOMETRY",
            )

        if (
            self._is_stale(self._imu, now)
            or self._is_stale(self._odometry, now)
        ):
            return HealthSnapshot(
                state=HealthState.STALE,
                ready=False,
                reason="STATE_STALE",
            )

        if any(
            observation.quality is ObservationQuality.DEGRADED
            for observation in observations
        ):
            return HealthSnapshot(
                state=HealthState.DEGRADED,
                ready=False,
                reason="STATE_DEGRADED",
            )

        return HealthSnapshot(
            state=HealthState.VALID,
            ready=True,
            reason="STATE_VALID",
        )
