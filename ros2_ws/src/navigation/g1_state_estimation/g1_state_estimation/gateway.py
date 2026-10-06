"""Gate raw state before navigation-facing publication."""

from dataclasses import dataclass

from .health import HealthSnapshot, StateHealthMonitor
from .validation import validate_imu, validate_odometry


@dataclass(frozen=True)
class ValidatedOdometryUpdate:
    """Health result and optional navigation-safe odometry."""

    health: HealthSnapshot
    odometry: object | None


class StateValidationGateway:
    """Release odometry only while both raw streams are healthy."""

    def __init__(self, *, stale_after_seconds, monotonic):
        self._monitor = StateHealthMonitor(
            stale_after_seconds=stale_after_seconds,
            monotonic=monotonic,
        )

    def observe_imu(self, message):
        """Validate one IMU sample and return current health."""

        self._monitor.observe_imu(validate_imu(message))
        return self._monitor.snapshot()

    def observe_odometry(self, message):
        """Validate odometry and release it only when ready."""

        self._monitor.observe_odometry(
            validate_odometry(message)
        )
        health = self._monitor.snapshot()

        return ValidatedOdometryUpdate(
            health=health,
            odometry=message if health.ready else None,
        )

    def snapshot(self):
        """Return current readiness without changing observations."""

        return self._monitor.snapshot()
