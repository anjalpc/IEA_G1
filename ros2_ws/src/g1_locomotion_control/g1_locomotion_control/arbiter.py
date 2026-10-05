"""Thread-safe, hardware-independent locomotion authority."""

from dataclasses import dataclass
import secrets
import threading
import time

from g1_robot_interfaces.locomotion import (
    OWNER_NONE,
    PlanarVelocityLimits,
    validate_lease_id,
    validate_locomotion_owner,
    validate_planar_velocity,
)


@dataclass(frozen=True)
class AcquireResult:
    """Result of a locomotion lease request."""

    granted: bool
    lease_id: str
    active_owner: int
    reason: str


@dataclass(frozen=True)
class CommandResult:
    """Result of one locomotion velocity command."""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class ReleaseResult:
    """Result of a locomotion lease release."""

    released: bool
    active_owner: int
    reason: str


@dataclass(frozen=True)
class ArbiterSnapshot:
    """Public arbiter state without lease or requester information."""

    active_owner: int
    ready: bool
    stopped: bool
    emergency_stop: bool
    command_age_seconds: float
    reason: str


class LocomotionArbiter:
    """Own one locomotion lease and serialize backend access."""

    def __init__(
        self,
        *,
        backend,
        velocity_limits,
        command_timeout_seconds,
        monotonic=time.monotonic,
        lease_factory=None,
    ):
        timeout = float(command_timeout_seconds)
        if timeout <= 0.0:
            raise ValueError(
                "command_timeout_seconds must be positive"
            )

        if not isinstance(
            velocity_limits, PlanarVelocityLimits
        ):
            raise ValueError(
                "velocity_limits must be PlanarVelocityLimits"
            )

        self._backend = backend
        self._velocity_limits = velocity_limits
        self._command_timeout_seconds = timeout
        self._monotonic = monotonic
        self._lease_factory = (
            lease_factory
            if lease_factory is not None
            else lambda: secrets.token_urlsafe(32)
        )
        self._lock = threading.RLock()
        self._active_owner = OWNER_NONE
        self._lease_id = None
        self._lease_started_at = None
        self._last_valid_command_at = None
        self._reason = "STARTING"
        self._emergency_stop = False
        self._stopped = True

        self._backend.activate()
        self._reason = "READY"

    def acquire(self, owner, requester_id):
        """Grant one exclusive opaque locomotion lease."""
        with self._lock:
            try:
                owner = validate_locomotion_owner(owner)
            except ValueError:
                return AcquireResult(
                    False,
                    "",
                    self._active_owner,
                    "INVALID_REQUEST",
                )

            if not isinstance(requester_id, str):
                return AcquireResult(
                    False,
                    "",
                    self._active_owner,
                    "INVALID_REQUEST",
                )
            if not requester_id.strip():
                return AcquireResult(
                    False,
                    "",
                    self._active_owner,
                    "INVALID_REQUEST",
                )
            if self._active_owner != OWNER_NONE:
                return AcquireResult(
                    False,
                    "",
                    self._active_owner,
                    "LEASE_ACTIVE",
                )

            lease_id = self._lease_factory()
            try:
                validate_lease_id(lease_id)
            except ValueError as exc:
                raise RuntimeError(
                    "lease generator returned an invalid token"
                ) from exc

            now = self._monotonic()
            self._active_owner = owner
            self._lease_id = lease_id
            self._lease_started_at = now
            self._last_valid_command_at = now
            self._reason = "LEASE_GRANTED"
            self._stopped = True

            return AcquireResult(
                True,
                lease_id,
                self._active_owner,
                self._reason,
            )

    def command(
        self,
        *,
        owner,
        lease_id,
        linear_x,
        linear_y,
        linear_z,
        angular_x,
        angular_y,
        angular_z,
    ):
        """Validate ownership and dispatch one planar command."""
        with self._lock:
            if owner != self._active_owner:
                return CommandResult(False, "OWNER_MISMATCH")
            if lease_id != self._lease_id:
                return CommandResult(False, "LEASE_MISMATCH")

            velocity = validate_planar_velocity(
                linear_x=linear_x,
                linear_y=linear_y,
                linear_z=linear_z,
                angular_x=angular_x,
                angular_y=angular_y,
                angular_z=angular_z,
                limits=self._velocity_limits,
            )
            self._backend.command_velocity(*velocity)
            self._last_valid_command_at = self._monotonic()
            self._reason = "COMMAND_ACCEPTED"
            self._stopped = velocity == (0.0, 0.0, 0.0)
            return CommandResult(True, self._reason)

    def release(self, owner, lease_id):
        """Stop and release only the exact active lease."""
        with self._lock:
            if owner != self._active_owner:
                return ReleaseResult(
                    False,
                    self._active_owner,
                    "OWNER_MISMATCH",
                )
            if lease_id != self._lease_id:
                return ReleaseResult(
                    False,
                    self._active_owner,
                    "LEASE_MISMATCH",
                )

            self._backend.stop()
            self._clear_lease("RELEASED")
            return ReleaseResult(
                True,
                self._active_owner,
                self._reason,
            )

    def snapshot(self):
        """Return public state without ownership secrets."""
        with self._lock:
            if self._last_valid_command_at is None:
                command_age = -1.0
            else:
                command_age = max(
                    0.0,
                    self._monotonic()
                    - self._last_valid_command_at,
                )

            return ArbiterSnapshot(
                active_owner=self._active_owner,
                ready=bool(self._backend.healthy()),
                stopped=self._stopped,
                emergency_stop=self._emergency_stop,
                command_age_seconds=command_age,
                reason=self._reason,
            )

    def _clear_lease(self, reason):
        """Clear private ownership state while holding the lock."""
        self._active_owner = OWNER_NONE
        self._lease_id = None
        self._lease_started_at = None
        self._last_valid_command_at = None
        self._reason = reason
        self._stopped = True
