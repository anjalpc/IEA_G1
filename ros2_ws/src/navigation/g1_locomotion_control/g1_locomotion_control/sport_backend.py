"""Hardware-free boundary for a future Unitree Sport adapter."""

import math
from typing import Protocol


class SportBackendError(RuntimeError):
    """Stable error raised at the Sport backend boundary."""


class SportClientPort(Protocol):
    """Project-owned neutral interface for a native Sport client."""

    def move(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        """Forward one planar velocity command."""

    def stop_move(self):
        """Request stopped locomotion."""

    def healthy(self):
        """Return whether the native client is available."""


class SportBackend:
    """Adapt a neutral Sport client to the locomotion backend contract."""

    def __init__(self, client):
        self._client = client
        self.active = False
        self.stopped = True
        self.faulted = False

    def activate(self):
        """Activate or recover without commanding motion."""
        if self.active and not self.faulted:
            return

        try:
            client_healthy = bool(self._client.healthy())
        except Exception as exc:
            self.faulted = True
            raise SportBackendError(
                "SPORT_CLIENT_UNAVAILABLE"
            ) from exc

        if not client_healthy:
            self.faulted = True
            raise SportBackendError(
                "SPORT_CLIENT_UNAVAILABLE"
            )

        was_active = self.active
        self.active = True
        if not was_active:
            self.stopped = True
        self.faulted = False

    def command_velocity(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        """Forward an already-bounded planar command exactly."""
        if not self.active:
            raise SportBackendError(
                "SPORT_BACKEND_INACTIVE"
            )
        if self.faulted:
            raise SportBackendError(
                "SPORT_CLIENT_UNAVAILABLE"
            )

        try:
            command = (
                float(forward_mps),
                float(lateral_mps),
                float(yaw_rate_rps),
            )
        except (TypeError, ValueError) as exc:
            raise SportBackendError(
                "SPORT_MOVE_FAILED"
            ) from exc

        if not all(math.isfinite(value) for value in command):
            raise SportBackendError(
                "SPORT_MOVE_FAILED"
            )

        try:
            self._client.move(*command)
        except Exception as exc:
            self.faulted = True
            self.stopped = False
            raise SportBackendError(
                "SPORT_MOVE_FAILED"
            ) from exc

        self.stopped = command == (0.0, 0.0, 0.0)

    def stop(self):
        """Request stop whenever the backend is active."""
        if not self.active:
            return

        try:
            self._client.stop_move()
        except Exception as exc:
            self.faulted = True
            self.stopped = False
            raise SportBackendError(
                "SPORT_STOP_FAILED"
            ) from exc

        self.stopped = True

    def healthy(self):
        """Return fail-closed backend and client health."""
        if not self.active or self.faulted:
            return False
        try:
            return bool(self._client.healthy())
        except Exception:
            return False

    def deactivate(self):
        """Deactivate locally only after a confirmed stop."""
        if not self.active:
            return
        if not self.stopped:
            raise SportBackendError(
                "SPORT_BACKEND_NOT_STOPPED"
            )
        self.active = False
