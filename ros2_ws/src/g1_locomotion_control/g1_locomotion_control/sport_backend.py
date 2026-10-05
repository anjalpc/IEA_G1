"""Hardware-free boundary for a future Unitree Sport adapter."""

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
        """Activate without commanding motion."""
        if self.active:
            return
        if not self._client.healthy():
            self.faulted = True
            raise SportBackendError(
                "SPORT_CLIENT_UNAVAILABLE"
            )
        self.active = True
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
        command = (
            float(forward_mps),
            float(lateral_mps),
            float(yaw_rate_rps),
        )
        self._client.move(*command)
        self.stopped = command == (0.0, 0.0, 0.0)

    def stop(self):
        """Request stop whenever the backend is active."""
        if not self.active:
            return
        self._client.stop_move()
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
