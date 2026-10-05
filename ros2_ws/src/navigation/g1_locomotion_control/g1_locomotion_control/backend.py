"""Backend-neutral locomotion boundary and recording implementation."""

from typing import Protocol


class LocomotionBackend(Protocol):
    """Operations required by the locomotion arbiter."""

    def activate(self):
        """Prepare the backend without commanding motion."""

    def command_velocity(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        """Apply one validated planar velocity command."""

    def stop(self):
        """Request zero locomotion safely and idempotently."""

    def healthy(self):
        """Return whether the backend can accept commands."""

    def deactivate(self):
        """Release backend resources after a stop attempt."""


class RecordingBackend:
    """Record backend operations without touching robot hardware."""

    def __init__(self):
        self.active = False
        self.stopped = True
        self.stop_count = 0
        self.commands = []
        self.events = []
        self._healthy = True

    def activate(self):
        """Activate once without producing a motion command."""
        if self.active:
            return
        self.active = True
        self.stopped = True
        self.events.append(("activate",))

    def command_velocity(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        """Record one already-validated planar velocity."""
        if not self.active:
            raise RuntimeError("recording backend is not active")
        command = (
            float(forward_mps),
            float(lateral_mps),
            float(yaw_rate_rps),
        )
        self.commands.append(command)
        self.events.append(("command_velocity", *command))
        self.stopped = command == (0.0, 0.0, 0.0)

    def stop(self):
        """Record every safe-stop attempt."""
        self.stop_count += 1
        self.stopped = True
        self.events.append(("stop",))

    def healthy(self):
        """Return the configured hardware-free health state."""
        return self._healthy

    def set_healthy(self, healthy):
        """Set health deterministically for tests."""
        self._healthy = bool(healthy)

    def deactivate(self):
        """Deactivate once without contacting hardware."""
        if not self.active:
            return
        self.active = False
        self.stopped = True
        self.events.append(("deactivate",))
