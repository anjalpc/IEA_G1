"""Thread-safe ownership classification for Unitree arm actions."""

import threading
import time


class ArmOwnership:
    """Classify native arm feedback as idle, locally owned, or external."""

    IDLE = "IDLE"
    OWNED = "OWNED"
    EXTERNAL = "EXTERNAL"
    UNAVAILABLE = "UNAVAILABLE"

    def __init__(self, stale_timeout_s=2.0, idle_stable_s=0.5,
                 monotonic=time.monotonic):
        if stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s must be positive")
        if idle_stable_s < 0.0:
            raise ValueError("idle_stable_s must not be negative")
        self.stale_timeout_s = float(stale_timeout_s)
        self.idle_stable_s = float(idle_stable_s)
        self.monotonic = monotonic
        self._lock = threading.Lock()
        self._expected = None
        self._last = None
        self._last_update = None
        self._idle_since = None
        self._monitor_available = True

    def mark_monitor_available(self):
        """Record that the native arm-state reader is running."""
        with self._lock:
            self._monitor_available = True

    def mark_monitor_unavailable(self):
        """Fail closed after the native arm-state reader exits."""
        with self._lock:
            self._monitor_available = False

    def expect(self, action_id):
        """Record the next action ID intentionally requested by this bridge."""
        with self._lock:
            self._expected = int(action_id)

    def clear_expectation(self, action_id=None):
        """Clear an expectation, optionally only when its ID still matches."""
        with self._lock:
            if action_id is None or self._expected == int(action_id):
                self._expected = None

    def update(self, action_id, name="", holding=False, timestamp=None):
        """Store one normalized native `/arm/action/state` sample."""
        now = self.monotonic() if timestamp is None else float(timestamp)
        sample = {
            "id": int(action_id),
            "name": str(name),
            "holding": bool(holding),
        }
        with self._lock:
            self._last = sample
            self._last_update = now
            if sample["id"] == 0:
                if self._idle_since is None:
                    self._idle_since = now
                self._expected = None
            else:
                self._idle_since = None

    def snapshot(self, timestamp=None):
        """Return the current fail-safe ownership classification."""
        now = self.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            sample = dict(self._last) if self._last is not None else None
            updated = self._last_update
            idle_since = self._idle_since
            expected = self._expected
            monitor_available = self._monitor_available

        if not monitor_available:
            return self._result(self.UNAVAILABLE, sample, expected, False)
        if sample is None or updated is None:
            # `/arm/action/state` is event-driven. On an idle robot it emits
            # nothing until an action begins, so healthy startup silence is
            # the normal available state rather than missing feedback.
            return self._result(self.IDLE, sample, expected, True)
        if sample["id"] == 0:
            # Idle is a durable rest state, not a heartbeat: the native
            # publisher stops sending once idle is reached, so idle must not
            # expire just because no further sample has arrived.
            stable = (idle_since is not None
                      and now - idle_since >= self.idle_stable_s)
            status = self.IDLE if stable else self.UNAVAILABLE
            return self._result(status, sample, expected, stable)
        if now - updated > self.stale_timeout_s:
            return self._result(self.UNAVAILABLE, sample, expected, False)
        if expected is not None and sample["id"] == expected:
            return self._result(self.OWNED, sample, expected, False)
        return self._result(self.EXTERNAL, sample, expected, False)

    @staticmethod
    def _result(status, sample, expected, idle_stable):
        values = sample or {"id": None, "name": "", "holding": False}
        return {
            "status": status,
            "id": values["id"],
            "name": values["name"],
            "holding": values["holding"],
            "expected_id": expected,
            "idle_stable": bool(idle_stable),
        }

    def require_available(self):
        """Reject a new command unless no active arm action is observed."""
        state = self.snapshot()
        if state["status"] != self.IDLE:
            raise RuntimeError(self.block_reason(state))
        return state

    @staticmethod
    def block_reason(state):
        """Build a concise operator-facing reason for a blocked command."""
        if state["status"] == ArmOwnership.EXTERNAL:
            return (
                "external Unitree arm action is active: "
                f"{state['id']} {state['name']}".rstrip()
            )
        if state["status"] == ArmOwnership.OWNED:
            return f"Unitree arm action {state['id']} is already active"
        return "Unitree arm state is unavailable or not stably idle"
