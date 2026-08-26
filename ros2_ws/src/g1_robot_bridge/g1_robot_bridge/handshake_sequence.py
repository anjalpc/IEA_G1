"""Hardware-independent sequencing for a guarded handshake."""

import time


SHAKE_HAND = 27
RELEASE_ARM = 99


class HandshakeSequence:
    """Coordinate arm and hand backends with cleanup on every exit path."""

    def __init__(self, arm, hand, duration, cancelled, enabled,
                 set_state, sleep=time.sleep, monotonic=time.monotonic):
        self.arm = arm
        self.hand = hand
        self.duration = duration
        self.cancelled = cancelled
        self.enabled = enabled
        self.set_state = set_state
        self.sleep = sleep
        self.monotonic = monotonic

    def run(self):
        self.set_state("PRESENTING")
        self.arm.execute(SHAKE_HAND)
        if self._cancelled():
            self._cleanup()
            self.set_state("CANCELLED")
            return

        self.set_state("GRIPPING")
        self.hand.grip()
        deadline = self.monotonic() + self.duration
        while self.monotonic() < deadline:
            if self._cancelled():
                self._cleanup()
                self.set_state("CANCELLED")
                return
            self.sleep(0.02)

        self.set_state("RELEASING")
        self._cleanup()
        self.set_state("COMPLETE")

    def _cancelled(self):
        return self.cancelled.is_set() or not self.enabled()

    def _cleanup(self):
        # Open the fingers before asking the arm action service to leave its
        # holding action. Each command is attempted even if the other fails.
        first_error = None
        try:
            self.hand.release()
        except Exception as exc:  # pragma: no cover - re-raised after arm cleanup
            first_error = exc
        try:
            self.arm.execute(RELEASE_ARM)
        except Exception:
            if first_error is None:
                raise
        if first_error is not None:
            raise first_error
