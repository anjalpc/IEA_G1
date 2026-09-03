"""Hardware-independent automatic G1 handshake sequence."""

import time
import threading


class SafeStateError(RuntimeError):
    """Report that one or more devices failed to reach their safe state."""


class SafeStateCoordinator:
    """Open the hand and release the arm exactly once per interaction."""

    def __init__(self, arm, hand):
        self.arm = arm
        self.hand = hand
        self._condition = threading.Condition()
        self._complete = False
        self._running = False
        self._error = None

    def prepare(self):
        """Reset both backends before a newly accepted interaction."""
        with self._condition:
            if self._running:
                raise RuntimeError("safe-state cleanup is still running")
            self._complete = False
            self._error = None
        self.arm.prepare_operation()
        self.hand.prepare_operation()

    def release_once(self):
        """Open the hand, then release the arm; concurrent callers wait."""
        with self._condition:
            while self._running:
                self._condition.wait()
            if self._complete:
                if self._error is not None:
                    raise self._error
                return
            self._running = True

        errors = []
        try:
            try:
                self.hand.release()
            except Exception as exc:
                errors.append(f"hand release failed: {exc}")
            try:
                self.arm.release()
            except Exception as exc:
                errors.append(f"arm release failed: {exc}")
        finally:
            with self._condition:
                self._error = (
                    SafeStateError("; ".join(errors)) if errors else None
                )
                self._complete = True
                self._running = False
                self._condition.notify_all()

        if self._error is not None:
            raise self._error


class ExternalYieldCoordinator:
    """Open the hand exactly once while yielding arm control externally."""

    def __init__(self, hand):
        self.hand = hand
        self._condition = threading.Condition()
        self._complete = False
        self._running = False
        self._error = None

    def prepare(self):
        """Reset hand-only cleanup before a newly accepted interaction."""
        with self._condition:
            if self._running:
                raise RuntimeError("external-yield cleanup is still running")
            self._complete = False
            self._error = None

    def release_once(self):
        """Open the hand once; concurrent callers wait for the same result."""
        with self._condition:
            while self._running:
                self._condition.wait()
            if self._complete:
                if self._error is not None:
                    raise self._error
                return
            self._running = True

        error = None
        try:
            self.hand.release()
        except Exception as exc:
            error = SafeStateError(f"hand release failed: {exc}")
        finally:
            with self._condition:
                self._error = error
                self._complete = True
                self._running = False
                self._condition.notify_all()

        if error is not None:
            raise error


class AutomaticHandshake:
    """Provide independent arm-presentation and timed-grip operations."""

    def __init__(
        self,
        arm,
        hand,
        hold_seconds,
        settle_seconds,
        cancelled,
        set_state,
        sleep=time.sleep,
        monotonic=time.monotonic,
        safe_state=None,
        external_yield=None,
        yield_requested=None,
        presentation_action=27,
    ):
        self.arm = arm
        self.hand = hand
        self.hold_seconds = hold_seconds
        self.settle_seconds = settle_seconds
        self.cancelled = cancelled
        self.set_state = set_state
        self.sleep = sleep
        self.monotonic = monotonic
        self.safe_state = safe_state or SafeStateCoordinator(arm, hand)
        self.external_yield = (
            external_yield or ExternalYieldCoordinator(hand))
        self.yield_requested = yield_requested or threading.Event()
        self.presentation_action = int(presentation_action)

    def prepare(self):
        """Prepare cancellation and backends before starting a worker."""
        self.cancelled.clear()
        self.yield_requested.clear()
        try:
            self.safe_state.prepare()
            self.external_yield.prepare()
        except Exception as prepare_error:
            self.cancelled.set()
            try:
                self.safe_state.release_once()
            except Exception as release_error:
                raise SafeStateError(
                    f"preparation failed: {prepare_error}; "
                    f"safe release also failed: {release_error}"
                ) from prepare_error
            raise

    def present(self):
        """Raise the arm into the presentation posture."""
        if self._interrupted():
            return self._finish_interruption()

        self.set_state("PRESENTING")
        self._execute_arm(self.presentation_action)

        if not self._wait(self.settle_seconds):
            return self._finish_interruption()

        self.set_state("ARM_PRESENTED")
        return "arm presentation complete"

    def grip(self):
        """Curl, hold, open the hand, and lower the arm."""
        if self._interrupted():
            return self._finish_interruption()

        self.set_state("GRIPPING")
        self.hand.grip()

        if self._interrupted():
            return self._finish_interruption()

        self.set_state("HOLDING")
        if not self._wait(self.hold_seconds):
            return self._finish_interruption()

        self.set_state("RELEASING")
        self._cleanup()
        self.set_state("COMPLETE")
        return "handshake complete"

    def expire_presentation(self):
        """Safely lower a dashboard-presented arm after its timeout."""
        self.set_state("PRESENTATION_TIMEOUT")
        self.set_state("RELEASING")
        self._cleanup()
        self.set_state("COMPLETE")
        return "presentation timed out; hand opened and arm released"

    def release_presentation(self):
        """Safely lower an arm when the operator toggles presentation off."""
        self.set_state("RELEASING")
        self._cleanup()
        self.set_state("ARM_RELEASED")
        return "arm presentation released; hand opened and arm released"

    def cancel(self):
        """Request cancellation and synchronously return to a safe state."""
        self.cancelled.set()
        return self._cancel()

    def request_external_yield(self):
        """Interrupt hand work and mark this interaction as externally owned."""
        self.yield_requested.set()
        self.hand.interrupt()

    def yield_arm_control(self):
        """Neutralize only the hand and leave the externally owned arm alone."""
        self.yield_requested.set()
        self.set_state("YIELDING_TO_EXTERNAL_ARM")
        self.external_yield.release_once()
        self.set_state("EXTERNAL_ARM_CONTROL")
        return "external arm takeover detected; hand opened and arm untouched"

    def _cancel(self):
        self.set_state("RELEASING")
        self._cleanup()
        self.set_state("CANCELLED")
        return "handshake cancelled"

    def _wait(self, duration):
        deadline = self.monotonic() + duration
        while self.monotonic() < deadline:
            if self._interrupted():
                return False
            remaining = max(0.0, deadline - self.monotonic())
            self.sleep(min(0.02, remaining))
        return True

    def _cancelled(self):
        return self.cancelled.is_set()

    def _interrupted(self):
        return self.cancelled.is_set() or self.yield_requested.is_set()

    def _finish_interruption(self):
        if self.yield_requested.is_set():
            return self.yield_arm_control()
        return self._cancel()

    def _cleanup(self):
        self.safe_state.release_once()

    def _execute_arm(self, action_id):
        result = self.arm.execute(action_id)
        if result not in (None, 0):
            raise RuntimeError(f"action {action_id} returned {result}")
