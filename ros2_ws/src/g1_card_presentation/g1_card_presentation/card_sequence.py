"""Backend-independent left-hand card-presentation primitives."""

import threading
import time


class CardSafeStateError(RuntimeError):
    """Report failure to open the hand or release arm authority."""


class CardSafeStateCoordinator:
    """Coordinate idempotent hand and arm safety cleanup."""

    def __init__(
        self,
        arm,
        hand,
        set_state,
        release_dwell_s=0.5,
    ):
        self.arm = arm
        self.hand = hand
        self.set_state = set_state
        self.release_dwell_s = float(release_dwell_s)

        self.condition = threading.Condition()
        self.running = False
        self.released = {
            "hand": False,
            "arm": False,
        }
        self.errors = {
            "hand": None,
            "arm": None,
        }

    @property
    def complete(self):
        """Return whether cleanup attempted both subsystems."""
        with self.condition:
            return all(self.released.values())

    def reset(self):
        """Reset cleanup state for a new interaction."""
        with self.condition:
            if self.running:
                raise RuntimeError(
                    "card cleanup is still running"
                )

            self.released = {
                "hand": False,
                "arm": False,
            }
            self.errors = {
                "hand": None,
                "arm": None,
            }

    def release_once(
        self,
        release_hand=True,
        release_arm=True,
    ):
        """Release each requested subsystem at most once."""
        with self.condition:
            while self.running:
                self.condition.wait()

            do_hand = (
                release_hand
                and not self.released["hand"]
            )
            do_arm = (
                release_arm
                and not self.released["arm"]
            )

            if not do_hand and not do_arm:
                errors = [
                    self.errors[name]
                    for name, requested in (
                        ("hand", release_hand),
                        ("arm", release_arm),
                    )
                    if requested and self.errors[name]
                ]
                if errors:
                    raise CardSafeStateError(
                        "; ".join(errors)
                    )
                return

            self.running = True

        new_errors = {}

        try:
            if do_hand:
                self.set_state("RELEASING_CARD")
                try:
                    self.hand.release()
                except Exception as exc:
                    new_errors["hand"] = (
                        f"left hand release failed: {exc}"
                    )

            if (
                do_hand
                and do_arm
                and self.release_dwell_s > 0.0
            ):
                time.sleep(self.release_dwell_s)

            if do_arm:
                self.set_state("RETRACTING")
                try:
                    self.arm.release_presentation()
                except Exception as exc:
                    new_errors["arm"] = (
                        f"left arm release failed: {exc}"
                    )

        finally:
            with self.condition:
                if do_hand:
                    self.released["hand"] = True
                    self.errors["hand"] = new_errors.get(
                        "hand"
                    )

                if do_arm:
                    self.released["arm"] = True
                    self.errors["arm"] = new_errors.get(
                        "arm"
                    )

                self.running = False
                self.condition.notify_all()

        errors = [
            error
            for error in new_errors.values()
            if error
        ]
        if errors:
            raise CardSafeStateError("; ".join(errors))


class CardPresentation:
    """Independent hand and arm presentation primitives."""

    def __init__(
        self,
        arm,
        hand,
        pose,
        fraction,
        execution_timeout_s,
        set_state=lambda _state: None,
        release_dwell_s=0.5,
        handover_dwell_s=1.5,
        safe_state=None,
    ):
        self.arm = arm
        self.hand = hand

        self.pose = str(pose)
        self.fraction = float(fraction)
        self.execution_timeout_s = float(execution_timeout_s)

        self.set_state = set_state
        self.handover_dwell_s = float(handover_dwell_s)

        self.stop_requested = threading.Event()

        self.hand_prepared = False
        self.hand_gripped = False

        self.arm_prepared = False
        self.arm_presented = False

        self.interaction_active = False

        self.safe_state = (
            safe_state
            or CardSafeStateCoordinator(
                arm,
                hand,
                set_state,
                release_dwell_s,
            )
        )

    def _ensure_interaction(self):
        """Prepare shared interaction state when starting from idle."""

        if self.interaction_active:
            return

        self.stop_requested.clear()
        self.safe_state.reset()
        self.interaction_active = True

    def _maybe_finish_interaction(self):
        """Return internal coordination state to idle when fully safe."""

        if self.hand_gripped:
            return

        if self.arm_prepared or self.arm_presented:
            return

        self.hand_prepared = False
        self.interaction_active = False

    def grip(self):
        """Close the left hand without moving either arm."""

        self._ensure_interaction()

        if self.hand_gripped:
            return "card is already gripped"

        if not self.hand_prepared:
            # Mark conservatively before the blocking call so cleanup
            # cannot miss a partially acquired hand operation.
            self.hand_prepared = True
            self.hand.prepare_operation()

        if self.stop_requested.is_set():
            return "card grip stopped"

        self.set_state("GRIPPING_CARD")

        result = self.hand.grip()

        if self.stop_requested.is_set():
            return "card grip stopped"

        self.hand_gripped = True

        self.set_state("CARD_GRIPPED")

        return result

    def release_card(self):
        """Open the hand and retract afterward when presented."""
        if not self.hand_gripped:
            return "card hand is already open"

        if not self.hand_prepared:
            raise RuntimeError(
                "left hand is not prepared"
            )

        self.set_state("RELEASING_CARD")
        hand_result = self.hand.release()

        self.hand_gripped = False
        self.set_state("CARD_RELEASED")

        arm_result = None

        if (
            self.arm_presented
            and not self.stop_requested.is_set()
        ):
            arm_result = self.retract()
        else:
            self._maybe_finish_interaction()

        return {
            "hand": hand_result,
            "arm": arm_result,
        }

    def handover(self):
        """Open an offered card, pause for collection, then retract."""
        if not self.hand_gripped or not self.arm_presented:
            raise RuntimeError("card is not gripped and presented")

        self.set_state("HANDOVER_OPENING")
        hand_result = self.hand.release()
        self.hand_gripped = False
        self.set_state("HANDOVER_RELEASED")

        if self.stop_requested.wait(self.handover_dwell_s):
            return "card handover stopped"

        return {
            "hand": hand_result,
            "arm": self.retract(),
        }

    def present(self):
        """Present the arm without changing the hand."""

        self._ensure_interaction()

        if self.arm_presented:
            return "card arm is already presented"

        if not self.arm_prepared:
            self.set_state("PREPARING")

            # Mark conservatively before the blocking call so cleanup
            # cannot miss partially acquired arm authority.
            self.arm_prepared = True
            self.arm.prepare_presentation()

        if self.stop_requested.is_set():
            return "card presentation stopped"

        self.set_state("PRESENTING")

        result = self.arm.present(
            self.pose,
            self.fraction,
            self.execution_timeout_s,
        )

        if self.stop_requested.is_set():
            return "card presentation stopped"

        self.arm_presented = True

        self.set_state("CARD_PRESENTED")

        return result

    def retract(self):
        """Retract the presentation arm without changing the hand."""

        if not self.arm_prepared:
            return "card arm is already down"

        self.set_state("RETRACTING")

        result = self.arm.release_presentation()

        self.arm_prepared = False
        self.arm_presented = False

        self.set_state("ARM_DOWN")

        self._maybe_finish_interaction()

        return result

    def cancel(self):
        """Safely release every currently active subsystem."""

        self.interrupt_active()

        self.safe_state.release_once(
            release_hand=self.hand_prepared,
            release_arm=self.arm_prepared,
        )

        self._clear_runtime_state()

        self.set_state("CANCELLED")

        return "card presentation cancelled"

    def request_stop(self):
        """Request interruption of a running operation."""
        self.stop_requested.set()

    def interrupt_active(self):
        """Interrupt only subsystems that may own hardware."""
        self.request_stop()

        active_backends = []

        if self.hand_prepared:
            active_backends.append(self.hand)

        if self.arm_prepared:
            active_backends.append(self.arm)

        for backend in active_backends:
            try:
                backend.interrupt()
            except Exception:
                pass

    def ensure_safe(self):
        """Perform coordinated cleanup after an unexpected failure."""

        self.stop_requested.set()

        self.safe_state.release_once(
            release_hand=self.hand_prepared,
            release_arm=self.arm_prepared,
        )

        self._clear_runtime_state()

    def _clear_runtime_state(self):
        self.hand_prepared = False
        self.hand_gripped = False

        self.arm_prepared = False
        self.arm_presented = False

        self.interaction_active = False
