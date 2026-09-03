"""Backend-independent left-hand card-presentation sequence."""


class CardPresentation:
    """Grip a card, present it with the left arm, and release safely."""

    def __init__(self, arm, hand, pose, fraction, move_seconds):
        self.arm = arm
        self.hand = hand
        self.pose = str(pose)
        self.fraction = float(fraction)
        self.move_seconds = float(move_seconds)
        self.prepared = False

    def prepare(self):
        self.arm.prepare_presentation()
        try:
            self.hand.prepare_operation()
        except Exception:
            self.arm.release_presentation()
            raise
        self.prepared = True

    def present(self):
        if not self.prepared:
            raise RuntimeError("card presentation is not prepared")
        self.hand.grip()
        return self.arm.present(
            self.pose, self.fraction, self.move_seconds)

    def release(self):
        """Open the hand before returning/releasing arm authority."""
        errors = []
        try:
            self.hand.release()
        except Exception as exc:
            errors.append(f"left hand: {exc}")
        try:
            self.arm.release_presentation()
        except Exception as exc:
            errors.append(f"left arm: {exc}")
        self.prepared = False
        if errors:
            raise RuntimeError("; ".join(errors))

    def cancel(self):
        self.hand.interrupt()
        self.arm.interrupt()
        self.release()
