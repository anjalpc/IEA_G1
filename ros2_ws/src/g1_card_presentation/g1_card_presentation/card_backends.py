"""Adapt generic bridge clients to the left-card application."""


class CardHand:
    """Configured left-hand behavior over a generic hand client."""

    def __init__(self, client, target, grip_ramp, release_ramp):
        self.client = client
        self.target = list(target)
        self.grip_ramp = float(grip_ramp)
        self.release_ramp = float(release_ramp)

    def prepare_operation(self):
        self.client.prepare_operation()

    def interrupt(self):
        self.client.interrupt()

    def grip(self):
        return self.client.command_position(
            "left", self.target, self.grip_ramp)

    def release(self):
        return self.client.neutral("left", self.release_ramp)
