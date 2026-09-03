"""Adapt generic bridge clients to the configured handshake behavior."""


class HandshakeHand:
    """Configured handshake behavior over a generic hand client."""

    def __init__(self, client, hand, grip_target, grip_ramp, release_ramp):
        self.client = client
        self.hand = hand
        self.grip_target = list(grip_target)
        self.grip_ramp = float(grip_ramp)
        self.release_ramp = float(release_ramp)

    def prepare_operation(self):
        self.client.prepare_operation()

    def interrupt(self):
        self.client.interrupt()

    def grip(self):
        return self.client.command_position(
            self.hand, self.grip_target, self.grip_ramp)

    def release(self):
        return self.client.neutral(self.hand, self.release_ramp)
