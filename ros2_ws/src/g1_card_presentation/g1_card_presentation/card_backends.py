"""Adapt generic bridge clients to the configured card side."""

from g1_card_presentation.motion import card_arm_motion, load_card_motion


class CardArm:
    """Keep card names and calibration out of the hardware bridge."""

    def __init__(self, client, side, profile=None):
        self.client = client
        if side not in ("left", "right"):
            raise ValueError("card side must be 'left' or 'right'")
        self.side = side
        self.profile = load_card_motion() if profile is None else profile
        # Validate the complete recipe before a workflow can begin.
        card_arm_motion(self.profile, self.side)

    def prepare_presentation(self):
        return self.client.prepare_motion()

    def present(self, execution_timeout_s):
        return self.client.execute_motion(
            card_arm_motion(self.profile, self.side), execution_timeout_s)

    def release_presentation(self):
        return self.client.release_motion()

    def interrupt(self):
        return self.client.interrupt()


class CardHand:
    """Configured card-hand behavior over a generic hand client."""

    def __init__(
        self,
        client,
        hand,
        target,
        grip_ramp,
        release_ramp,
    ):
        self.client = client
        if hand not in ("left", "right"):
            raise ValueError("card hand must be 'left' or 'right'")
        self.hand = hand
        self.target = list(target)
        self.grip_ramp = float(grip_ramp)
        self.release_ramp = float(release_ramp)

    def prepare_operation(self):
        self.client.prepare_operation()

    def interrupt(self):
        self.client.interrupt()

    def grip(self):
        return self.client.command_position(
            self.hand,
            self.target,
            self.grip_ramp,
        )

    def release(self):
        return self.client.neutral(
            self.hand,
            self.release_ramp,
        )
