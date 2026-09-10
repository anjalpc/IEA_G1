"""Adapt generic bridge clients to the left-card application."""

from g1_card_presentation.motion import card_arm_motion, load_card_motion


class CardArm:
    """Keep card names and calibration out of the hardware bridge."""

    def __init__(self, client, profile=None):
        self.client = client
        self.profile = load_card_motion() if profile is None else profile
        # Validate the complete recipe before a workflow can begin.
        card_arm_motion(self.profile, 1.0)

    def prepare_presentation(self):
        return self.client.prepare_motion()

    def present(self, pose, fraction, execution_timeout_s):
        if pose != 'left_card_presentation':
            raise ValueError('unsupported card presentation pose')
        return self.client.execute_motion(
            card_arm_motion(self.profile, fraction), execution_timeout_s)

    def release_presentation(self):
        return self.client.release_motion()

    def interrupt(self):
        return self.client.interrupt()


class CardHand:
    """Configured left-hand behavior over a generic hand client."""

    def __init__(
        self,
        client,
        target,
        grip_ramp,
        release_ramp,
    ):
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
            "left",
            self.target,
            self.grip_ramp,
        )

    def release(self):
        return self.client.neutral(
            "left",
            self.release_ramp,
        )
